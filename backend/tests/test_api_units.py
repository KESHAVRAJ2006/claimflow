"""API building blocks that need no database: errors, auth, body limit, uploads, events, webhooks, settings."""

import asyncio
import io
import json
import uuid
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError
from starlette.datastructures import Headers

from app.api.problems import ProblemError
from app.core.config import Settings
from app.main import create_app
from app.services import uploads
from app.services.events import EventBroker, format_sse, replay_events
from app.services.uploads import receive_claim_document, safe_filename
from app.services.webhooks import SIGNATURE_HEADER, HttpWebhookSender, sign
from scripts.generate_policy_pdfs import build_pdf, layout_page

KEY = "unit-test-api-key"


def make_pdf(lines: list[str]) -> bytes:
    """A real, small PDF with a text layer, built with the same writer as the policy wordings."""
    return build_pdf([layout_page(lines, title="Claim form", label="Claim form")], "Claim form")


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    monkeypatch.setenv("API_KEY", KEY)
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "2048")
    with TestClient(create_app()) as test_client:
        yield test_client


# ---- problem details and auth -----------------------------------------------------------------------------------


def test_unknown_route_is_a_problem_document(client: TestClient) -> None:
    response = client.get("/api/nope", headers={"X-Request-ID": "req-1"})
    assert response.status_code == 404
    assert response.headers["content-type"] == "application/problem+json"
    body = response.json()
    assert body["type"] == "urn:claimflow:problem:not-found" and body["instance"] == "/api/nope"
    assert body["request_id"] == "req-1"


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_protected_endpoints_require_the_api_key(client: TestClient, headers: dict[str, str]) -> None:
    protected = [
        ("GET", "/api/claims"), ("GET", f"/api/claims/{uuid.uuid4()}"), ("GET", "/api/metrics"),
        ("GET", "/api/policies/search?q=theft"), ("POST", f"/api/claims/{uuid.uuid4()}/run"),
    ]  # fmt: skip
    for method, path in protected:
        response = client.request(method, path, headers=headers)
        assert response.status_code == 401, path
        assert response.json()["type"] == "urn:claimflow:problem:unauthorized"


def test_health_stays_public(client: TestClient) -> None:
    assert client.get("/api/health").status_code in (200, 503)  # 503 locally when Postgres is not running


def test_missing_server_key_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEY", "")
    with TestClient(create_app()) as unconfigured:
        response = unconfigured.get("/api/claims", headers={"X-API-Key": ""})
    assert response.status_code == 503 and response.json()["title"] == "Authentication not configured"


def test_query_validation_errors_are_listed(client: TestClient) -> None:
    response = client.get("/api/claims?page=0&sort=password", headers={"X-API-Key": KEY})
    assert response.status_code == 422
    locations = [error["location"] for error in response.json()["errors"]]
    assert ["query", "page"] in locations and ["query", "sort"] in locations


def test_oversized_body_is_rejected_before_parsing(client: TestClient) -> None:
    # MAX_UPLOAD_BYTES is 2 KB here; the middleware allows 64 KB of multipart overhead on top.
    response = client.post("/api/claims", content=b"x" * 80_000, headers={"X-API-Key": KEY})
    assert response.status_code == 413 and response.json()["type"] == "urn:claimflow:problem:payload-too-large"


def test_production_requires_a_long_api_key() -> None:
    with pytest.raises(ValidationError, match="API_KEY"):
        Settings(database_url=SecretStr("postgresql+asyncpg://a:b@h/db"), environment="production", api_key=None)
    long_key = SecretStr("k" * 32)
    Settings(database_url=SecretStr("postgresql+asyncpg://a:b@h/db"), environment="production", api_key=long_key)


# ---- uploads ----------------------------------------------------------------------------------------------------


def _upload(data: bytes, name: str = "claim.pdf") -> UploadFile:
    return UploadFile(io.BytesIO(data), filename=name, headers=Headers({"content-type": "application/pdf"}))


@pytest.mark.parametrize(
    ("name", "expected"),
    [("claim.pdf", "claim.pdf"), ("C:\\Users\\me\\scan (1).pdf", "scan _1_.pdf"), ("../../etc/passwd", "passwd"),
     ("\x00\n.pdf", "__.pdf"), ("", "document.pdf"), ("..", "document.pdf")],
)  # fmt: skip
def test_safe_filename(name: str, expected: str) -> None:
    assert safe_filename(name) == expected


async def test_valid_pdf_is_extracted_and_deleted(tmp_path: Path) -> None:
    document = await receive_claim_document(
        _upload(make_pdf(["Repair estimate", "Total 42,000"])), tmp_path, 1_000_000, 5
    )
    assert document.name == "claim.pdf" and "Repair estimate" in document.pages[0].text
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("data", "limit", "status"),
    [
        (b"MZ\x90\x00 an executable renamed to .pdf", 1_000_000, 415),  # magic bytes, not the extension
        (b"%PDF-1.4 truncated garbage", 1_000_000, 422),  # looks like a PDF, cannot be parsed
        (b"%PDF-" + b"x" * 5000, 1000, 413),  # over the size limit
    ],
)
async def test_bad_uploads_are_rejected_and_always_deleted(
    tmp_path: Path, data: bytes, limit: int, status: int
) -> None:
    with pytest.raises(ProblemError) as problem:
        await receive_claim_document(_upload(data), tmp_path, limit, 5)
    assert problem.value.status_code == status
    assert list(tmp_path.iterdir()) == []


async def test_file_is_deleted_even_when_extraction_crashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*_args: object) -> None:
        raise RuntimeError("parser bug")

    monkeypatch.setattr(uploads, "extract_pdf_file", explode)
    with pytest.raises(RuntimeError, match="parser bug"):
        await receive_claim_document(_upload(make_pdf(["x"])), tmp_path, 1_000_000, 5)
    assert list(tmp_path.iterdir()) == []


async def test_page_limit(tmp_path: Path) -> None:
    pdf = build_pdf([layout_page([f"page {n}"], title=None, label="p") for n in range(3)], "three pages")
    with pytest.raises(ProblemError, match="3 pages"):
        await receive_claim_document(_upload(pdf), tmp_path, 1_000_000, 2)


# ---- events -------------------------------------------------------------------------------------------------------


async def test_subscriber_gets_history_then_live_events_then_stops() -> None:
    broker, claim_id = EventBroker(), uuid.uuid4()
    broker.open(claim_id, uuid.uuid4())
    broker.publish(claim_id, {"type": "run_queued"})
    received: list[dict[str, object]] = []

    async def consume() -> None:
        async for event in broker.subscribe(claim_id):
            received.append(event)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    broker.publish(claim_id, {"type": "node_started", "node": "intake"})
    broker.close(claim_id)
    await asyncio.wait_for(task, 1)
    assert [(e["id"], e["type"]) for e in received] == [(1, "run_queued"), (2, "node_started")]


async def test_reconnect_skips_events_already_received() -> None:
    broker, claim_id = EventBroker(), uuid.uuid4()
    broker.open(claim_id, uuid.uuid4())
    for kind in ("run_queued", "node_started", "node_finished"):
        broker.publish(claim_id, {"type": kind})
    broker.close(claim_id)
    assert [e["id"] async for e in broker.subscribe(claim_id, after_id=2)] == [3]


def test_sse_frame_format() -> None:
    frame = format_sse({"id": 7, "type": "tool_call_finished", "result_summary": "line one\nline two"})
    lines = frame.split("\n")
    assert lines[:2] == ["id: 7", "event: tool_call_finished"] and frame.endswith("\n\n")
    assert json.loads(lines[2].removeprefix("data: "))["result_summary"] == "line one\nline two"  # escaped, one line


def test_replay_rebuilds_nested_tool_events() -> None:
    runs = [
        {"agent_name": "intake", "round": 1, "output": {}, "latency_ms": 5, "created_at": "t"},
        {"agent_name": "investigator", "round": 1, "latency_ms": 9, "created_at": "t",
         "output": {"tool_calls": [{"call_id": "T1", "tool": "get_policy_status"}]}},
    ]  # fmt: skip
    events = replay_events(runs, {"type": "run_completed"})
    assert [e["type"] for e in events] == [
        "node_started", "node_finished", "node_started", "tool_call_started", "tool_call_finished", "node_finished",
        "run_completed",
    ]  # fmt: skip
    assert [e["id"] for e in events] == list(range(1, 8)) and all(e["replayed"] for e in events)


# ---- webhooks ---------------------------------------------------------------------------------------------------


async def test_webhook_is_signed_over_the_exact_body() -> None:
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(lambda request: seen.append(request) or httpx.Response(200))
    sender = HttpWebhookSender("https://n8n.test/hook", "s3cret", httpx.AsyncClient(transport=transport))
    await sender.send("claim.decided", {"claim_number": "CLM-2026-000001"})
    (request,) = seen
    assert request.headers[SIGNATURE_HEADER] == sign(request.content, "s3cret")
    assert json.loads(request.content)["event"] == "claim.decided"
    await sender.aclose()


async def test_webhook_failure_never_raises() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(500))
    sender = HttpWebhookSender("https://n8n.test/hook", None, httpx.AsyncClient(transport=transport))
    await sender.send("claim.triaged", {})  # logged, not raised
    await HttpWebhookSender(None, None).send("claim.triaged", {})  # not configured: no request at all
    await sender.aclose()


class _ConnectedRequest:
    """Stands in for starlette's Request: the client never reports a disconnect on its own."""

    async def is_disconnected(self) -> bool:
        return False


async def test_client_leaving_mid_run_unsubscribes_cleanly() -> None:
    # A reviewer navigating away while a run is live: the server cancels the response task while the stream is
    # waiting for the next event. Closing must neither raise ("asynchronous generator is already running") nor
    # leave the subscriber registered with the broker.
    from app.api.routes.claims import _sse

    broker = EventBroker()
    claim_id = uuid.uuid4()
    broker.open(claim_id, uuid.uuid4())
    broker.publish(claim_id, {"type": "run_queued"})
    stream = _sse(broker.subscribe(claim_id), _ConnectedRequest())  # type: ignore[arg-type]
    assert (await anext(stream)).startswith("retry:")
    assert "run_queued" in await anext(stream)
    assert len(broker._channels[claim_id].subscribers) == 1  # noqa: SLF001 — the leak is what's under test

    waiting = asyncio.ensure_future(anext(stream))  # blocks: no further event is published
    await asyncio.sleep(0.05)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    await stream.aclose()
    assert broker._channels[claim_id].subscribers == set()  # noqa: SLF001

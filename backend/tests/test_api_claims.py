"""The claims API end to end, against a throwaway seeded database, with the graph driven by a scripted model.

Covers submission (validation, upload handling), the background run and its SSE stream (live, replayed and
resumed), the persisted trace, human decisions with audit and webhooks, re-runs, listing, metrics and search.
"""

import itertools
import json
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from qdrant_client import AsyncQdrantClient

from app.agents.graph import AgentDependencies, build_claim_graph
from app.agents.llm import LlmClient
from app.agents.loaders import rule_context_loader
from app.main import create_app
from app.retrieval.retriever import PolicyRetriever
from app.services.events import EventBroker
from app.services.triage import TriageRunner
from scripts.generate_policy_pdfs import OUTPUT_DIR
from tests.agent_fakes import COVERAGE_TEXT, Brain, ScriptedChatModel, stub_tools
from tests.conftest import ScratchDatabase
from tests.fakes import FakeEmbedder
from tests.test_api_units import make_pdf

pytestmark = pytest.mark.integration

KEY = "integration-test-api-key"
AUTH = {"X-API-Key": KEY}
PLAN = [[("get_policy_status", {"policy_number": "MOT-2025-000011"})],
        [("get_coverage_section", {"product_type": "motor", "incident_type": "collision"})]]  # fmt: skip
# Unique amounts: resubmitting the same amount and date on one policy would (correctly) trigger R07, the
# duplicate-claim rule, and escalate every claim after the first.
_amounts = itertools.count(20_001)
PDF = make_pdf(["Motor claim - repair estimate", "Rear bumper and tail lamp replaced", "Total: 20,000.00"])


class RecordingWebhooks:
    """Captures webhook events instead of sending them."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def send(self, event: str, payload: dict[str, Any]) -> None:
        self.events.append((event, payload))


@pytest.fixture
def api(test_database: ScratchDatabase, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    """The app on the test database, with the triage graph driven by a scripted model and stub tools."""
    monkeypatch.setenv("DATABASE_URL", test_database.owner_url)
    monkeypatch.setenv("TOOLS_DATABASE_URL", test_database.tools_url)
    monkeypatch.setenv("API_KEY", KEY)
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    with TestClient(create_app()) as client:
        state = client.app.state  # type: ignore[attr-defined]
        deps = AgentDependencies(
            llm=LlmClient([ScriptedChatModel(brain=Brain(default_plan=PLAN))]),
            tools=stub_tools(),
            # The real loader over the read-only login: the rules run on the claim the API actually stored.
            load_rule_context=rule_context_loader(state.tools_database),
        )
        state.webhooks = RecordingWebhooks()
        state.triage_runner = TriageRunner(
            build_claim_graph(deps), state.session_factory, state.event_broker, state.webhooks, max_concurrent=2
        )
        yield client


def _edge_claim(client: TestClient, sequence: int) -> dict[str, Any]:
    """A seeded edge-case claim (CLM-<year>-9000NN) from the list endpoint."""
    items: list[dict[str, Any]] = []
    page, total = 1, 1
    while (page - 1) * 100 < total:
        # Same sort on every page, so pages neither overlap nor skip rows.
        body = client.get(
            "/api/claims", params={"page_size": 100, "page": page, "sort": "created_at"}, headers=AUTH
        ).json()
        items += body["items"]
        page, total = page + 1, body["total"]
    return next(item for item in items if item["claim_number"].endswith(f"-9{sequence:05d}"))


def _submit(
    client: TestClient, policy_number: str, amount: str | None = None, pdf: bytes = PDF, **overrides: Any
) -> Any:
    amount = amount or f"{next(_amounts)}.00"
    form = {
        "policy_number": policy_number,
        "incident_type": "collision",
        "incident_date": (datetime.now(UTC).date() - timedelta(days=10)).isoformat(),
        "claimed_amount": amount,
        "description": "Rear-ended at a traffic signal; bumper and tail lamp damaged.",
    } | overrides
    return client.post(
        "/api/claims", data={"claim": json.dumps(form)}, files={"document": ("estimate.pdf", pdf, "application/pdf")},
        headers=AUTH,
    )  # fmt: skip


def _policy_of(client: TestClient, claim: dict[str, Any]) -> str:
    return str(claim["policy_number"])


def _clean_motor_policy(client: TestClient) -> str:
    # The R01 edge-case customer: motor policy started 200 days ago, 500,000 sum insured, KYC verified, one claim.
    return _policy_of(client, _edge_claim(client, 1))


def _wait(client: TestClient, claim_id: str, timeout_s: float = 30) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        detail = client.get(f"/api/claims/{claim_id}", headers=AUTH).json()
        if detail["status"] not in ("submitted", "processing"):
            return detail
        time.sleep(0.1)
    raise AssertionError(f"claim {claim_id} did not finish triage")


def _events(client: TestClient, claim_id: str, headers: dict[str, str] | None = None) -> list[dict[str, Any]]:
    events = []
    with client.stream("GET", f"/api/claims/{claim_id}/stream", headers=AUTH | (headers or {})) as response:
        assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line.removeprefix("data: ")))
    return events


# ---- submission, run, stream, detail ---------------------------------------------------------------------------


def test_submitted_claim_is_triaged_streamed_and_persisted(api: TestClient, tmp_path: Path) -> None:
    response = _submit(api, _clean_motor_policy(api))
    assert response.status_code == 202, response.text
    accepted = response.json()
    assert accepted["triage"] == "queued" and accepted["stream_url"] == f"/api/claims/{accepted['claim_id']}/stream"
    assert list((tmp_path / "uploads").iterdir()) == []  # the PDF is gone before the response was even sent

    events = _events(api, accepted["claim_id"])
    kinds = [(e["type"], e.get("node") or e.get("tool")) for e in events]
    assert kinds[0] == ("run_queued", None) and kinds[-1][0] == "run_completed"
    assert [e["id"] for e in events] == list(range(1, len(events) + 1))
    start = kinds.index(("node_started", "investigator"))
    assert kinds[start + 1] == ("tool_call_started", "get_policy_status")
    nodes = [node for kind, node in kinds if kind == "node_finished"]
    assert nodes == ["intake", "investigator", "rules", "decision", "reflection", "route_final"]

    detail = _wait(api, accepted["claim_id"])
    assert detail["status"] == "awaiting_review" and detail["recommended_outcome"] == "approve"
    assert detail["document_filename"] == "estimate.pdf" and detail["extracted_fields"]["source"] == "form_and_document"
    assert [run["agent_name"] for run in detail["runs"]] == nodes
    assert [call["tool"] for call in detail["tool_call_log"]] == ["get_policy_status", "get_coverage_section"]
    assert all({"args", "result_summary", "latency_ms"} <= call.keys() for call in detail["tool_call_log"])
    assert detail["evidence"][0]["text"] == COVERAGE_TEXT
    assert detail["recommendation"]["citations"][0]["label"] == "Motor_Policy.pdf p.2"
    assert detail["risk_report"]["hard_blocks"] == [] and detail["risk_report"]["risk_score"] < 70
    assert [a["action"] for a in detail["audit"]] == ["claim.submitted", "claim.triage_started", "claim.triaged"]
    assert "not an automated determination" in detail["disclaimer"]
    assert api.app.state.webhooks.events[-1][0] == "claim.triaged"  # type: ignore[attr-defined]


def test_stream_is_replayed_from_the_database_and_resumable(api: TestClient) -> None:
    claim_id = _submit(api, _clean_motor_policy(api)).json()["claim_id"]
    live = _events(api, claim_id)
    api.app.state.event_broker = EventBroker()  # type: ignore[attr-defined]  # as after a restart
    replayed = _events(api, claim_id)
    assert all(event["replayed"] for event in replayed) and replayed[-1]["type"] == "run_completed"
    assert [e["type"] for e in replayed if e["type"].startswith(("node", "tool"))] == [
        e["type"] for e in live if e["type"].startswith(("node", "tool"))
    ]
    resumed = _events(api, claim_id, {"Last-Event-ID": "3"})
    assert resumed[0]["id"] == 4


def test_seeded_claim_can_be_run_and_the_rules_decide(api: TestClient) -> None:
    lapsed = _edge_claim(api, 2)  # R02: incident after the policy lapsed
    assert api.post(f"/api/claims/{lapsed['id']}/run", headers=AUTH).status_code == 202
    detail = _wait(api, lapsed["id"])
    # The scripted agent says "covered, 0.9"; the R02 hard block still rejects. No agent can override a rule.
    assert detail["recommendation"]["covered"] is True
    assert detail["recommended_outcome"] == "reject" and detail["recommendation"]["hard_blocks"] == ["R02"]
    rerun = api.post(f"/api/claims/{lapsed['id']}/run", headers=AUTH)
    assert rerun.status_code == 409 and rerun.json()["type"] == "urn:claimflow:problem:claim-not-rerunnable"


def test_run_without_llm_is_unavailable(api: TestClient) -> None:
    api.app.state.triage_runner = None  # type: ignore[attr-defined]
    reason = "No LLM key is configured. Set GROQ_API_KEY or GOOGLE_API_KEY in .env."
    api.app.state.triage_disabled_reason = reason  # type: ignore[attr-defined]
    response = api.post(f"/api/claims/{_edge_claim(api, 3)['id']}/run", headers=AUTH)
    # The 503 names the missing piece recorded at startup, not a guess.
    assert response.status_code == 503 and response.json()["detail"] == reason
    stored = _submit(api, _clean_motor_policy(api))
    assert stored.status_code == 202 and stored.json()["triage"] == "unavailable"
    events = _events(api, stored.json()["claim_id"])
    assert [e["type"] for e in events] == ["run_unavailable"]


# ---- submission validation --------------------------------------------------------------------------------------


def _claim_count(client: TestClient) -> int:
    return int(client.get("/api/claims", params={"page_size": 1}, headers=AUTH).json()["total"])


@pytest.mark.parametrize(
    ("kwargs", "status", "problem"),
    [
        ({"pdf": b"MZ\x90\x00 renamed.exe"}, 415, "not-a-pdf"),
        ({"amount": "12.345"}, 422, "validation-error"),
        ({"policy_number": "MOT-1999-000000"}, 422, "unknown-policy"),
        ({"incident_type": "surgery"}, 422, "incident-type-not-covered-by-product"),
    ],
)
def test_invalid_submissions_store_nothing_and_leave_no_file(
    api: TestClient, tmp_path: Path, kwargs: dict[str, Any], status: int, problem: str
) -> None:
    before = _claim_count(api)
    policy = kwargs.pop("policy_number", _clean_motor_policy(api))
    response = _submit(api, policy, **kwargs)
    assert response.status_code == status, response.text
    assert response.json()["type"] == f"urn:claimflow:problem:{problem}"
    assert _claim_count(api) == before
    uploads = tmp_path / "uploads"
    assert not uploads.exists() or list(uploads.iterdir()) == []


def test_bad_claim_json_points_at_the_field(api: TestClient) -> None:
    response = _submit(api, _clean_motor_policy(api), amount="-5")
    assert response.json()["errors"][0]["location"] == ["body", "claim", "claimed_amount"]


# ---- human decisions --------------------------------------------------------------------------------------------


def _triaged(client: TestClient, amount: str | None = None) -> dict[str, Any]:
    return _wait(client, _submit(client, _clean_motor_policy(client), amount=amount).json()["claim_id"])


def _decide(client: TestClient, claim_id: str, **body: Any) -> Any:
    return client.post(f"/api/claims/{claim_id}/decision", json={"reviewer": "priya.nair"} | body, headers=AUTH)


def test_accepting_the_recommendation_is_audited_and_notified(api: TestClient) -> None:
    claim = _triaged(api)
    response = _decide(api, claim["id"], action="approve")
    assert response.status_code == 200
    assert response.json() | {"decided_at": None} == {
        "claim_id": claim["id"], "status": "approved", "final_outcome": "approve", "overridden": False,
        "decided_by": "reviewer:priya.nair", "decided_at": None,
    }  # fmt: skip
    detail = api.get(f"/api/claims/{claim['id']}", headers=AUTH).json()
    assert detail["audit"][-1]["action"] == "claim.decided"
    assert api.app.state.webhooks.events[-1][0] == "claim.decided"  # type: ignore[attr-defined]
    again = _decide(api, claim["id"], action="reject", reason="changed my mind entirely")
    assert again.status_code == 409  # already decided


def test_override_requires_a_reason(api: TestClient) -> None:
    claim = _triaged(api)
    short = _decide(api, claim["id"], action="reject", reason="no")
    assert short.status_code == 422 and short.json()["type"] == "urn:claimflow:problem:reason-required"
    response = _decide(api, claim["id"], action="reject", reason="Repair invoice is from an unregistered garage.")
    assert response.json()["overridden"] is True and response.json()["status"] == "rejected"
    detail = api.get(f"/api/claims/{claim['id']}", headers=AUTH).json()
    assert detail["audit"][-1]["action"] == "claim.overridden"
    assert detail["override_reason"] == "Repair invoice is from an unregistered garage."
    assert detail["recommended_outcome"] == "approve"  # the AI recommendation is kept next to the human decision


def test_escalated_claim_request_info_then_decide_with_reason(api: TestClient) -> None:
    claim = _triaged(api, amount="150000.00")  # over the auto-decision limit
    assert claim["status"] == "escalated"
    assert _decide(api, claim["id"], action="request_info").status_code == 422
    asked = _decide(api, claim["id"], action="request_info", reason="Please upload the police report.")
    assert asked.status_code == 200 and asked.json()["status"] == "escalated"
    assert _decide(api, claim["id"], action="approve").status_code == 422  # escalated: reason required
    decided = _decide(api, claim["id"], action="approve", reason="Surveyor confirmed the damage on site.")
    assert decided.json()["status"] == "approved" and decided.json()["overridden"] is False


def test_unknown_claim_is_404(api: TestClient) -> None:
    for response in (
        api.get(f"/api/claims/{uuid.uuid4()}", headers=AUTH),
        _decide(api, str(uuid.uuid4()), action="approve"),
        api.get(f"/api/claims/{uuid.uuid4()}/stream", headers=AUTH),
    ):
        assert response.status_code == 404 and response.json()["type"] == "urn:claimflow:problem:claim-not-found"


# ---- listing, metrics, search ---------------------------------------------------------------------------------


def test_list_filters_sorts_and_paginates(api: TestClient) -> None:
    page = api.get("/api/claims", params={"page_size": 5, "sort": "claimed_amount", "order": "asc"}, headers=AUTH)
    body = page.json()
    amounts = [float(item["claimed_amount"]) for item in body["items"]]
    assert len(amounts) == 5 and amounts == sorted(amounts) and body["total"] >= 200
    ranged = api.get("/api/claims", params={"amount_min": "50000", "amount_max": "60000"}, headers=AUTH).json()
    assert all(50000 <= float(item["claimed_amount"]) <= 60000 for item in ranged["items"])
    approved = api.get("/api/claims", params=[("status", "approved"), ("status", "rejected")], headers=AUTH).json()
    assert {item["status"] for item in approved["items"]} <= {"approved", "rejected"}
    prefix = api.get("/api/claims", params={"q": "clm-2"}, headers=AUTH).json()  # case-insensitive prefix
    assert prefix["total"] > 0 and all(item["claim_number"].startswith("CLM-2") for item in prefix["items"])
    literal = api.get("/api/claims", params={"q": "%"}, headers=AUTH).json()
    assert literal["total"] == 0  # "%" is matched literally, not as a wildcard
    bad = api.get("/api/claims", params={"amount_min": "10", "amount_max": "5"}, headers=AUTH)
    assert bad.status_code == 422


def test_metrics_reflect_triaged_claims(api: TestClient) -> None:
    _triaged(api)
    metrics = api.get("/api/metrics", params={"days": 7}, headers=AUTH).json()
    assert metrics["claims_today"] >= 1 and metrics["triaged_in_window"] >= 1
    assert metrics["avg_tool_calls_per_claim"] >= 2 and metrics["latency_ms_p50"] is not None
    assert len(metrics["volume"]) == 7 and metrics["volume"][-1]["submitted"] >= 1
    assert 0 <= metrics["escalation_rate"] <= 1


def test_policy_search(api: TestClient) -> None:
    assert api.get("/api/policies/search", params={"q": "theft"}, headers=AUTH).status_code == 503  # no model loaded

    async def ingest() -> PolicyRetriever:
        retriever = PolicyRetriever(AsyncQdrantClient(location=":memory:"), FakeEmbedder(), "api_search_test")
        await retriever.ingest_policy_documents(OUTPUT_DIR)
        return retriever

    api.app.state.policy_retriever = api.portal.call(ingest)  # type: ignore[attr-defined,union-attr]
    body = api.get("/api/policies/search", params={"q": "vehicle theft", "product_type": "motor"}, headers=AUTH).json()
    assert body["passages"] and all(p["document"] == "Motor_Policy.pdf" for p in body["passages"])
    assert body["passages"][0]["label"].startswith("Motor_Policy.pdf p.")

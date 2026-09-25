"""scripts.doctor: each check spots its problem and prints the fix."""

import asyncio
import hashlib
import hmac
from pathlib import Path

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult

from app.core.config import Settings
from scripts.doctor import (
    CheckResult,
    check_api_key,
    check_console_key,
    check_llm_providers,
    check_webhook,
    render,
    run_checks,
)

OWNER = "postgresql+asyncpg://claimflow:pw@db:5432/claimflow"
SECRET = "doctor-test-secret-0123456789"
URL = "http://n8n.test/webhook/claimflow"


def settings(**values: object) -> Settings:
    return Settings(_env_file=None, database_url=OWNER, **values)  # type: ignore[call-arg]


def test_render_fails_only_on_failures_and_shows_fixes() -> None:
    lines: list[str] = []
    code = render(
        [
            CheckResult("Postgres", "ok", "fine", "never shown"),
            CheckResult("LLM fallback", "warn", "one provider", "add a key"),
            CheckResult("n8n webhook", "skip", "off"),
        ],
        lines.append,
    )
    assert code == 0
    assert any("fix: add a key" in line for line in lines)
    assert not any("never shown" in line for line in lines)
    assert render([CheckResult("Qdrant", "fail", "down", "start it")], lines.append) == 1


def test_console_key(tmp_path: Path) -> None:
    configured = settings(api_key="the-api-key")
    env_local = tmp_path / ".env.local"
    assert check_console_key(configured, env_local).status == "skip"
    env_local.write_text("BACKEND_URL=http://localhost:8000\nAPI_KEY=the-api-key\n", encoding="utf-8")
    assert check_console_key(configured, env_local).status == "ok"
    env_local.write_text("API_KEY=another-key\n", encoding="utf-8")
    mismatch = check_console_key(configured, env_local)
    assert mismatch.status == "fail" and "npm run dev" in (mismatch.fix or "")
    env_local.write_text("# API_KEY=commented-out\n", encoding="utf-8")
    assert check_console_key(configured, env_local).status == "fail"


def test_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_KEY", raising=False)  # the container's own key would otherwise fill it in
    assert check_api_key(settings()).status == "fail"
    assert check_api_key(settings(api_key="k" * 20)).status == "ok"


def _webhook(status: int, seen: list[httpx.Request] | None = None) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(status, text='{"error":"bad signature"}' if status == 401 else "{}")

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_webhook_ping_is_signed_like_real_events() -> None:
    seen: list[httpx.Request] = []
    async with _webhook(202, seen) as client:
        result = await check_webhook(settings(n8n_webhook_url=URL, webhook_secret=SECRET), live=True, client=client)
    assert result.status == "ok"
    body = seen[0].content
    expected = "sha256=" + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
    assert seen[0].headers["X-ClaimFlow-Signature"] == expected


@pytest.mark.parametrize(("status", "hint"), [(401, "CLAIMFLOW_WEBHOOK_SECRET"), (404, "publish")])
async def test_webhook_rejections_name_the_cause(status: int, hint: str) -> None:
    async with _webhook(status) as client:
        result = await check_webhook(settings(n8n_webhook_url=URL, webhook_secret=SECRET), live=True, client=client)
    assert result.status == "fail" and hint in (result.fix or "")


async def test_webhook_skips() -> None:
    assert (await check_webhook(settings(), live=True)).status == "skip"
    assert (await check_webhook(settings(n8n_webhook_url=URL, webhook_secret=SECRET), live=False)).status == "skip"


class RateLimitError(Exception):
    """Named like the Groq SDK's exception, which is what the classifier keys on."""


RateLimitError.__module__ = "groq._exceptions"


class RateLimitedModel(FakeListChatModel):
    async def _agenerate(self, messages: list[BaseMessage], *args: object, **kwargs: object) -> ChatResult:
        raise RateLimitError("Error code: 429 - rate_limit_exceeded")


async def test_llm_checks() -> None:
    ok = FakeListChatModel(responses=["OK"])
    limited = RateLimitedModel(responses=["unused"])
    results = await check_llm_providers(settings(), live=True, models=[("Groq", limited), ("Gemini", ok)])
    by_name = {result.name: result for result in results}
    assert by_name["LLM Groq"].status == "fail" and "Wait a minute" in (by_name["LLM Groq"].fix or "")
    assert by_name["LLM Gemini"].status == "ok"
    single = await check_llm_providers(settings(), live=True, models=[("Gemini", ok)])
    assert [result.status for result in single] == ["ok", "warn"]
    offline = await check_llm_providers(settings(), live=False, models=[("Gemini", ok)])
    assert offline[0].status == "skip"
    assert (await check_llm_providers(settings(), live=True, models=[]))[0].status == "fail"


@pytest.mark.integration
def test_doctor_passes_on_the_running_stack() -> None:
    from app.core.config import get_settings

    current = get_settings()
    if current.tools_database_url is None:
        pytest.skip("TOOLS_DATABASE_URL is not set")
    results = {result.name: result for result in asyncio.run(run_checks(current, live=False))}
    if results["Postgres"].status == "fail":
        pytest.skip("the development database is not reachable here")
    for name in ("Postgres", "Read-only login", "Qdrant"):
        assert results[name].status == "ok", results[name]

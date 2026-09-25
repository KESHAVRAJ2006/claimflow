"""Failures reach people as one actionable sentence: LLM provider errors, broken logins, bad configuration."""

import asyncio
import logging
import uuid
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ValidationError
from sqlalchemy import pool, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.agents import llm as llm_module
from app.agents.llm import LlmClient
from app.agents.llm_errors import AllProvidersFailedError, classify_llm_error, explain_llm_error, retry_after_seconds
from app.api.routes.claims import _terminal_event
from app.core.config import Settings
from app.db.models import Claim
from app.db.readonly import ReadOnlyViolationError
from app.domain.enums import ClaimStatus
from app.services import claims as claim_service
from app.services.triage import explain_run_failure
from tests.conftest import ScratchDatabase

OWNER = "postgresql+asyncpg://claimflow:pw@db:5432/claimflow"


def provider_error(name: str, module: str, message: str = "", **attributes: object) -> Exception:
    """An exception shaped like a provider SDK's, without importing the SDK."""
    error_type = type(name, (Exception,), {"__module__": module})
    error = error_type(message)
    for key, value in attributes.items():
        setattr(error, key, value)
    return error


@pytest.mark.parametrize(
    ("error", "kind", "provider"),
    [
        (provider_error("RateLimitError", "groq._exceptions", "Error code: 429"), "rate_limit", "Groq"),
        (provider_error("GoogleRateLimitError", "langchain_google_genai.chat_models"), "rate_limit", "Gemini"),
        (
            provider_error("ClientError", "google.genai.errors", "429 RESOURCE_EXHAUSTED", code=429),
            "rate_limit",
            "Gemini",
        ),
        (provider_error("AuthenticationError", "groq._exceptions", "Invalid API Key"), "auth", "Groq"),
        (provider_error("ClientError", "google.genai.errors", "400 API key not valid", code=400), "auth", "Gemini"),
        (provider_error("NotFoundError", "groq._exceptions", "model_decommissioned"), "model_not_found", "Groq"),
        (provider_error("APITimeoutError", "groq._exceptions", "Request timed out."), "timeout", "Groq"),
        (provider_error("APIConnectionError", "groq._exceptions", "Connection error."), "connection", "Groq"),
    ],
)
def test_provider_errors_are_recognised(error: Exception, kind: str, provider: str) -> None:
    assert classify_llm_error(error) == kind
    assert explain_llm_error(error).startswith(f"{provider}: ")


def test_advice_names_the_fix() -> None:
    rate_limited = explain_llm_error(provider_error("RateLimitError", "groq._exceptions", "429"))
    assert "Wait a minute" in rate_limited and "GOOGLE_API_KEY" in rate_limited
    retired = explain_llm_error(provider_error("NotFoundError", "groq._exceptions", "model_decommissioned"))
    assert "GROQ_MODEL" in retired and "docker compose up -d" in retired


def test_unknown_errors_are_shortened_not_dumped() -> None:
    error = ValueError("x" * 1000)
    assert classify_llm_error(error) is None
    text = explain_llm_error(error)
    assert text.startswith("ValueError: ") and len(text) < 200


def test_run_failures_explain_llm_and_login_problems() -> None:
    assert "Groq: rate limit" in explain_run_failure(provider_error("RateLimitError", "groq._exceptions", "429"))
    for login_error in (
        ReadOnlyViolationError("claimflow_agent can INSERT into claims"),
        provider_error("InvalidPasswordError", "asyncpg.exceptions", "password authentication failed"),
    ):
        assert "provision_readonly_role" in explain_run_failure(login_error)
    generic = explain_run_failure(KeyError("boom"))
    assert generic == "Triage failed with KeyError; see the server log."


# ---- configuration that can only fail later is refused at startup ---------------------------------------------------


def test_webhook_url_needs_a_signing_secret() -> None:
    url = "http://n8n:5678/webhook/claimflow"
    with pytest.raises(ValidationError, match="WEBHOOK_SECRET"):
        Settings(_env_file=None, database_url=OWNER, n8n_webhook_url=url)  # type: ignore[call-arg]
    with pytest.raises(ValidationError, match="WEBHOOK_SECRET"):
        Settings(_env_file=None, database_url=OWNER, n8n_webhook_url=url, webhook_secret="short")  # type: ignore[call-arg]
    ok = Settings(_env_file=None, database_url=OWNER, n8n_webhook_url=url, webhook_secret="s" * 16)  # type: ignore[call-arg]
    assert ok.n8n_webhook_url == url


def test_qdrant_url_needs_a_scheme() -> None:
    with pytest.raises(ValidationError, match="QDRANT_URL"):
        Settings(_env_file=None, database_url=OWNER, qdrant_url="abc.cloud.qdrant.io:6333")  # type: ignore[call-arg]
    Settings(_env_file=None, database_url=OWNER, qdrant_url="https://abc.cloud.qdrant.io:6333")  # type: ignore[call-arg]


# ---- a reloaded page shows the stored failure reason ---------------------------------------------------------------


def test_replayed_failure_carries_the_stored_reason() -> None:
    claim = Claim(status=ClaimStatus.FAILED)
    assert _terminal_event(claim, True, "Groq: rate limit reached.")["reason"] == "Groq: rate limit reached."
    assert _terminal_event(claim, True)["reason"] == "See the audit log."


@pytest.mark.integration
def test_latest_failure_reason_reads_the_newest_audit_row(test_database: ScratchDatabase) -> None:
    async def scenario() -> tuple[str | None, str | None]:
        engine = create_async_engine(test_database.owner_url, poolclass=pool.NullPool)
        try:
            async with AsyncSession(engine) as session:
                claim_id = (await session.scalars(select(Claim.id).limit(1))).one()
                before = await claim_service.latest_failure_reason(session, uuid.uuid4())
                for reason in ("first failure", "second failure"):
                    claim_service._audit(session, claim_id, "system", "claim.triage_failed", reason=reason)
                    await session.commit()
                return before, await claim_service.latest_failure_reason(session, claim_id)
        finally:
            await engine.dispose()

    unknown, latest = asyncio.run(scenario())
    assert unknown is None
    assert latest == "second failure"


# ---- a rate-limited primary hands over to the fallback instead of waiting ------------------------------------------


def test_groq_fails_over_immediately_when_gemini_is_configured() -> None:
    from app.agents.llm import provider_models

    both = provider_models(
        Settings(_env_file=None, database_url=OWNER, groq_api_key="g" * 20, google_api_key="k" * 20, llm_max_retries=2)  # type: ignore[call-arg]
    )
    retries = {name: model.max_retries for name, model in both}  # type: ignore[attr-defined]
    assert retries == {"Groq": 0, "Gemini": 2}
    groq_only = provider_models(Settings(_env_file=None, database_url=OWNER, groq_api_key="g" * 20, llm_max_retries=2))  # type: ignore[call-arg]
    assert [model.max_retries for _, model in groq_only] == [2]  # type: ignore[attr-defined]


class _Scripted(BaseChatModel):
    """A chat model that raises the queued errors in turn, then answers "ok"."""

    errors: list[Exception]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        return self

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return ChatResult(generations=[ChatGeneration(message=AIMessage("ok"))])


def rate_limited(hint: str = "Please try again in 3.5s.") -> Exception:
    return provider_error("RateLimitError", "groq._exceptions", f"Error code: 429 - {hint}")


@pytest.fixture
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        recorded.append(seconds)

    monkeypatch.setattr(llm_module, "_sleep", fake_sleep)
    return recorded


def test_every_failing_provider_is_named(caplog: pytest.LogCaptureFixture, waits: list[float]) -> None:
    # LangChain's own fallback re-raised only the first error; ours keeps each provider's.
    client = LlmClient(
        [_Scripted(errors=[RuntimeError("groq is down")]), _Scripted(errors=[RuntimeError("gemini broke")])]
    )
    with caplog.at_level(logging.WARNING, logger="app.agents.llm"), pytest.raises(AllProvidersFailedError) as caught:
        asyncio.run(client.with_tools([]).ainvoke([HumanMessage("hi")]))
    assert [str(error) for _, error in caught.value.errors] == ["groq is down", "gemini broke"]
    assert sum(record.getMessage() == "LLM provider failed" for record in caplog.records) == 2
    assert waits == []  # not rate limits: no point waiting


def test_all_rate_limited_waits_for_the_soonest_then_succeeds(waits: list[float]) -> None:
    client = LlmClient(
        [
            _Scripted(errors=[rate_limited("Please try again in 3.5s.")]),
            _Scripted(errors=[rate_limited("retryDelay': '9s'")]),
        ]
    )
    reply = asyncio.run(client.with_tools([]).ainvoke([HumanMessage("hi")]))
    assert reply.content == "ok"
    assert waits == [3.5]


def test_daily_quotas_are_not_waited_for(waits: list[float]) -> None:
    client = LlmClient(
        [
            _Scripted(errors=[rate_limited("Please try again in 7m12.5s.")]),
            _Scripted(errors=[rate_limited("Please try again in 45m0s.")]),
        ]
    )
    with pytest.raises(AllProvidersFailedError) as caught:
        asyncio.run(client.with_tools([]).ainvoke([HumanMessage("hi")]))
    assert waits == []
    assert "out of free quota" in explain_llm_error(caught.value)
    assert "GOOGLE_API_KEY" not in explain_llm_error(caught.value)  # both keys are already set


def test_waiting_gives_up_after_the_allowed_rounds(waits: list[float]) -> None:
    rounds = llm_module.RATE_LIMIT_ROUNDS + 1
    client = LlmClient(
        [
            _Scripted(errors=[rate_limited() for _ in range(rounds)]),
            _Scripted(errors=[rate_limited() for _ in range(rounds)]),
        ]
    )
    with pytest.raises(AllProvidersFailedError):
        asyncio.run(client.with_tools([]).ainvoke([HumanMessage("hi")]))
    assert waits == [3.5] * llm_module.RATE_LIMIT_ROUNDS


@pytest.mark.parametrize(
    ("hint", "expected"),
    [
        ("Please try again in 18.6825s.", 18.6825),
        ("retry in 450ms", 1.0),
        ("'retryDelay': '37s'", None),
        ("Please try again in 7m12.5s.", None),
        ("no hint at all", 20.0),
    ],
)
def test_retry_after_seconds(hint: str, expected: float | None) -> None:
    assert retry_after_seconds(rate_limited(hint)) == expected


def test_single_provider_advice_suggests_the_second_key() -> None:
    error = rate_limited()
    assert "GOOGLE_API_KEY" in explain_llm_error(error)
    assert "GOOGLE_API_KEY" not in explain_llm_error(error, single_provider=False)


GEMINI_DAILY = (
    "429 RESOURCE_EXHAUSTED. You exceeded your current quota. "
    "quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier, retryDelay': '44s'"
)


def test_a_daily_quota_is_recognised_despite_its_short_retry_delay() -> None:
    error = provider_error("GoogleRateLimitError", "langchain_google_genai.chat_models", GEMINI_DAILY)
    assert classify_llm_error(error) == "daily_quota"
    assert retry_after_seconds(error) is None  # 44s is Google's suggestion; the quota resets tomorrow
    groq_daily = rate_limited("Rate limit reached on tokens per day (TPD). Please try again in 3.5s.")
    assert classify_llm_error(groq_daily) == "daily_quota"


def test_a_provider_out_of_daily_quota_is_skipped(waits: list[float]) -> None:
    gemini_daily = provider_error("GoogleRateLimitError", "langchain_google_genai.chat_models", GEMINI_DAILY)
    groq = _Scripted(errors=[rate_limited()])
    gemini = _Scripted(errors=[gemini_daily])
    client = LlmClient([groq, gemini])
    runnable = client.with_tools([])
    assert asyncio.run(runnable.ainvoke([HumanMessage("first")])).content == "ok"
    assert asyncio.run(runnable.ainvoke([HumanMessage("second")])).content == "ok"
    # Gemini refused once for the day and was not asked again; Groq's per-minute limit was waited out.
    assert gemini.calls == 1
    assert waits == [3.5]


def test_waiting_stops_at_the_per_call_budget(waits: list[float]) -> None:
    hint = "Please try again in 25s."
    client = LlmClient(
        [
            _Scripted(errors=[rate_limited(hint) for _ in range(9)]),
            _Scripted(errors=[rate_limited(hint) for _ in range(9)]),
        ]
    )
    with pytest.raises(AllProvidersFailedError):
        asyncio.run(client.with_tools([]).ainvoke([HumanMessage("hi")]))
    assert sum(waits) <= llm_module.MAX_WAIT_PER_CALL_S
    assert waits == [25.0, 25.0, 25.0]


def test_only_a_batch_client_waits_out_a_long_per_minute_hint(waits: list[float]) -> None:
    # Groq tells a large request on its 8,000 tokens-a-minute tier to come back in 40-60 s.
    def providers() -> list[Any]:
        return [_Scripted(errors=[rate_limited("Please try again in 45s.")]) for _ in range(2)]

    with pytest.raises(AllProvidersFailedError):
        asyncio.run(LlmClient(providers()).with_tools([]).ainvoke([HumanMessage("hi")]))
    assert waits == []  # interactive: a person is watching, so fail fast and say why
    batch = LlmClient(providers(), wait=llm_module.BATCH_WAIT)
    assert asyncio.run(batch.with_tools([]).ainvoke([HumanMessage("hi")])).content == "ok"
    assert waits == [45.0]


class _Hanging(_Scripted):
    """A provider whose connection stalls: the call never returns on its own."""

    async def _agenerate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        self.calls += 1
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")


def test_a_stalled_provider_is_cut_off_and_the_next_one_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_module, "DEADLINE_SLACK_S", 0.0)
    stalled, backup = _Hanging(errors=[]), _Scripted(errors=[])
    client = LlmClient([stalled, backup], call_timeout_s=0.05)
    reply = asyncio.run(client.with_tools([]).ainvoke([HumanMessage("hi")]))
    assert reply.content == "ok" and stalled.calls == 1 and backup.calls == 1


def test_a_single_stalled_provider_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_module, "DEADLINE_SLACK_S", 0.0)
    client = LlmClient([_Hanging(errors=[])], call_timeout_s=0.05)
    with pytest.raises(TimeoutError) as caught:
        asyncio.run(client.with_tools([]).ainvoke([HumanMessage("hi")]))
    assert classify_llm_error(caught.value) == "timeout"

"""One LLM interface for every agent: Groq first, Gemini when Groq fails.

Agents never name a provider. They ask this client for "a model that can call these tools" or "a model that
returns this Pydantic schema", so switching or adding a provider touches only this file.

Why Groq primary: gpt-oss-120b on Groq answers tool-calling prompts in about a second, which keeps a 6-step
investigation interactive. Why a fallback: Groq's free tier allows 8,000 tokens per minute and one decision prompt
is about 5,000, so a claim routinely meets a "429"; it should move to Gemini, not fail or wait. When BOTH are
rate-limited at once (Gemini's free tier allows about 10 requests a minute), the chain waits for whichever frees up
first and tries again, instead of failing the step and escalating the claim for a reason that has nothing to do
with the claim.
"""

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from app.agents.llm_errors import (
    AllProvidersFailedError,
    classify_llm_error,
    explain_llm_error,
    retry_after_seconds,
)
from app.core.config import Settings

logger = logging.getLogger(__name__)

ToolSpec = BaseTool | dict[str, Any] | type[BaseModel]
T = TypeVar("T", bound=BaseModel)


class LlmNotConfiguredError(RuntimeError):
    """No LLM provider has an API key."""


class LlmClient:
    """A primary chat model plus ordered fallbacks."""

    def __init__(
        self, models: Sequence[BaseChatModel], *, call_timeout_s: float | None = None, wait: "WaitPolicy | None" = None
    ) -> None:
        """Wrap chat models in priority order.

        Args:
            models: Primary first, then fallbacks. At least one.
            call_timeout_s: LLM_TIMEOUT_S. Each provider call gets a hard deadline derived from it (see
                ``_deadline``); None (tests with scripted models) means no deadline.
            wait: How long to wait out rate limits; None for INTERACTIVE_WAIT.

        Raises:
            ValueError: If ``models`` is empty.
        """
        if not models:
            raise ValueError("LlmClient needs at least one chat model")
        self._models = tuple(models)
        self._call_timeout_s = call_timeout_s
        self._wait = wait or INTERACTIVE_WAIT
        # Providers out of daily quota, by position in the chain (names can repeat), skipped until the time here.
        self._cooldowns: dict[int, tuple[float, BaseException]] = {}

    def with_tools(self, tools: Sequence[ToolSpec], *, tool_choice: str | None = None) -> Runnable[Any, Any]:
        """A model that can call the given tools.

        Args:
            tools: LangChain tools, OpenAI-style tool dicts, or Pydantic classes.
            tool_choice: Force a specific tool by name, or None to let the model choose (or answer in text).

        Returns:
            A runnable returning an AIMessage, falling back to the next provider on any error.
        """
        bound = [
            model.bind_tools(tools, tool_choice=tool_choice) if tool_choice else model.bind_tools(tools)
            for model in self._models
        ]
        return self._chain(bound)

    def structured(self, schema: type[BaseModel]) -> Runnable[Any, Any]:
        """A model whose answer is parsed into ``schema``.

        Args:
            schema: The Pydantic model the answer must match.

        Returns:
            A runnable returning ``{"raw": AIMessage, "parsed": schema | None, "parsing_error": Exception | None}``.
            ``include_raw`` means a malformed answer comes back as a value we can retry on, not an exception.
        """
        bound = [model.with_structured_output(schema, include_raw=True) for model in self._models]
        return self._chain(bound)

    def _chain(self, bound: Sequence[Runnable[Any, Any]]) -> Runnable[Any, Any]:
        """Call the providers in order, skipping any that is out of daily quota and waiting out per-minute limits.

        Every call gets a hard deadline. A single provider is not retried here (its SDK already retries and
        honours Retry-After). With several, this chain replaces LangChain's ``with_fallbacks``, which re-raises
        only the FIRST provider's error (so a second provider's failure was invisible), gives up at once when all
        are rate-limited, and keeps calling a provider whose quota for the day is gone.

        Args:
            bound: One runnable per model, in the same order as ``self._models``.

        Returns:
            A runnable with the same input and output as each provider's.
        """
        providers = tuple(
            _Provider(provider_name(model), runnable, _deadline(model, self._call_timeout_s))
            for model, runnable in zip(self._models, bound, strict=True)
        )
        cooldowns, wait = self._cooldowns, self._wait

        async def call_in_order(value: Any, config: RunnableConfig) -> Any:
            return await _call_providers(providers, value, config, cooldowns, wait)

        return RunnableLambda(call_in_order, name="llm_providers")


# Measured free tiers (September 2026): Groq allows 8,000 tokens a minute, about 1.5 of our calls; Gemini 2.5 Flash
# allows 20 requests a DAY. A run needs 12-18 calls, so on free keys most of its time is spent inside these limits.
# Per-minute limits are waited out (up to RATE_LIMIT_ROUNDS times, MAX_WAIT_PER_CALL_S in total); that makes a
# free-tier run slower but lets it reach a real decision instead of escalating because a quota ran out.
RATE_LIMIT_ROUNDS = 4
MAX_WAIT_PER_CALL_S = 90.0


@dataclass(frozen=True)
class WaitPolicy:
    """How long one call may wait for per-minute limits to reset before it fails."""

    rounds: int = RATE_LIMIT_ROUNDS
    max_total_s: float = MAX_WAIT_PER_CALL_S
    # A longer hint means a quota that will not reset while the caller waits (see retry_after_seconds).
    max_hint_s: float = 30.0


# A person is watching the claim: fail within a couple of minutes and say why.
INTERACTIVE_WAIT = WaitPolicy()
# Batch jobs (the evaluation): nobody is waiting, so ride out every per-minute reset. A large request on Groq's
# 8,000 tokens-a-minute tier is often told to come back in 40-60 s, which the interactive policy never waits for.
BATCH_WAIT = WaitPolicy(rounds=12, max_total_s=600.0, max_hint_s=65.0)
# A provider out of daily quota is skipped for this long, instead of being asked (and refusing) on every step.
DAILY_QUOTA_COOLDOWN_S = 15 * 60
# Patched in tests so they neither wait nor depend on the clock.
_sleep = asyncio.sleep
_now = time.monotonic


# Added to each provider's deadline: the SDK's own backoff sleeps between its retries, plus connection setup.
DEADLINE_SLACK_S = 30.0


@dataclass(frozen=True)
class _Provider:
    name: str
    runnable: Runnable[Any, Any]
    deadline_s: float | None


def _deadline(model: BaseChatModel, call_timeout_s: float | None) -> float | None:
    """The longest one provider call may take, retries included.

    The SDKs' ``timeout`` bounds each network read, not the whole call: a stalled connection held one decision step
    for 6.5 minutes in a measured run. An outer deadline in our own code is the only bound that always holds.

    Args:
        model: The chat model (its ``max_retries`` says how many attempts one call can make).
        call_timeout_s: LLM_TIMEOUT_S, the limit for one attempt; None for no deadline.

    Returns:
        Seconds, or None.
    """
    if call_timeout_s is None:
        return None
    attempts = int(getattr(model, "max_retries", 0) or 0) + 1
    return call_timeout_s * attempts + DEADLINE_SLACK_S


def provider_name(model: BaseChatModel) -> str:
    """A readable provider name for logs and messages.

    Args:
        model: A chat model.

    Returns:
        "Groq", "Gemini", or the model's class name.
    """
    return {"ChatGroq": "Groq", "ChatGoogleGenerativeAI": "Gemini"}.get(type(model).__name__, type(model).__name__)


async def _call_providers(
    providers: Sequence[_Provider],
    value: Any,
    config: RunnableConfig,
    cooldowns: dict[int, tuple[float, BaseException]],
    wait: WaitPolicy = INTERACTIVE_WAIT,
) -> Any:
    """Try each available provider in order; if all are rate-limited, wait for the soonest and go round again.

    Args:
        providers: The providers in priority order, each with its deadline.
        value: The call's input.
        config: The caller's run config (callbacks, LangGraph context).
        cooldowns: Provider position -> (skip until, the daily-quota error), shared by every call of one client.
        wait: How long rate limits may be waited out.

    Returns:
        The first successful provider's output.

    Raises:
        AllProvidersFailedError: With each provider's own error, once waiting cannot help any more.
    """
    waited = 0.0
    for round_number in range(wait.rounds + 1):
        now = _now()
        cooling = {position for position, (until, _) in cooldowns.items() if until > now}
        indexed = list(enumerate(providers))
        # If every provider is cooling down, ask them anyway: a quota may have reset, and there is no one else.
        active = [item for item in indexed if item[0] not in cooling] or indexed
        errors: list[tuple[str, BaseException]] = []
        for position, provider in active:
            name = provider.name
            try:
                async with asyncio.timeout(provider.deadline_s):
                    result = await provider.runnable.ainvoke(value, config)
            except Exception as error:  # noqa: BLE001 — any provider failure moves on to the next provider
                kind = classify_llm_error(error)
                logger.warning(
                    "LLM provider failed", extra={"provider": name, "kind": kind, "error": repr(error)[:300]}
                )
                if kind == "daily_quota":
                    cooldowns[position] = (_now() + DAILY_QUOTA_COOLDOWN_S, error)
                    logger.warning(
                        "LLM provider out of daily quota; skipping it for a while",
                        extra={"provider": name, "seconds": DAILY_QUOTA_COOLDOWN_S},
                    )
                errors.append((name, error))
            else:
                cooldowns.pop(position, None)
                return result
        # Providers skipped for their daily quota still belong in the report of why nothing answered.
        tried = {position for position, _ in active}
        errors.extend((item.name, cooldowns[position][1]) for position, item in indexed if position not in tried)
        # One provider: its SDK has already waited out Retry-After, so waiting again here would only double it.
        if len(providers) == 1:
            raise errors[0][1]
        only_limits = all(classify_llm_error(error) in ("rate_limit", "daily_quota") for _, error in errors)
        hints = [hint for _, error in errors if (hint := retry_after_seconds(error, cap=wait.max_hint_s)) is not None]
        if round_number < wait.rounds and only_limits and hints and waited + min(hints) <= wait.max_total_s:
            delay = min(hints)
            waited += delay
            logger.info("all LLM providers rate-limited; waiting", extra={"seconds": delay, "round": round_number + 1})
            await _sleep(delay)
            continue
        raise AllProvidersFailedError(tuple(errors))
    raise AssertionError("unreachable: the last round always returns or raises")


@dataclass(frozen=True)
class StructuredOutcome(Generic[T]):
    """Result of asking for structured output with one retry."""

    parsed: T | None
    attempts: int
    errors: tuple[str, ...]


async def invoke_structured(
    client: LlmClient, schema: type[T], messages: Sequence[BaseMessage], *, retries: int = 1
) -> StructuredOutcome[T]:
    """Ask for ``schema``; on a malformed answer, retry with the validation error appended.

    Showing the model its own error is far more effective than repeating the same prompt: it tells the model
    exactly which field was wrong.

    Args:
        client: The LLM client.
        schema: Pydantic model to parse into.
        messages: The prompt.
        retries: Extra attempts after the first. The spec allows exactly one for intake.

    Returns:
        The parsed value (None if every attempt failed), the number of attempts and the error of each failure.
    """
    runnable = client.structured(schema)
    history = list(messages)
    errors: list[str] = []
    for attempt in range(1, retries + 2):
        try:
            response = await runnable.ainvoke(history)
        except Exception as error:  # noqa: BLE001 — every provider failed; report it like a parse failure
            logger.warning("structured LLM call failed", extra={"schema": schema.__name__, "error": repr(error)})
            errors.append(f"LLM call failed: {explain_llm_error(error)}")
            continue
        parsed = response.get("parsed")
        if isinstance(parsed, schema) and response.get("parsing_error") is None:
            return StructuredOutcome(parsed, attempt, tuple(errors))
        error_text = str(response.get("parsing_error") or f"answer did not match {schema.__name__}")
        errors.append(error_text)
        history.append(
            HumanMessage(
                f"Your previous answer was rejected by validation:\n{error_text}\n"
                f"Answer again, returning a {schema.__name__} that fixes exactly these problems."
            )
        )
    return StructuredOutcome(None, retries + 1, tuple(errors))


def provider_models(settings: Settings) -> list[tuple[str, BaseChatModel]]:
    """One chat model per configured provider, in priority order: Groq, then Gemini.

    Args:
        settings: Application settings.

    Returns:
        (provider name, model) pairs; empty when no key is set. ``scripts.doctor`` tests each one on its own.
    """
    # Imported here so modules that only need the interface (and tests with a fake model) don't pay for the SDKs.
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_groq import ChatGroq

    models: list[tuple[str, BaseChatModel]] = []
    # With a fallback configured, Groq must not retry on its own. Its SDK honours Retry-After on a rate limit and
    # waits 15-40 s per attempt, so the fallback never got a chance: a measured run spent most of 6 minutes waiting
    # on 17 such retries. With no retries, a 429 hands the call to Gemini at once. Retrying (and waiting) only
    # makes sense for the last provider in the chain, which has nothing to hand over to.
    has_fallback = settings.google_api_key is not None
    # temperature 0: triage should give the same answer for the same evidence; creativity is a defect here.
    if settings.groq_api_key is not None:
        groq = ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=0,
            max_retries=0 if has_fallback else settings.llm_max_retries,
            timeout=settings.llm_timeout_s,
        )
        models.append(("Groq", groq))
    if settings.google_api_key is not None:
        gemini = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0,
            max_retries=settings.llm_max_retries,
            timeout=settings.llm_timeout_s,
        )
        models.append(("Gemini", gemini))
    return models


def create_llm_client(settings: Settings, *, wait: WaitPolicy | None = None) -> LlmClient:
    """Build the client from configured keys: Groq first, then Gemini.

    Args:
        settings: Application settings.
        wait: Rate-limit waiting; None for INTERACTIVE_WAIT, BATCH_WAIT for jobs nobody is watching.

    Returns:
        A client with every provider that has a key.

    Raises:
        LlmNotConfiguredError: If neither GROQ_API_KEY nor GOOGLE_API_KEY is set.
    """
    models = [model for _, model in provider_models(settings)]
    if not models:
        raise LlmNotConfiguredError("Set GROQ_API_KEY and/or GOOGLE_API_KEY in .env to run the agents")
    return LlmClient(models, call_timeout_s=settings.llm_timeout_s, wait=wait)

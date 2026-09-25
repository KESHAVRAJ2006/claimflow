"""Turn an LLM provider failure into one sentence a reviewer or developer can act on.

A raw provider error ("RateLimitError: Error code: 429 - {'error': {...}}") ends up as a claim's failure reason
and in the live trace. It tells a reviewer nothing, and the fix is almost always one of five: wait, add the
fallback key, fix a key, change a retired model name, or check the network. This module names which one.

It matches on exception class names and HTTP status codes rather than importing the SDKs' exception classes, so
it works for Groq, Gemini and their LangChain wrappers alike, and for any provider added later.
"""

import re
from typing import Literal

ErrorKind = Literal["rate_limit", "daily_quota", "auth", "model_not_found", "timeout", "connection"]

# Class-name fragments, checked in order; the first match wins.
_NAME_HINTS: tuple[tuple[str, ErrorKind], ...] = (
    ("ratelimit", "rate_limit"),
    ("resourceexhausted", "rate_limit"),
    ("authentication", "auth"),
    ("permissiondenied", "auth"),
    ("notfound", "model_not_found"),
    ("timeout", "timeout"),
    ("connection", "connection"),
    ("connect", "connection"),
)
# Message fragments for errors whose class is generic (for example google.genai's ClientError).
_TEXT_HINTS: tuple[tuple[str, ErrorKind], ...] = (
    ("resource_exhausted", "rate_limit"),
    ("rate limit", "rate_limit"),
    ("quota", "rate_limit"),
    ("api key not valid", "auth"),
    ("invalid api key", "auth"),
    ("permission_denied", "auth"),
    ("decommissioned", "model_not_found"),
    ("model_not_found", "model_not_found"),
    ("is not found for api version", "model_not_found"),
    ("does not exist", "model_not_found"),
    ("timed out", "timeout"),
)
_STATUS_HINTS: dict[int, ErrorKind] = {429: "rate_limit", 401: "auth", 403: "auth", 404: "model_not_found"}
# A rate limit that resets daily, not per minute. Groq says "tokens per day (TPD)"; Gemini's free tier names the
# quota "GenerateRequestsPerDayPerProjectPerModel-FreeTier" while still suggesting "retryDelay: 44s", so the quota
# name, not the delay, decides.
_DAILY_MARKERS = ("perday", "per day", "(tpd)", "(rpd)")

# Only true when a single provider is configured; with two, the other one already takes over automatically.
_SECOND_PROVIDER_HINT = "; setting both GROQ_API_KEY and GOOGLE_API_KEY lets the other provider take over"

_ADVICE: dict[ErrorKind, str] = {
    "rate_limit": "rate limit or quota reached. Wait a minute and re-run the claim",
    "daily_quota": "the free daily quota is used up. It resets within a day; until then the other provider "
    "carries the load, or add billing to the key",
    "auth": "the API key was rejected. Check GROQ_API_KEY / GOOGLE_API_KEY in .env, then run `docker compose up -d` "
    "so the container gets the new value",
    "model_not_found": "the model is not available to this key (it may have been retired). Set GROQ_MODEL / "
    "GEMINI_MODEL in .env to a current model and run `docker compose up -d`",
    "timeout": "the provider did not answer in time (LLM_TIMEOUT_S). Re-run the claim; if it keeps happening the "
    "provider is overloaded",
    "connection": "the provider could not be reached. Check the network connection from the API container",
}


class AllProvidersFailedError(RuntimeError):
    """Every configured LLM provider failed the same call. Holds each provider's own error, in order."""

    def __init__(self, errors: tuple[tuple[str, BaseException], ...]) -> None:
        """Record the failures.

        Args:
            errors: (provider name, exception) for every provider tried.
        """
        detail = "; ".join(f"{name}: {type(error).__name__}: {_one_line(error)}" for name, error in errors)
        super().__init__(f"all LLM providers failed ({detail})")
        self.errors = errors


def _one_line(error: BaseException, limit: int = 160) -> str:
    message = " ".join(str(error).split())
    return message if len(message) <= limit else f"{message[: limit - 3]}..."


def _status(error: BaseException) -> int | None:
    for attribute in ("status_code", "code", "status"):
        value = getattr(error, attribute, None)
        if isinstance(value, int):
            return value
    return None


def classify_llm_error(error: BaseException) -> ErrorKind | None:
    """Recognise a provider failure.

    Args:
        error: The exception an LLM call raised, or an AllProvidersFailedError wrapping one per provider.

    Returns:
        The kind of failure (for several providers: the shared kind, else the first recognised one), or None if it
        is not a recognisable provider error.
    """
    if isinstance(error, AllProvidersFailedError):
        kinds = [classify_llm_error(inner) for _, inner in error.errors]
        if len(set(kinds)) == 1:
            return kinds[0]
        return next((kind for kind in kinds if kind is not None), None)
    kind = _base_kind(error)
    if kind == "rate_limit" and any(marker in str(error).lower() for marker in _DAILY_MARKERS):
        return "daily_quota"
    return kind


def _base_kind(error: BaseException) -> ErrorKind | None:
    name = type(error).__name__.lower()
    for fragment, kind in _NAME_HINTS:
        if fragment in name:
            return kind
    status = _status(error)
    if status in _STATUS_HINTS:
        return _STATUS_HINTS[status]
    text = str(error).lower()
    for fragment, kind in _TEXT_HINTS:
        if fragment in text:
            return kind
    return None


def provider_of(error: BaseException) -> str:
    """Name the provider an error came from, for the message.

    Args:
        error: The exception.

    Returns:
        "Groq", "Gemini", "Groq and Gemini" (every provider failed) or "The LLM provider".
    """
    if isinstance(error, AllProvidersFailedError):
        return " and ".join(name for name, _ in error.errors)
    origin = f"{type(error).__module__}.{type(error).__name__}".lower()
    if "groq" in origin:
        return "Groq"
    if "google" in origin or "gemini" in origin:
        return "Gemini"
    return "The LLM provider"


def explain_llm_error(error: BaseException, *, single_provider: bool = True) -> str:
    """One actionable sentence for a failed LLM call.

    Args:
        error: The exception, or an AllProvidersFailedError from the provider chain.
        single_provider: Whether only one provider is configured, so suggesting the second key is useful.

    Returns:
        "Groq: rate limit or quota reached. Wait a minute ..." for a recognised failure, a combined sentence when
        every provider failed, otherwise the exception type and a shortened message (never the provider payload).
    """
    if isinstance(error, AllProvidersFailedError):
        kinds = {name: classify_llm_error(inner) for name, inner in error.errors}
        if set(kinds.values()) <= {"rate_limit", "daily_quota"}:
            limits = ", ".join(
                f"{name} ({'daily' if kind == 'daily_quota' else 'per-minute'} limit)" for name, kind in kinds.items()
            )
            return (
                f"Every LLM provider is out of free quota for now: {limits}. Wait a minute and re-run the claim; "
                "a paid tier on either key removes the limit."
            )
        return " ".join(explain_llm_error(inner, single_provider=False) for _, inner in error.errors)
    kind = classify_llm_error(error)
    if kind is not None:
        hint = _SECOND_PROVIDER_HINT if kind == "rate_limit" and single_provider else ""
        return f"{provider_of(error)}: {_ADVICE[kind]}{hint}."
    short = _one_line(error)
    return f"{type(error).__name__}: {short}" if short else type(error).__name__


# "Please try again in 18.68s", "try again in 7m12.5s", "retry in 450ms", Gemini's "retryDelay': '37s'".
_RETRY_IN = re.compile(r"(?:try again|retry) in\s+(?:(\d+)m)?([\d.]+)(ms|s)\b", re.IGNORECASE)
_RETRY_DELAY = re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?([\d.]+)s", re.IGNORECASE)


def retry_after_seconds(error: BaseException, *, default: float = 20.0, cap: float = 30.0) -> float | None:
    """How long a rate-limited provider asks us to wait.

    Args:
        error: A rate-limit error.
        default: Wait when the provider gives no hint (free tiers reset per minute).
        cap: Longest wait worth taking inside one call.

    Returns:
        Seconds to wait (at least 1), or None when the provider asks for longer than ``cap``: that is a daily or
        hourly quota, and waiting inside the call would only hold the run up before failing anyway.
    """
    if classify_llm_error(error) == "daily_quota":
        return None  # whatever delay it suggests, a daily quota will not reset within one call
    text = str(error)
    seconds: float | None = None
    if match := _RETRY_IN.search(text):
        minutes, value, unit = match.groups()
        seconds = float(value) / (1000 if unit.lower() == "ms" else 1) + 60 * int(minutes or 0)
    elif match := _RETRY_DELAY.search(text):
        seconds = float(match.group(1))
    if seconds is None:
        headers = getattr(getattr(error, "response", None), "headers", None) or {}
        header = headers.get("retry-after") if hasattr(headers, "get") else None
        seconds = float(header) if header and str(header).replace(".", "", 1).isdigit() else default
    return None if seconds > cap else max(seconds, 1.0)

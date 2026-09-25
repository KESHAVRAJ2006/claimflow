"""API key authentication.

One shared key in the ``X-API-Key`` header. It is sent by the Next.js server (Phase 8 proxies API calls through
its own route handlers), so the key never reaches a browser. A per-user identity provider would replace this in a
multi-tenant deployment; for one internal console a single secret keeps the surface small.
"""

import secrets
from typing import Annotated

from fastapi import Depends, Security, status
from fastapi.security import APIKeyHeader

from app.api.problems import ProblemError
from app.core.config import Settings, get_settings

# auto_error=False: we raise our own problem+json 401 instead of FastAPI's plain 403.
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False, description="Shared API key")


def require_api_key(
    provided: Annotated[str | None, Security(_api_key_header)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    """Reject requests without the right API key.

    Args:
        provided: The X-API-Key header value.
        settings: Application settings.

    Raises:
        ProblemError: 401 when the key is missing or wrong; 503 when the server has no key configured.
    """
    if settings.api_key is None:
        # Fail closed: an unconfigured key must not mean "open to everyone".
        raise ProblemError(
            status.HTTP_503_SERVICE_UNAVAILABLE, "auth-not-configured", "Authentication not configured",
            "The server has no API_KEY configured, so protected endpoints are disabled.",
        )  # fmt: skip
    # compare_digest takes the same time wherever the strings differ, so response timing can't leak the key.
    if provided is None or not secrets.compare_digest(provided.encode(), settings.api_key.get_secret_value().encode()):
        raise ProblemError(
            status.HTTP_401_UNAUTHORIZED, "unauthorized", "Missing or invalid API key",
            "Send a valid key in the X-API-Key header.",
        )  # fmt: skip

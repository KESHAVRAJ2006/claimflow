"""RFC 7807 "problem details": one error shape for every failure the API returns.

Every error, whether ours, FastAPI's validation error or an unexpected crash, becomes
``application/problem+json`` with ``type``, ``title``, ``status``, ``detail``, ``instance`` and the request id, so the
frontend needs exactly one error parser and a user can quote the request id in a bug report.

``type`` is a URN rather than a URL: it identifies the kind of problem without pretending there is a documentation
page to fetch.
"""

import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import get_settings
from app.core.logging import request_id_var

logger = logging.getLogger(__name__)

PROBLEM_JSON = "application/problem+json"


class ProblemError(Exception):
    """Raise from a route or service to return a specific problem response."""

    def __init__(
        self, status_code: int, slug: str, title: str, detail: str, *, extra: dict[str, Any] | None = None
    ) -> None:
        """Describe the problem.

        Args:
            status_code: HTTP status.
            slug: Stable machine-readable kind, e.g. "claim-not-found"; becomes ``urn:claimflow:problem:<slug>``.
            title: Short summary that is the same for every occurrence of this kind.
            detail: Explanation specific to this occurrence.
            extra: Additional members, e.g. ``{"errors": [...]}``.
        """
        super().__init__(detail)
        self.status_code = status_code
        self.slug = slug
        self.title = title
        self.detail = detail
        self.extra = extra or {}


def problem_response(
    request: Request,
    status_code: int,
    slug: str,
    title: str,
    detail: str,
    extra: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Build a problem+json response.

    Args:
        request: The request that failed.
        status_code: HTTP status.
        slug: Problem kind.
        title: Short summary.
        detail: Occurrence-specific explanation.
        extra: Extension members.
        headers: Extra response headers (e.g. WWW-Authenticate).

    Returns:
        The response.
    """
    body = {
        "type": f"urn:claimflow:problem:{slug}",
        "title": title,
        "status": status_code,
        "detail": detail,
        "instance": request.url.path,
        "request_id": request_id_var.get(),
        **(extra or {}),
    }
    return JSONResponse(body, status_code=status_code, media_type=PROBLEM_JSON, headers=headers)


async def _problem_error(request: Request, error: Exception) -> JSONResponse:
    assert isinstance(error, ProblemError)  # noqa: S101 — registered for ProblemError only
    return problem_response(request, error.status_code, error.slug, error.title, error.detail, error.extra)


async def _http_error(request: Request, error: Exception) -> JSONResponse:
    assert isinstance(error, StarletteHTTPException)  # noqa: S101
    slug = {404: "not-found", 405: "method-not-allowed", 401: "unauthorized"}.get(error.status_code, "http-error")
    return problem_response(
        request, error.status_code, slug, str(error.detail), str(error.detail), headers=getattr(error, "headers", None)
    )


async def _validation_error(request: Request, error: Exception) -> JSONResponse:
    assert isinstance(error, RequestValidationError)  # noqa: S101
    # "input" is dropped: it can echo back a whole uploaded payload or a secret the client mistyped.
    errors = [
        {"location": list(item.get("loc", ())), "message": item.get("msg", ""), "type": item.get("type", "")}
        for item in error.errors()
    ]
    return problem_response(
        request, 422, "validation-error", "Request validation failed",
        f"{len(errors)} field(s) failed validation.", {"errors": errors},
    )  # fmt: skip


async def _unexpected_error(request: Request, error: Exception) -> JSONResponse:
    logger.exception("unhandled error", extra={"path": request.url.path})
    # In production the message could reveal hostnames, SQL or file paths; the request id links to the full log.
    detail = "An unexpected error occurred." if get_settings().is_production else f"{type(error).__name__}: {error}"
    return problem_response(request, status.HTTP_500_INTERNAL_SERVER_ERROR, "internal-error", "Internal error", detail)


def install_problem_handlers(app: FastAPI) -> None:
    """Route every error type through the problem+json renderer.

    Args:
        app: The application.
    """
    app.add_exception_handler(ProblemError, _problem_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(Exception, _unexpected_error)

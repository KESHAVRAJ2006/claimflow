"""ASGI middleware that assigns a request ID and logs every request."""

import logging
import re
import time
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.logging import request_id_var

logger = logging.getLogger("claimflow.request")

# Only accept simple client-supplied IDs; anything else could inject fake lines or control chars into logs.
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# Docker polls /api/health every few seconds; logging those at INFO would bury real traffic.
_QUIET_PATHS = frozenset({"/api/health"})


class RequestContextMiddleware:
    """Attach a request ID to logs and responses, and emit one log line per request.

    Written as raw ASGI rather than Starlette's BaseHTTPMiddleware because BaseHTTPMiddleware wraps the
    response body, which breaks or buffers the Server-Sent Events stream added in Phase 7.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Wrap the downstream ASGI application.

        Args:
            app: The next ASGI app in the chain.
        """
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI connection.

        Args:
            scope: Connection metadata (type, path, headers, ...).
            receive: Callable yielding incoming messages.
            send: Callable for outgoing messages.
        """
        # Lifespan and websocket events pass straight through.
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope["headers"])  # ASGI headers are a list of (lowercase bytes, bytes) pairs
        incoming = headers.get(b"x-request-id", b"").decode("latin-1")
        request_id = incoming if _SAFE_REQUEST_ID.match(incoming) else uuid.uuid4().hex
        token = request_id_var.set(request_id)

        start = time.perf_counter()
        status_code = 500  # stays 500 if the app raises before sending a response

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                # Echo the ID so a user reporting a problem can quote it and we can grep the logs.
                message["headers"] = [*message.get("headers", []), (b"x-request-id", request_id.encode())]
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            path = scope["path"]
            level = logging.DEBUG if path in _QUIET_PATHS and status_code < 400 else logging.INFO
            logger.log(
                level,
                "request completed",
                extra={
                    "method": scope["method"],
                    "path": path,
                    "status_code": status_code,
                    "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                },
            )
            request_id_var.reset(token)

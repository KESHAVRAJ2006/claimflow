"""Reject oversized request bodies before they are parsed.

Starlette's multipart parser writes the whole upload to a temporary file before the route handler runs, so a size
check inside the handler comes too late: a 2 GB upload would already be on disk. This middleware stops the request
as soon as the declared Content-Length, or the bytes actually received (for chunked uploads that declare none),
exceed the limit.
"""

import json

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.logging import request_id_var

# Multipart framing and the JSON form field add a little on top of the file itself.
MULTIPART_OVERHEAD_BYTES = 64 * 1024


class BodyTooLargeError(Exception):
    """The request body passed the limit while streaming."""


class BodySizeLimitMiddleware:
    """Enforce a maximum request body size with a problem+json 413."""

    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        """Wrap the application.

        Args:
            app: Downstream ASGI app.
            max_body_bytes: Largest body accepted, in bytes.
        """
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one connection.

        Args:
            scope: Connection metadata.
            receive: Incoming message callable.
            send: Outgoing message callable.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_body_bytes:
            await self._reject(scope, send)
            return

        received = 0
        response_started = False

        async def counting_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    raise BodyTooLargeError
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            response_started = response_started or message["type"] == "http.response.start"
            await send(message)

        try:
            await self.app(scope, counting_receive, tracking_send)
        except BodyTooLargeError:
            if not response_started:
                await self._reject(scope, send)

    async def _reject(self, scope: Scope, send: Send) -> None:
        """Send the 413 problem response."""
        body = json.dumps(
            {
                "type": "urn:claimflow:problem:payload-too-large",
                "title": "Request body too large",
                "status": 413,
                "detail": f"The request body exceeds {self.max_body_bytes} bytes.",
                "instance": scope["path"],
                "request_id": request_id_var.get(),
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/problem+json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

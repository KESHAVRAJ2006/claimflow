"""Outgoing webhooks to n8n, which turns triage results and human decisions into email and Slack messages.

Delivery is best effort and never blocks or fails the request that caused it: a notification outage must not stop a
reviewer from recording a decision. Every payload is signed (HMAC-SHA256 over the exact body bytes) so n8n can
reject a forged call to its webhook URL.
"""

import hashlib
import hmac
import json
import logging
from datetime import UTC, datetime
from typing import Any, Protocol

import httpx

logger = logging.getLogger(__name__)

SIGNATURE_HEADER = "X-ClaimFlow-Signature"
# n8n answers in milliseconds; anything slower is an outage we log, not something to wait on.
TIMEOUT_S = 5.0


class WebhookSender(Protocol):
    """Anything that can deliver an event (the real sender, or a recorder in tests)."""

    async def send(self, event: str, payload: dict[str, Any]) -> None:
        """Deliver one event."""


def sign(body: bytes, secret: str) -> str:
    """Compute the signature header value.

    Args:
        body: The exact request body bytes.
        secret: Shared secret.

    Returns:
        ``sha256=<hex digest>``, the format GitHub and Stripe use, which n8n's Crypto node can reproduce.
    """
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


class HttpWebhookSender:
    """POSTs events to one URL."""

    def __init__(self, url: str | None, secret: str | None, client: httpx.AsyncClient | None = None) -> None:
        """Configure the sender.

        Args:
            url: Webhook URL; None disables sending (events are only logged).
            secret: HMAC secret; None sends unsigned (development only).
            client: HTTP client to reuse; created lazily if None.
        """
        self._url = url
        self._secret = secret
        self._client = client

    async def send(self, event: str, payload: dict[str, Any]) -> None:
        """Deliver an event; log and swallow any failure.

        Args:
            event: Event name, e.g. "claim.triaged".
            payload: JSON-serialisable data.
        """
        if self._url is None:
            logger.info("webhook not configured; event not sent", extra={"event": event})
            return
        body = json.dumps(
            {"event": event, "sent_at": datetime.now(UTC).isoformat(), **payload}, default=str, separators=(",", ":")
        ).encode()
        headers = {"Content-Type": "application/json"}
        if self._secret:
            headers[SIGNATURE_HEADER] = sign(body, self._secret)
        try:
            self._client = self._client or httpx.AsyncClient(timeout=TIMEOUT_S)
            response = await self._client.post(self._url, content=body, headers=headers)
            if response.status_code == httpx.codes.UNAUTHORIZED:
                # The one failure with a known cause: n8n recomputed the signature and got a different one.
                logger.warning(
                    "webhook rejected by n8n: signature mismatch. WEBHOOK_SECRET here must equal "
                    "CLAIMFLOW_WEBHOOK_SECRET in n8n; after changing .env, recreate both containers",
                    extra={"event": event, "response": response.text[:200]},
                )
                return
            response.raise_for_status()
        except httpx.HTTPError as error:
            logger.warning("webhook delivery failed", extra={"event": event, "error": repr(error)})

    async def aclose(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()

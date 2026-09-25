"""Structured JSON logging with a per-request correlation ID."""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, TextIO

# A ContextVar is isolated per asyncio task, so concurrent requests never see each other's ID.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

# Attributes every LogRecord has; anything else on the record came from `extra=` and is worth emitting.
_STANDARD_ATTRS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {
    "message",
    "asctime",
    "taskName",
    "color_message",  # uvicorn adds an ANSI-coloured duplicate of the message
}


class JsonFormatter(logging.Formatter):
    """Render each log record as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise a log record to JSON.

        Args:
            record: The record produced by a logger call.

        Returns:
            A single-line JSON string.
        """
        payload: dict[str, Any] = {
            # Timezone-aware UTC so logs from different machines sort correctly.
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id is not None:
            payload["request_id"] = request_id
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # default=str handles Decimal, UUID and datetime values passed via `extra=`.
        return json.dumps(payload, default=str)


def configure_logging(level: str, stream: TextIO | None = None) -> None:
    """Route all application and uvicorn logs through a single JSON handler on stdout.

    Args:
        level: Minimum level name, e.g. ``"INFO"``.
        stream: Where to write; stdout when None. The MCP server passes stderr, because over the stdio
            transport stdout carries the protocol itself and one stray log line would corrupt it.
    """
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    # Uvicorn installs its own plain-text handlers; strip them so its logs propagate to our JSON handler.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = []
        uvicorn_logger.propagate = True
    # Our middleware logs requests with latency and request_id, so uvicorn's access log would be a duplicate.
    logging.getLogger("uvicorn.access").disabled = True
    # qdrant-client uses httpx, which logs every HTTP call at INFO and would drown out app logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

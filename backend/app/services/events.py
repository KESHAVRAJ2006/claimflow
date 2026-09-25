"""In-process event broker for triage runs, and Server-Sent Events formatting.

Each run's events are kept in order with increasing ids. A subscriber first receives the history, then live events,
so a browser that connects after the run started (or reconnects with ``Last-Event-ID``) misses nothing.

Single-process by design: the history lives in this process's memory. With several API workers, events would need
a shared bus (Redis pub/sub or Postgres LISTEN/NOTIFY). After a restart, the stream endpoint rebuilds the trace
from ``claim_runs`` instead (see ``replay_events``).
"""

import asyncio
import json
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Any

# Finished runs kept for late subscribers; older ones are replayed from the database instead.
MAX_FINISHED_RUNS = 200
# The stream closes after one of these; run_unavailable is only sent when replaying a claim with no run.
TERMINAL_EVENTS = frozenset({"run_completed", "run_failed", "run_unavailable"})


@dataclass
class _RunChannel:
    """Events of one claim's current run."""

    run_id: uuid.UUID
    history: list[dict[str, Any]] = field(default_factory=list)
    subscribers: set[asyncio.Queue[dict[str, Any] | None]] = field(default_factory=set)
    closed: bool = False


class EventBroker:
    """Publish/subscribe for run events, keyed by claim id."""

    def __init__(self) -> None:
        """Create an empty broker."""
        self._channels: OrderedDict[uuid.UUID, _RunChannel] = OrderedDict()

    def open(self, claim_id: uuid.UUID, run_id: uuid.UUID) -> None:
        """Start a new channel for a run, replacing any earlier run of the same claim.

        Args:
            claim_id: The claim.
            run_id: The new run.
        """
        self._channels[claim_id] = _RunChannel(run_id)
        self._channels.move_to_end(claim_id)
        finished = [key for key, channel in self._channels.items() if channel.closed]
        for key in finished[: max(0, len(finished) - MAX_FINISHED_RUNS)]:
            del self._channels[key]

    def is_active(self, claim_id: uuid.UUID) -> bool:
        """Whether the claim has a run that is still publishing.

        Args:
            claim_id: The claim.

        Returns:
            True while the run is in progress.
        """
        channel = self._channels.get(claim_id)
        return channel is not None and not channel.closed

    def has_history(self, claim_id: uuid.UUID) -> bool:
        """Whether this process holds events for the claim.

        Args:
            claim_id: The claim.

        Returns:
            True if a run (active or recently finished) is in memory.
        """
        return claim_id in self._channels

    def publish(self, claim_id: uuid.UUID, event: dict[str, Any]) -> dict[str, Any]:
        """Append an event and deliver it to live subscribers.

        Args:
            claim_id: The claim.
            event: Event payload; must contain ``type``.

        Returns:
            The event with its sequence ``id`` and ``run_id`` added.
        """
        channel = self._channels[claim_id]
        stamped = {**event, "id": len(channel.history) + 1, "run_id": str(channel.run_id)}
        channel.history.append(stamped)
        for queue in channel.subscribers:
            queue.put_nowait(stamped)
        return stamped

    def close(self, claim_id: uuid.UUID) -> None:
        """Mark a run finished and release its subscribers.

        Args:
            claim_id: The claim.
        """
        channel = self._channels.get(claim_id)
        if channel is None:
            return
        channel.closed = True
        for queue in channel.subscribers:
            queue.put_nowait(None)  # sentinel: no more events

    async def subscribe(self, claim_id: uuid.UUID, after_id: int = 0) -> AsyncIterator[dict[str, Any]]:
        """Yield past events after ``after_id``, then live ones until the run ends.

        Args:
            claim_id: The claim.
            after_id: Last event id the client already has (from Last-Event-ID).

        Yields:
            Events in order.
        """
        channel = self._channels.get(claim_id)
        if channel is None:
            return
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        # Snapshot and register together (no await in between), so no event can fall between history and live.
        history = [event for event in channel.history if event["id"] > after_id]
        if not channel.closed:
            channel.subscribers.add(queue)
        try:
            for event in history:
                yield event
            if channel.closed:
                return
            while (event := await queue.get()) is not None:
                if event["id"] > after_id:
                    yield event
        finally:
            channel.subscribers.discard(queue)


def format_sse(event: dict[str, Any]) -> str:
    """Encode one event as a Server-Sent Events frame.

    Args:
        event: Payload with ``type`` and, optionally, ``id``.

    Returns:
        ``id:``, ``event:`` and ``data:`` lines ending with a blank line. The browser's EventSource dispatches it
        to listeners of that event type and remembers the id for reconnects.
    """
    lines = []
    if "id" in event:
        lines.append(f"id: {event['id']}")
    lines.append(f"event: {event['type']}")
    # json.dumps never emits a raw newline, so the payload always fits on one "data:" line.
    lines.append(f"data: {json.dumps(event, default=str, separators=(',', ':'))}")
    return "\n".join(lines) + "\n\n"


def replay_events(runs: Iterable[dict[str, Any]], terminal: dict[str, Any]) -> list[dict[str, Any]]:
    """Rebuild a run's event stream from stored claim_runs rows (used after a restart or for old runs).

    Args:
        runs: claim_runs rows as dicts (agent_name, round, output, latency_ms, created_at), oldest first.
        terminal: The final event (run_completed / run_failed / run_unavailable).

    Returns:
        Events in the same shapes the live stream sends, each marked ``replayed``.
    """
    events: list[dict[str, Any]] = []
    for run in runs:
        base = {"node": run["agent_name"], "round": run["round"], "replayed": True}
        events.append({"type": "node_started", **base, "at": str(run["created_at"])})
        for call in run["output"].get("tool_calls", []) if run["agent_name"] == "investigator" else []:
            events.append({"type": "tool_call_started", "replayed": True, **call})
            events.append({"type": "tool_call_finished", "replayed": True, **call})
        events.append({"type": "node_finished", **base, "latency_ms": run["latency_ms"], "output": run["output"]})
    events.append({**terminal, "replayed": True})
    return [{**event, "id": index} for index, event in enumerate(events, start=1)]

"""Run the triage graph in the background: stream its events, persist every node run, record the outcome.

The API answers ``202 Accepted`` as soon as the claim is stored; the graph then runs as an asyncio task. While it
runs, graph events go to the EventBroker (the SSE stream reads them) and each finished node is written to
claim_runs. All writes go through ``app.services.claims``: deterministic code, never an agent.

Concurrency is capped by a semaphore because every run makes around ten LLM calls, and rate limits are per key.
"""

import asyncio
import logging
import uuid
from collections.abc import Callable
from contextlib import AsyncExitStack
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.graph import AgentDependencies, build_claim_graph, open_checkpointer
from app.agents.llm import LlmClient, LlmNotConfiguredError, create_llm_client
from app.agents.llm_errors import classify_llm_error, explain_llm_error
from app.agents.loaders import build_investigator_tools, rule_context_loader
from app.agents.state import ClaimInput, ClaimState
from app.api.problems import ProblemError
from app.core.config import Settings
from app.db.readonly import ReadOnlyDatabase, ReadOnlyViolationError
from app.retrieval.retriever import PolicyRetriever
from app.services import claims as claim_service
from app.services.events import EventBroker
from app.services.webhooks import WebhookSender

logger = logging.getLogger(__name__)

# A cancelled run still tries to record its failure, but must not hold up shutdown for long.
FAILURE_WRITE_TIMEOUT_S = 5.0


class TriageRunner:
    """Starts and tracks background graph runs."""

    def __init__(
        self,
        graph: Any,
        session_factory: async_sessionmaker[AsyncSession],
        broker: EventBroker,
        webhooks: WebhookSender,
        max_concurrent: int,
    ) -> None:
        """Wire the runner.

        Args:
            graph: The compiled triage graph.
            session_factory: Sessions for the owner database login (the only login that writes).
            broker: Event broker the SSE endpoint reads.
            webhooks: Notification sender.
            max_concurrent: Most runs in flight at once; the rest wait their turn.
        """
        self._graph = graph
        self._sessions = session_factory
        self._broker = broker
        self._webhooks = webhooks
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._tasks: dict[uuid.UUID, asyncio.Task[None]] = {}

    def is_running(self, claim_id: uuid.UUID) -> bool:
        """Whether a run for this claim is queued or in progress.

        Args:
            claim_id: The claim.

        Returns:
            True if a task exists and has not finished.
        """
        task = self._tasks.get(claim_id)
        return task is not None and not task.done()

    async def start(self, claim: ClaimInput) -> uuid.UUID:
        """Queue a run and return immediately.

        Args:
            claim: Graph input.

        Returns:
            The new run id.

        Raises:
            ProblemError: 409 if this claim is already being triaged.
        """
        if self.is_running(claim.claim_id):
            raise ProblemError(409, "triage-in-progress", "Triage already running", f"{claim.claim_number} is running.")
        run_id = uuid.uuid4()
        self._broker.open(claim.claim_id, run_id)
        self._broker.publish(claim.claim_id, {"type": "run_queued", "claim_id": str(claim.claim_id)})
        task = asyncio.create_task(self._run(claim, run_id), name=f"triage-{claim.claim_number}")
        self._tasks[claim.claim_id] = task
        task.add_done_callback(self._forget)
        return run_id

    def _forget(self, task: asyncio.Task[None]) -> None:
        """Drop a finished task from the registry (only if a newer run hasn't replaced it)."""
        for claim_id, current in list(self._tasks.items()):
            if current is task:
                del self._tasks[claim_id]

    async def wait(self, claim_id: uuid.UUID) -> None:
        """Wait for a claim's run to finish (tests and scripts).

        Args:
            claim_id: The claim.
        """
        task = self._tasks.get(claim_id)
        if task is not None:
            await asyncio.shield(task)

    async def shutdown(self) -> None:
        """Cancel every run; each records itself as FAILED so it can be re-run after restart."""
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _run(self, claim: ClaimInput, run_id: uuid.UUID) -> None:
        """Execute one run end to end. Never raises except CancelledError; failures are recorded instead."""
        claim_id = claim.claim_id
        try:
            async with self._semaphore:
                async with self._sessions() as session:
                    await claim_service.mark_processing(session, claim_id, run_id)
                config = {"configurable": {"thread_id": f"{claim_id}:{run_id}"}}
                state: ClaimState = {}
                async for mode, chunk in self._graph.astream(
                    {"claim": claim}, config, stream_mode=["custom", "updates", "values"]
                ):
                    if mode == "custom":
                        self._broker.publish(claim_id, chunk)
                    elif mode == "updates":
                        await self._persist_node_runs(claim_id, run_id, chunk)
                    else:
                        state = chunk  # the full state after each step; the last one is the final state
                async with self._sessions() as session:
                    stored = await claim_service.record_outcome(session, claim_id, run_id, state)
                final = state.get("final")
                self._broker.publish(claim_id, {
                    "type": "run_completed" if final else "run_failed",
                    "status": stored.status.value,
                    "outcome": final.outcome.value if final else None,
                    "reason": state.get("failure_reason"),
                })  # fmt: skip
                await self._webhooks.send("claim.triaged", {
                    "claim_id": str(claim_id), "claim_number": claim.claim_number, "status": stored.status.value,
                    "recommended_outcome": final.outcome.value if final else None,
                    "confidence": str(final.confidence) if final else None,
                    "risk_score": final.risk_score if final else None,
                    "reasons": list(final.reasons) if final else [state.get("failure_reason")],
                    "requires_human_review": True,
                })  # fmt: skip
        except asyncio.CancelledError:
            await self._fail(claim_id, run_id, "Triage was interrupted because the API shut down; re-run it.")
            raise
        except Exception as error:  # noqa: BLE001 — record any crash on the claim instead of losing it
            logger.exception("triage run failed", extra={"claim_id": str(claim_id), "run_id": str(run_id)})
            await self._fail(claim_id, run_id, explain_run_failure(error))
        finally:
            self._broker.close(claim_id)

    async def _persist_node_runs(self, claim_id: uuid.UUID, run_id: uuid.UUID, updates: dict[str, Any]) -> None:
        """Write the NodeRun of every node that just finished."""
        for update in updates.values():
            for node_run in (update or {}).get("node_runs", []):
                async with self._sessions() as session:
                    await claim_service.record_node_run(session, claim_id, run_id, node_run)

    async def _fail(self, claim_id: uuid.UUID, run_id: uuid.UUID, reason: str) -> None:
        """Publish and record a failed run, best effort."""
        self._broker.publish(claim_id, {"type": "run_failed", "status": "failed", "reason": reason})
        try:
            async with asyncio.timeout(FAILURE_WRITE_TIMEOUT_S), self._sessions() as session:
                await claim_service.record_failure(session, claim_id, run_id, reason)
        except Exception:  # noqa: BLE001 — the database may be the thing that failed; the log keeps the trail
            logger.exception("could not record triage failure", extra={"claim_id": str(claim_id)})


# asyncpg's names for "this login is wrong": a password rotated in .env but not in Postgres, or a missing role.
_LOGIN_ERRORS = ("invalidpassword", "invalidauthorization")


def explain_run_failure(error: BaseException) -> str:
    """The failure reason stored on the claim and shown to the reviewer.

    A provider failure or a broken read-only login gets its fix spelled out; anything else points at the server
    log, which has the traceback. The text never includes connection strings or provider payloads.

    Args:
        error: What the run raised.

    Returns:
        One or two sentences.
    """
    if classify_llm_error(error) is not None:
        return f"Triage failed. {explain_llm_error(error)}"
    name = type(error).__name__.lower()
    if isinstance(error, ReadOnlyViolationError) or any(fragment in name for fragment in _LOGIN_ERRORS):
        return (
            "Triage failed: the agents' read-only database login is not working. Run "
            "`docker compose exec backend python -m scripts.provision_readonly_role`, or "
            "`python -m scripts.doctor` for a full check."
        )
    return f"Triage failed with {type(error).__name__}; see the server log."


class TriageDisabledError(RuntimeError):
    """Triage cannot run in this process; the message says why and how to fix it."""


async def build_triage_runner(
    settings: Settings,
    stack: AsyncExitStack,
    *,
    session_factory: async_sessionmaker[AsyncSession],
    tools_database: ReadOnlyDatabase | None,
    retriever: PolicyRetriever | None,
    broker: EventBroker,
    webhooks: WebhookSender,
    llm_factory: Callable[[Settings], LlmClient] = create_llm_client,
) -> TriageRunner:
    """Assemble the runner if everything the agents need is configured.

    Args:
        settings: Application settings.
        stack: Exit stack owning the checkpointer connection for the app's lifetime.
        session_factory: Owner-login sessions.
        tools_database: Read-only database for the tools, or None if not configured.
        retriever: Policy retriever, or None if the embedding model is not loaded.
        broker: Event broker.
        webhooks: Webhook sender.
        llm_factory: Builds the LLM client (swapped in tests).

    Returns:
        The runner.

    Raises:
        TriageDisabledError: When triage cannot run; the message names the missing piece and the fix. Claims can
            still be submitted and decided without it.
    """
    try:
        llm = llm_factory(settings)
    except LlmNotConfiguredError as error:
        raise TriageDisabledError(
            "No LLM key is configured. Set GROQ_API_KEY or GOOGLE_API_KEY in .env, then run `docker compose up -d` "
            "so the container picks it up."
        ) from error
    if tools_database is None:
        raise TriageDisabledError(
            "The agents' read-only database login is not configured (TOOLS_DATABASE_URL, or AGENT_DB_USER and "
            "AGENT_DB_PASSWORD)."
        )
    if retriever is None:
        raise TriageDisabledError(
            "The embedding model is not loaded (LOAD_EMBEDDING_MODEL is false), so policy search is off."
        )
    checkpointer = await stack.enter_async_context(open_checkpointer(settings.checkpoint_path))
    deps = AgentDependencies(
        llm=llm,
        tools=build_investigator_tools(retriever, tools_database),
        load_rule_context=rule_context_loader(tools_database),
    )
    return TriageRunner(
        build_claim_graph(deps, checkpointer), session_factory, broker, webhooks, settings.max_concurrent_runs
    )

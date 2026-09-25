"""Claims API: submit, list, detail, live trace (SSE), (re)run triage, and the human decision.

Routes validate input and translate between HTTP and the services; they contain no business rules. The status
machine, the override check and every write live in ``app.services.claims``.
"""

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator
from datetime import date
from decimal import Decimal
from typing import Annotated, Any, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Header, Query, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.deps import (
    get_db_session,
    get_event_broker,
    get_session_factory,
    get_triage_disabled_reason,
    get_triage_runner,
    get_webhooks,
)
from app.api.problems import ProblemError
from app.api.security import require_api_key
from app.core.config import Settings, get_settings
from app.db.models import Claim
from app.domain.enums import ClaimStatus
from app.schemas.claims import (
    ClaimAccepted,
    ClaimDetail,
    ClaimPage,
    ClaimSort,
    ClaimSubmission,
    DecisionRequest,
    DecisionResponse,
    RunStarted,
)
from app.services import claims as claim_service
from app.services.events import TERMINAL_EVENTS, EventBroker, format_sse, replay_events
from app.services.triage import TriageRunner
from app.services.uploads import receive_claim_document
from app.services.webhooks import WebhookSender

router = APIRouter(prefix="/claims", tags=["claims"], dependencies=[Depends(require_api_key)])

Session = Annotated[AsyncSession, Depends(get_db_session)]
Runner = Annotated[TriageRunner | None, Depends(get_triage_runner)]
DisabledReason = Annotated[str, Depends(get_triage_disabled_reason)]
# A comment line every 15s keeps proxies (and Render's load balancer) from closing an idle stream.
HEARTBEAT_S = 15.0
# Tells EventSource how long to wait before reconnecting after a dropped connection.
RETRY_MS = 3000
RERUNNABLE = frozenset({ClaimStatus.SUBMITTED, ClaimStatus.FAILED})


def _stream_url(claim_id: uuid.UUID) -> str:
    return f"/api/claims/{claim_id}/stream"


def _parse_submission(raw: str) -> ClaimSubmission:
    """Validate the JSON ``claim`` form field, reporting errors in the same shape as body validation."""
    try:
        return ClaimSubmission.model_validate_json(raw)
    except ValidationError as error:
        raise RequestValidationError(
            [{**item, "loc": ("body", "claim", *item["loc"])} for item in error.errors(include_url=False)]
        ) from error


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ClaimAccepted,
    summary="Submit a claim (multipart: PDF + JSON)",
)
async def submit_claim(
    session: Session,
    runner: Runner,
    settings: Annotated[Settings, Depends(get_settings)],
    claim: Annotated[str, Form(description="ClaimSubmission as a JSON string")],
    document: Annotated[UploadFile, File(description="Supporting document, PDF, at most 10 MB")],
) -> ClaimAccepted:
    """Store a claim and start triage in the background.

    Order matters: the cheap JSON check runs first, then the document is validated and extracted (and deleted),
    and only a fully valid submission is written to the database.

    Args:
        session: Database session.
        runner: Triage runner, or None if agents are not configured.
        settings: Application settings.
        claim: The form as JSON.
        document: The uploaded PDF.

    Returns:
        202 with the claim id and where to stream its progress.
    """
    submission = _parse_submission(claim)
    extracted = await receive_claim_document(
        document, settings.upload_dir, settings.max_upload_bytes, settings.max_pdf_pages
    )
    stored = await claim_service.create_claim(session, submission, extracted.name)
    triage: Literal["queued", "unavailable"] = "unavailable"
    if runner is not None:
        await runner.start(await claim_service.claim_input(session, stored, extracted))
        triage = "queued"
    return ClaimAccepted(
        claim_id=stored.id, claim_number=stored.claim_number, status=stored.status, triage=triage,
        stream_url=_stream_url(stored.id),
    )  # fmt: skip


@router.get("", response_model=ClaimPage, summary="List claims (paginated, filterable, sortable)")
async def list_claims(
    session: Session,
    status_filter: Annotated[list[ClaimStatus] | None, Query(alias="status")] = None,
    date_from: Annotated[date | None, Query(description="Submitted on or after (UTC date)")] = None,
    date_to: Annotated[date | None, Query(description="Submitted on or before (UTC date)")] = None,
    amount_min: Annotated[Decimal | None, Query(ge=0)] = None,
    amount_max: Annotated[Decimal | None, Query(ge=0)] = None,
    q: Annotated[str | None, Query(max_length=40, description="Claim or policy number prefix")] = None,
    sort: ClaimSort = "created_at",
    order: Literal["asc", "desc"] = "desc",
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 25,
) -> ClaimPage:
    """One page of the claims queue.

    Args:
        session: Database session.
        status_filter: Statuses to include (repeat the parameter for several).
        date_from: Earliest submission date.
        date_to: Latest submission date.
        amount_min: Smallest claimed amount.
        amount_max: Largest claimed amount.
        q: Claim or policy number prefix.
        sort: Sort column.
        order: Sort direction.
        page: 1-based page number.
        page_size: Rows per page.

    Returns:
        The page and the total match count.
    """
    if date_from and date_to and date_from > date_to:
        raise ProblemError(422, "invalid-range", "Invalid date range", "date_from is after date_to.")
    if amount_min is not None and amount_max is not None and amount_min > amount_max:
        raise ProblemError(422, "invalid-range", "Invalid amount range", "amount_min is greater than amount_max.")
    filters = claim_service.ClaimFilters(
        statuses=status_filter or (), date_from=date_from, date_to=date_to, amount_min=amount_min,
        amount_max=amount_max, query=q, sort=sort, descending=order == "desc", page=page, page_size=page_size,
    )  # fmt: skip
    items, total = await claim_service.list_claims(session, filters)
    return ClaimPage(items=items, page=page, page_size=page_size, total=total)


@router.get("/{claim_id}", response_model=ClaimDetail, summary="Claim detail with trace, tool log and citations")
async def get_claim(claim_id: uuid.UUID, session: Session) -> ClaimDetail:
    """Everything the claim detail page shows.

    Args:
        claim_id: The claim.
        session: Database session.

    Returns:
        The detail view.
    """
    return await claim_service.get_claim_detail(session, claim_id)


def _terminal_event(claim: Claim, has_runs: bool, failure_reason: str | None = None) -> dict[str, Any]:
    """The closing event for a replayed stream, derived from the stored claim and its latest failure reason."""
    if not has_runs or claim.status in (ClaimStatus.SUBMITTED, ClaimStatus.PROCESSING):
        return {"type": "run_unavailable", "status": claim.status.value,
                "detail": "No triage run is available to stream for this claim."}  # fmt: skip
    if claim.status is ClaimStatus.FAILED:
        return {"type": "run_failed", "status": claim.status.value, "reason": failure_reason or "See the audit log."}
    outcome = claim.recommended_outcome.value if claim.recommended_outcome else None
    return {"type": "run_completed", "status": claim.status.value, "outcome": outcome}


async def _replayed(events: list[dict[str, Any]], after_id: int) -> AsyncIterator[dict[str, Any]]:
    for event in events:
        if event["id"] > after_id:
            yield event


async def _sse(events: AsyncIterator[dict[str, Any]], request: Request) -> AsyncIterator[str]:
    """Turn events into SSE frames, adding heartbeats and stopping when the client leaves or the run ends."""
    yield f"retry: {RETRY_MS}\n\n"
    iterator = aiter(events)
    pending: asyncio.Future[dict[str, Any]] = asyncio.ensure_future(anext(iterator))
    try:
        while True:
            done, _ = await asyncio.wait({pending}, timeout=HEARTBEAT_S)
            if not done:
                if await request.is_disconnected():
                    return
                yield ": keep-alive\n\n"  # a comment frame; EventSource ignores it
                continue
            try:
                event = pending.result()
            except StopAsyncIteration:
                return
            yield format_sse(event)
            if event["type"] in TERMINAL_EVENTS:
                return
            pending = asyncio.ensure_future(anext(iterator))
    finally:
        if not pending.done():
            pending.cancel()
            # cancel() only requests cancellation. Until it lands, the generator is still inside anext(), and
            # aclose() on it raises "asynchronous generator is already running". That happened on every client
            # disconnect mid-run and left the subscriber registered with the broker.
            with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
                await pending
        # Closing the generator unregisters this subscriber from the broker.
        if hasattr(iterator, "aclose"):
            await iterator.aclose()


@router.get(
    "/{claim_id}/stream",
    summary="Live trace (Server-Sent Events): one event per node and per tool call",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def stream_claim(
    claim_id: uuid.UUID,
    request: Request,
    sessions: Annotated[async_sessionmaker[AsyncSession], Depends(get_session_factory)],
    broker: Annotated[EventBroker, Depends(get_event_broker)],
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
) -> StreamingResponse:
    """Stream a claim's run: live if it is running here, otherwise replayed from claim_runs.

    Event types: run_queued, node_started, tool_call_started, tool_call_finished, node_finished, then exactly one
    of run_completed, run_failed or run_unavailable. Every event has an ``id``; a reconnecting EventSource sends it
    back as Last-Event-ID and receives only what it missed.

    Args:
        claim_id: The claim.
        request: The request (to detect client disconnects).
        sessions: Session factory; the session is closed before streaming starts, so a long stream holds no
            database connection.
        broker: Event broker.
        last_event_id: Last event the client received.

    Returns:
        A text/event-stream response.
    """
    after_id = int(last_event_id) if last_event_id and last_event_id.isdigit() else 0
    async with sessions() as session:
        claim = await claim_service.get_claim(session, claim_id)
        if broker.has_history(claim_id):
            source: AsyncIterator[dict[str, Any]] = broker.subscribe(claim_id, after_id)
        else:
            runs = await claim_service.run_rows(session, claim_id)
            reason = await claim_service.latest_failure_reason(session, claim_id)
            events = replay_events([run.model_dump() for run in runs], _terminal_event(claim, bool(runs), reason))
            source = _replayed(events, after_id)
    return StreamingResponse(
        _sse(source, request),
        media_type="text/event-stream",
        # no-transform + X-Accel-Buffering: stop proxies (nginx, Render) from buffering or compressing the stream.
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


@router.post(
    "/{claim_id}/run",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=RunStarted,
    summary="(Re)run triage for a stored claim (seeded, or after a failure)",
)
async def run_claim(
    claim_id: uuid.UUID, session: Session, runner: Runner, disabled_reason: DisabledReason
) -> RunStarted:
    """Start triage for a claim that was never triaged or whose run failed.

    The uploaded document is not kept after submission, so a re-run reads the form fields and description only.

    Args:
        claim_id: The claim.
        session: Database session.
        runner: Triage runner.
        disabled_reason: Why triage is off, when it is.

    Returns:
        202 with the stream URL.
    """
    if runner is None:
        raise ProblemError(503, "triage-unavailable", "Triage unavailable", disabled_reason)
    claim = await claim_service.get_claim(session, claim_id)
    if claim.status not in RERUNNABLE:
        raise ProblemError(
            409, "claim-not-rerunnable", "Claim cannot be re-run",
            f"Only submitted or failed claims can be triaged; this one is {claim.status.value}.",
        )  # fmt: skip
    await runner.start(await claim_service.claim_input(session, claim, None))
    return RunStarted(claim_id=claim.id, status=claim.status, stream_url=_stream_url(claim.id))


@router.post("/{claim_id}/decision", response_model=DecisionResponse, summary="Record the human reviewer's decision")
async def decide(
    claim_id: uuid.UUID,
    body: DecisionRequest,
    session: Session,
    webhooks: Annotated[WebhookSender, Depends(get_webhooks)],
    background: BackgroundTasks,
) -> DecisionResponse:
    """Approve, reject or request information. The human decision is binding and audited.

    Args:
        claim_id: The claim.
        body: The reviewer's action.
        session: Database session.
        webhooks: Notification sender.
        background: Runs the webhook after the response, so a slow n8n never delays the reviewer.

    Returns:
        The claim's new state.
    """
    result = await claim_service.decide(session, claim_id, body)
    claim = result.claim
    background.add_task(webhooks.send, result.action, {
        "claim_id": str(claim.id), "claim_number": claim.claim_number, "status": claim.status.value,
        "final_outcome": claim.final_outcome.value if claim.final_outcome else None,
        "recommended_outcome": claim.recommended_outcome.value if claim.recommended_outcome else None,
        "overridden": result.overridden, "reviewer": body.reviewer, "reason": body.reason,
    })  # fmt: skip
    return DecisionResponse(
        claim_id=claim.id, status=claim.status, final_outcome=claim.final_outcome, overridden=result.overridden,
        decided_by=claim.decided_by, decided_at=claim.decided_at,
    )  # fmt: skip

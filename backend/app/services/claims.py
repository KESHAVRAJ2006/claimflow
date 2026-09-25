"""Claim persistence and queries. Every database write in the triage lifecycle happens here, in plain code.

DETERMINISTIC BY DESIGN: agents recommend, this module records. Status transitions, the override check and every
audit entry are fixed code, so "who changed what, when and why" never depends on a model's output.

Lifecycle: SUBMITTED -> PROCESSING -> AWAITING_REVIEW | ESCALATED | FAILED -> APPROVED | REJECTED (human).
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import Any

from fastapi import status
from sqlalchemy import Select, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.state import (
    ClaimDocument,
    ClaimInput,
    ClaimState,
    EvidencePassage,
    FinalRecommendation,
    NodeRun,
    ToolCallRecord,
)
from app.api.problems import ProblemError
from app.db.models import AuditLog, Claim, ClaimRun, Policy
from app.domain.enums import ClaimStatus, FinalOutcome, RecommendedOutcome
from app.domain.products import INCIDENT_TYPES_BY_PRODUCT
from app.rules.models import RiskReport
from app.schemas.claims import (
    AuditEntryOut,
    ClaimDetail,
    ClaimRunOut,
    ClaimSort,
    ClaimSubmission,
    ClaimSummary,
    DecisionRequest,
)

SYSTEM_ACTOR = "system:triage"
# Claim numbers 900000+ are reserved for labelled seed edge cases (see scripts/seed.py).
MAX_CLAIM_SEQUENCE = 899_999
CLAIM_NUMBER_ATTEMPTS = 5
# A one-word "no" is not a justification an auditor can use.
MIN_REASON_LENGTH = 10
REVIEWABLE = frozenset({ClaimStatus.AWAITING_REVIEW, ClaimStatus.ESCALATED})
_SORT_COLUMNS = {
    "created_at": Claim.created_at,
    "claimed_amount": Claim.claimed_amount,
    "incident_date": Claim.incident_date,
    "risk_score": Claim.risk_score,
    "confidence": Claim.confidence,
}


def _not_found(claim_id: uuid.UUID) -> ProblemError:
    return ProblemError(status.HTTP_404_NOT_FOUND, "claim-not-found", "Claim not found", f"No claim {claim_id}.")


def _audit(session: AsyncSession, claim_id: uuid.UUID | None, actor: str, action: str, **fields: Any) -> None:
    session.add(AuditLog(claim_id=claim_id, actor=actor, action=action, **fields))


# ---- submission -------------------------------------------------------------------------------------------------


async def _next_claim_number(session: AsyncSession) -> str:
    """The next free claim number for the current year (CLM-YYYY-NNNNNN).

    Args:
        session: Database session.

    Returns:
        A claim number; uniqueness is finally enforced by the database constraint (see ``create_claim``).

    Raises:
        ProblemError: 503 if this year's range is exhausted.
    """
    prefix = f"CLM-{datetime.now(UTC).year}-"
    latest = await session.scalar(
        select(func.max(Claim.claim_number)).where(
            Claim.claim_number.startswith(prefix), Claim.claim_number.not_like(prefix + "9%")
        )
    )
    sequence = int(latest[-6:]) + 1 if latest else 1
    if sequence > MAX_CLAIM_SEQUENCE:
        raise ProblemError(503, "claim-numbers-exhausted", "Claim numbers exhausted", "This year's range is used up.")
    return f"{prefix}{sequence:06d}"


async def create_claim(session: AsyncSession, submission: ClaimSubmission, document_filename: str | None) -> Claim:
    """Store a submitted claim and its audit entry.

    Args:
        session: Database session (committed here).
        submission: The validated form.
        document_filename: Sanitised name of the uploaded document.

    Returns:
        The stored claim, status SUBMITTED.

    Raises:
        ProblemError: 422 for an unknown policy or an incident type the product cannot have.
    """
    policy = await session.scalar(select(Policy).where(Policy.policy_number == submission.policy_number))
    if policy is None:
        raise ProblemError(422, "unknown-policy", "Unknown policy", f"No policy {submission.policy_number} exists.")
    allowed = INCIDENT_TYPES_BY_PRODUCT[policy.product_type]
    if submission.incident_type not in allowed:
        raise ProblemError(
            422, "incident-type-not-covered-by-product", "Incident type does not fit the policy",
            f"A {policy.product_type.value} policy cannot have a {submission.incident_type.value} claim; "
            f"expected one of: {', '.join(t.value for t in allowed)}.",
        )  # fmt: skip
    # Read before the loop: a rollback expires loaded objects, and touching an expired attribute would trigger a
    # lazy load, which async SQLAlchemy cannot do (MissingGreenlet).
    policy_id = policy.id
    for _ in range(CLAIM_NUMBER_ATTEMPTS):
        claim = Claim(
            claim_number=await _next_claim_number(session),
            policy_id=policy_id,
            incident_type=submission.incident_type,
            incident_date=submission.incident_date,
            description=submission.description,
            claimed_amount=submission.claimed_amount,
            status=ClaimStatus.SUBMITTED,
            document_filename=document_filename,
        )
        session.add(claim)
        try:
            await session.flush()
        except IntegrityError:
            # Two submissions picked the same number at once; the unique constraint caught it. Try the next one.
            await session.rollback()
            continue
        _audit(
            session, claim.id, "customer_portal", "claim.submitted",
            after={"claim_number": claim.claim_number, "claimed_amount": str(claim.claimed_amount),
                   "status": claim.status.value, "document": document_filename},
        )  # fmt: skip
        await session.commit()
        # created_at is set by the database; load it now, since a lazy load later would fail in async code.
        await session.refresh(claim)
        return claim
    raise ProblemError(503, "claim-number-conflict", "Could not allocate a claim number", "Please retry.")


async def get_claim(session: AsyncSession, claim_id: uuid.UUID) -> Claim:
    """Load a claim or raise 404.

    Args:
        session: Database session.
        claim_id: The claim.

    Returns:
        The claim.

    Raises:
        ProblemError: 404 if it does not exist.
    """
    claim = await session.get(Claim, claim_id)
    if claim is None:
        raise _not_found(claim_id)
    return claim


async def claim_input(session: AsyncSession, claim: Claim, document: ClaimDocument | None) -> ClaimInput:
    """Build the graph input for a stored claim.

    Args:
        session: Database session.
        claim: The claim.
        document: Extracted upload text, if any.

    Returns:
        The graph input.
    """
    policy = await session.get_one(Policy, claim.policy_id)
    return ClaimInput(
        claim_id=claim.id, claim_number=claim.claim_number, policy_number=policy.policy_number,
        product_type=policy.product_type, incident_type=claim.incident_type, incident_date=claim.incident_date,
        claimed_amount=claim.claimed_amount, description=claim.description, submitted_at=claim.created_at,
        document=document,
    )  # fmt: skip


# ---- triage lifecycle ------------------------------------------------------------------------------------------


async def mark_processing(session: AsyncSession, claim_id: uuid.UUID, run_id: uuid.UUID) -> None:
    """Move a claim to PROCESSING at the start of a run.

    Args:
        session: Database session (committed here).
        claim_id: The claim.
        run_id: The run starting.
    """
    claim = await get_claim(session, claim_id)
    before = claim.status.value
    claim.status = ClaimStatus.PROCESSING
    _audit(session, claim_id, SYSTEM_ACTOR, "claim.triage_started", before={"status": before},
           after={"status": claim.status.value, "run_id": str(run_id)})  # fmt: skip
    await session.commit()


async def record_node_run(session: AsyncSession, claim_id: uuid.UUID, run_id: uuid.UUID, run: NodeRun) -> None:
    """Store one node execution as a claim_runs row (the UI's reasoning trace).

    Args:
        session: Database session (committed here).
        claim_id: The claim.
        run_id: The run it belongs to; stored in ``input`` because a claim can be triaged more than once.
        run: The node run from the graph.
    """
    session.add(
        ClaimRun(
            claim_id=claim_id,
            agent_name=run.agent_name,
            input={"run_id": str(run_id), "round": run.round, **run.input},
            output=run.output,
            latency_ms=run.latency_ms,
        )
    )
    await session.commit()


async def record_outcome(session: AsyncSession, claim_id: uuid.UUID, run_id: uuid.UUID, state: ClaimState) -> Claim:
    """Store the result of a finished run.

    Args:
        session: Database session (committed here).
        claim_id: The claim.
        run_id: The run.
        state: Final graph state.

    Returns:
        The updated claim.
    """
    claim = await get_claim(session, claim_id)
    before = {"status": claim.status.value}
    intake = state.get("intake")
    if intake is not None:
        claim.extracted_fields = {
            **intake.extraction.model_dump(mode="json"),
            "mismatches": [m.model_dump(mode="json") for m in intake.mismatches],
            "source": intake.source,
        }
    final = state.get("final")
    if state.get("status") == "failed" or final is None:
        claim.status = ClaimStatus.FAILED
        _audit(session, claim_id, SYSTEM_ACTOR, "claim.triage_failed", before=before,
               after={"status": claim.status.value, "run_id": str(run_id)},
               reason=state.get("failure_reason") or "The run ended without a recommendation.")  # fmt: skip
    else:
        claim.recommended_outcome = final.outcome
        claim.confidence = final.confidence
        claim.risk_score = final.risk_score
        claim.decision_rationale = final.summary + "\n\nRouting: " + " ".join(final.reasons)
        if final.open_questions:
            # What the AI could not look up; the reviewer checks it with the claimant before confirming.
            claim.decision_rationale += "\n\nOpen questions for the claimant: " + " ".join(
                f"({number}) {question}" for number, question in enumerate(final.open_questions, start=1)
            )
        # Escalations wait for a senior reviewer; approve/reject recommendations for any reviewer to confirm.
        claim.status = (
            ClaimStatus.ESCALATED if final.outcome is RecommendedOutcome.ESCALATE else ClaimStatus.AWAITING_REVIEW
        )
        _audit(session, claim_id, SYSTEM_ACTOR, "claim.triaged", before=before,
               after={"status": claim.status.value, "run_id": str(run_id), "recommended_outcome": final.outcome.value,
                      "confidence": str(final.confidence), "risk_score": final.risk_score})  # fmt: skip
    await session.commit()
    return claim


async def record_failure(session: AsyncSession, claim_id: uuid.UUID, run_id: uuid.UUID | None, reason: str) -> None:
    """Mark a claim FAILED after a run crashed or was interrupted.

    Args:
        session: Database session (committed here).
        claim_id: The claim.
        run_id: The run, if known.
        reason: What went wrong (no secrets; shown to reviewers).
    """
    claim = await get_claim(session, claim_id)
    before = {"status": claim.status.value}
    claim.status = ClaimStatus.FAILED
    after = {"status": claim.status.value, "run_id": str(run_id) if run_id else None}
    _audit(session, claim_id, SYSTEM_ACTOR, "claim.triage_failed", before=before, after=after, reason=reason[:1000])
    await session.commit()


async def latest_failure_reason(session: AsyncSession, claim_id: uuid.UUID) -> str | None:
    """The reason recorded for the claim's most recent failed run.

    Args:
        session: Database session.
        claim_id: The claim.

    Returns:
        The reason from the newest ``claim.triage_failed`` audit row, or None if the claim never failed.
    """
    statement = (
        select(AuditLog.reason)
        .where(AuditLog.claim_id == claim_id, AuditLog.action == "claim.triage_failed")
        .order_by(AuditLog.id.desc())
        .limit(1)
    )
    return await session.scalar(statement)


async def fail_interrupted_runs(session: AsyncSession) -> int:
    """At startup, fail claims left PROCESSING by a previous process (their in-memory run died with it).

    Args:
        session: Database session (committed here).

    Returns:
        How many claims were marked FAILED; they can be re-run with POST /api/claims/{id}/run.
    """
    stuck = (await session.scalars(select(Claim).where(Claim.status == ClaimStatus.PROCESSING))).all()
    for claim in stuck:
        claim.status = ClaimStatus.FAILED
        _audit(session, claim.id, SYSTEM_ACTOR, "claim.triage_failed", before={"status": "processing"},
               after={"status": "failed"}, reason="The API restarted while this claim was being triaged.")  # fmt: skip
    await session.commit()
    return len(stuck)


# ---- human decision --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionOutcome:
    """What a reviewer action changed."""

    claim: Claim
    action: str  # audit action written
    overridden: bool


async def decide(session: AsyncSession, claim_id: uuid.UUID, request: DecisionRequest) -> DecisionOutcome:
    """Record a reviewer's action. The human decision is binding; the AI recommendation is kept alongside it.

    Args:
        session: Database session (committed here).
        claim_id: The claim.
        request: The reviewer's action.

    Returns:
        The updated claim and what was recorded.

    Raises:
        ProblemError: 404 unknown claim; 409 claim not awaiting review; 422 missing or too-short reason.
    """
    # FOR UPDATE: two reviewers clicking at once are serialised, so the second sees the first's decision (409).
    claim = await session.scalar(select(Claim).where(Claim.id == claim_id).with_for_update())
    if claim is None:
        raise _not_found(claim_id)
    if claim.status not in REVIEWABLE:
        raise ProblemError(
            status.HTTP_409_CONFLICT, "claim-not-reviewable", "Claim is not awaiting review",
            f"Claim {claim.claim_number} is {claim.status.value}; only claims awaiting review or escalated can be "
            "decided.",
        )  # fmt: skip
    reason = (request.reason or "").strip() or None
    actor = f"reviewer:{request.reviewer}"
    recommended = claim.recommended_outcome

    if request.action == "request_info":
        _require_reason(reason, "Say what information is needed.")
        _audit(session, claim_id, actor, "claim.info_requested", after={"status": claim.status.value}, reason=reason)
        await session.commit()
        return DecisionOutcome(claim, "claim.info_requested", overridden=False)

    final = FinalOutcome(request.action)
    # An override is a human outcome that differs from a definite AI recommendation.
    overridden = recommended in (RecommendedOutcome.APPROVE, RecommendedOutcome.REJECT) and recommended.value != final
    if overridden:
        _require_reason(reason, "Overriding the AI recommendation requires a reason.")
    elif recommended in (None, RecommendedOutcome.ESCALATE):
        # The AI did not decide, so the human's reasoning is the only record of why.
        _require_reason(reason, "Deciding an escalated claim requires a reason.")
    before = {"status": claim.status.value, "recommended_outcome": recommended.value if recommended else None}
    claim.final_outcome = final
    claim.decided_by = actor
    claim.decided_at = datetime.now(UTC)
    claim.override_reason = reason
    claim.status = ClaimStatus.APPROVED if final is FinalOutcome.APPROVE else ClaimStatus.REJECTED
    action = "claim.overridden" if overridden else "claim.decided"
    _audit(session, claim_id, actor, action, before=before,
           after={"status": claim.status.value, "final_outcome": final.value}, reason=reason)  # fmt: skip
    await session.commit()
    return DecisionOutcome(claim, action, overridden)


def _require_reason(reason: str | None, message: str) -> None:
    if reason is None or len(reason) < MIN_REASON_LENGTH:
        raise ProblemError(
            422, "reason-required", "A reason is required", f"{message} Give at least {MIN_REASON_LENGTH} characters."
        )


# ---- queries ------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ClaimFilters:
    """Filters and paging for the claims queue."""

    statuses: Sequence[ClaimStatus] = ()
    date_from: date | None = None  # submission date, inclusive (UTC)
    date_to: date | None = None
    amount_min: Decimal | None = None
    amount_max: Decimal | None = None
    query: str | None = None  # claim or policy number prefix
    sort: ClaimSort = "created_at"
    descending: bool = True
    page: int = 1
    page_size: int = 25


def _filtered(statement: Select[Any], filters: ClaimFilters) -> Select[Any]:
    if filters.statuses:
        statement = statement.where(Claim.status.in_(filters.statuses))
    if filters.date_from:
        statement = statement.where(Claim.created_at >= datetime.combine(filters.date_from, time.min, UTC))
    if filters.date_to:
        # Exclusive upper bound at the next midnight, so the whole "to" day is included.
        statement = statement.where(
            Claim.created_at < datetime.combine(filters.date_to + timedelta(days=1), time.min, UTC)
        )
    if filters.amount_min is not None:
        statement = statement.where(Claim.claimed_amount >= filters.amount_min)
    if filters.amount_max is not None:
        statement = statement.where(Claim.claimed_amount <= filters.amount_max)
    if filters.query:
        term = filters.query.strip().upper()
        # autoescape: a "%" or "_" typed in the search box is matched literally, not as a wildcard.
        statement = statement.where(
            Claim.claim_number.startswith(term, autoescape=True)
            | Policy.policy_number.startswith(term, autoescape=True)
        )
    return statement


def _summary(claim: Claim, policy_number: str, product_type: Any) -> dict[str, Any]:
    """The queue-row fields of a claim, with its policy's number and product."""
    return {
        "id": claim.id,
        "claim_number": claim.claim_number,
        "policy_number": policy_number,
        "product_type": product_type,
        "incident_type": claim.incident_type,
        "incident_date": claim.incident_date,
        "claimed_amount": claim.claimed_amount,
        "status": claim.status,
        "recommended_outcome": claim.recommended_outcome,
        "confidence": claim.confidence,
        "risk_score": claim.risk_score,
        "final_outcome": claim.final_outcome,
        "created_at": claim.created_at,
    }


async def list_claims(session: AsyncSession, filters: ClaimFilters) -> tuple[list[ClaimSummary], int]:
    """One page of the claims queue.

    Args:
        session: Database session.
        filters: Filters, sort and paging (sort column comes from a fixed map, never from raw input).

    Returns:
        The page of summaries and the total number of matching claims.
    """
    base = select(Claim, Policy.policy_number, Policy.product_type).join(Policy, Policy.id == Claim.policy_id)
    base = _filtered(base, filters)
    total = await session.scalar(select(func.count()).select_from(base.order_by(None).subquery())) or 0
    column = _SORT_COLUMNS[filters.sort]
    ordering = column.desc().nulls_last() if filters.descending else column.asc().nulls_last()
    rows = await session.execute(
        base.order_by(ordering, Claim.claim_number.desc())
        .offset((filters.page - 1) * filters.page_size)
        .limit(filters.page_size)
    )
    items = [ClaimSummary.model_validate(_summary(claim, number, product)) for claim, number, product in rows]
    return items, total


async def run_rows(session: AsyncSession, claim_id: uuid.UUID) -> list[ClaimRunOut]:
    """The claim_runs of the claim's latest run, oldest first.

    Args:
        session: Database session.
        claim_id: The claim.

    Returns:
        Node runs of the most recent triage run (empty if never triaged).
    """
    rows = (
        await session.scalars(
            select(ClaimRun).where(ClaimRun.claim_id == claim_id).order_by(ClaimRun.created_at, ClaimRun.id)
        )
    ).all()
    runs = [
        ClaimRunOut(
            id=row.id,
            run_id=uuid.UUID(row.input["run_id"]) if row.input.get("run_id") else None,
            agent_name=row.agent_name,
            round=int(row.input.get("round", 1)),
            input=row.input,
            output=row.output,
            latency_ms=row.latency_ms,
            created_at=row.created_at,
        )
        for row in rows
    ]
    latest = runs[-1].run_id if runs else None
    return [run for run in runs if run.run_id == latest]


async def get_claim_detail(session: AsyncSession, claim_id: uuid.UUID) -> ClaimDetail:
    """Everything the claim detail page shows, assembled from the claim, its latest run and its audit log.

    Args:
        session: Database session.
        claim_id: The claim.

    Returns:
        The detail view.

    Raises:
        ProblemError: 404 if the claim does not exist.
    """
    row = (
        await session.execute(
            select(Claim, Policy.policy_number, Policy.product_type)
            .join(Policy, Policy.id == Claim.policy_id)
            .where(Claim.id == claim_id)
        )
    ).one_or_none()
    if row is None:
        raise _not_found(claim_id)
    claim, policy_number, product_type = row
    runs = await run_rows(session, claim_id)
    tool_calls: list[ToolCallRecord] = []
    evidence: list[EvidencePassage] = []
    risk_report: RiskReport | None = None
    recommendation: FinalRecommendation | None = None
    for run in runs:
        if run.agent_name == "investigator":
            tool_calls += [ToolCallRecord.model_validate(call) for call in run.output.get("tool_calls", [])]
            evidence += [EvidencePassage.model_validate(p) for p in run.output.get("new_evidence", [])]
        elif run.agent_name == "rules":
            # Only "results" is stored state; risk_score and hard_blocks are recomputed from it.
            risk_report = RiskReport.model_validate({"results": run.output["results"]})
        elif run.agent_name == "route_final":
            recommendation = FinalRecommendation.model_validate(run.output)
    audit = (await session.scalars(select(AuditLog).where(AuditLog.claim_id == claim_id).order_by(AuditLog.id))).all()
    return ClaimDetail.model_validate(
        {
            **_summary(claim, policy_number, product_type),
            "description": claim.description,
            "document_filename": claim.document_filename,
            "extracted_fields": claim.extracted_fields,
            "decision_rationale": claim.decision_rationale,
            "decided_by": claim.decided_by,
            "decided_at": claim.decided_at,
            "override_reason": claim.override_reason,
            "run_id": runs[-1].run_id if runs else None,
            "runs": runs,
            "tool_call_log": tool_calls,
            "evidence": evidence,
            "risk_report": risk_report,
            "recommendation": recommendation,
            "audit": [AuditEntryOut.model_validate(entry) for entry in audit],
        }
    )

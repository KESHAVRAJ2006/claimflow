"""Read-only snapshots served as MCP resources: the policy list and the claims waiting for a human.

Both are read through the agents' SELECT-only login, like the tools. They leave out every column a client model
does not need: no customer names or contact details, and no free text. A claim description is written by the
claimant and the stored rationale by an LLM; either could carry instructions aimed at the client's model, so
neither is served here. A client that needs the facts of a claim uses the tools, whose output is fenced.
"""

from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select

from app.db.models import Claim, Policy
from app.db.readonly import ReadOnlyDatabase
from app.domain.enums import ClaimStatus, IncidentType, PolicyStatus, ProductType, RecommendedOutcome

# A resource is read in one piece into the client's context; 200 rows is about 30k characters of JSON.
MAX_RESOURCE_ROWS = 200

# Waiting for a person: not yet decided by a human, and not failed at intake (those need a resubmission).
PENDING_STATUSES = (
    ClaimStatus.SUBMITTED,
    ClaimStatus.PROCESSING,
    ClaimStatus.AWAITING_REVIEW,
    ClaimStatus.ESCALATED,
)


class _Snapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PolicyListItem(_Snapshot):
    """A policy's record, without any customer data."""

    policy_number: str
    product_type: ProductType
    status: PolicyStatus
    start_date: date
    end_date: date
    lapsed_on: date | None
    sum_insured: Decimal
    policy_document: str


class PolicyList(_Snapshot):
    """The policies resource. ``truncated`` says whether more policies exist than are listed."""

    total: int
    truncated: bool
    policies: tuple[PolicyListItem, ...]


class PendingClaim(_Snapshot):
    """A claim waiting for a human, with the AI's advisory recommendation when there is one."""

    claim_number: str
    policy_number: str
    product_type: ProductType
    incident_type: IncidentType
    incident_date: date
    claimed_amount: Decimal
    status: ClaimStatus
    recommended_outcome: RecommendedOutcome | None
    confidence: Decimal | None
    risk_score: int | None
    submitted_at: datetime


class PendingClaims(_Snapshot):
    """The pending-claims resource, oldest first (the order a review queue is worked in)."""

    total: int
    truncated: bool
    note: str
    claims: tuple[PendingClaim, ...]


async def fetch_policy_list(database: ReadOnlyDatabase, limit: int = MAX_RESOURCE_ROWS) -> PolicyList:
    """Read the policy records, ordered by policy number.

    Args:
        database: Read-only database.
        limit: Maximum policies to list.

    Returns:
        The snapshot.
    """
    statement = (
        select(
            Policy.policy_number,
            Policy.product_type,
            Policy.status,
            Policy.start_date,
            Policy.end_date,
            Policy.lapsed_on,
            Policy.sum_insured,
            Policy.policy_document,
        )
        .order_by(Policy.policy_number)
        .limit(limit)
    )
    async with database.connect() as connection:
        total = (await connection.execute(select(func.count()).select_from(Policy))).scalar_one()
        rows = (await connection.execute(statement)).mappings().all()
    items = tuple(PolicyListItem.model_validate(dict(row)) for row in rows)
    return PolicyList(total=total, truncated=total > len(items), policies=items)


async def fetch_pending_claims(database: ReadOnlyDatabase, limit: int = MAX_RESOURCE_ROWS) -> PendingClaims:
    """Read the claims that still need a human, oldest first.

    Args:
        database: Read-only database.
        limit: Maximum claims to list.

    Returns:
        The snapshot.
    """
    pending = Claim.status.in_(PENDING_STATUSES)
    statement = (
        select(
            Claim.claim_number,
            Policy.policy_number,
            Policy.product_type,
            Claim.incident_type,
            Claim.incident_date,
            Claim.claimed_amount,
            Claim.status,
            Claim.recommended_outcome,
            Claim.confidence,
            Claim.risk_score,
            Claim.created_at.label("submitted_at"),
        )
        .join(Policy, Policy.id == Claim.policy_id)
        .where(pending)
        .order_by(Claim.created_at, Claim.claim_number)
        .limit(limit)
    )
    async with database.connect() as connection:
        total = (await connection.execute(select(func.count()).select_from(Claim).where(pending))).scalar_one()
        rows = (await connection.execute(statement)).mappings().all()
    items = tuple(PendingClaim.model_validate(dict(row)) for row in rows)
    return PendingClaims(
        total=total,
        truncated=total > len(items),
        note=(
            "recommended_outcome is an AI-assisted recommendation awaiting human review, not a decision. "
            "Descriptions are omitted; use the tools for a claim's facts."
        ),
        claims=items,
    )

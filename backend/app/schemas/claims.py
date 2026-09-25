"""Request and response models for the claims API.

Money is ``Decimal`` and serialises as a string ("42000.00"); JSON numbers are binary floats and would lose cents.
Agent contracts (ToolCallRecord, EvidencePassage, FinalRecommendation, RiskReport) are reused as-is, so the API
cannot drift from what the graph produced.
"""

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.agents.state import DISCLAIMER, EvidencePassage, FinalRecommendation, ToolCallRecord
from app.domain.enums import ClaimStatus, FinalOutcome, IncidentType, ProductType, RecommendedOutcome
from app.rules.models import RiskReport

ClaimSort = Literal["created_at", "claimed_amount", "incident_date", "risk_score", "confidence"]


class ClaimSubmission(BaseModel):
    """The claim form, sent as the JSON ``claim`` field of the multipart upload."""

    model_config = ConfigDict(extra="forbid")

    policy_number: str = Field(pattern=r"^(MOT|HLT|HOM)-\d{4}-\d{6}$", examples=["MOT-2025-000011"])
    incident_type: IncidentType
    # No "not in the future" check here: that is rule R05's job, and the rules engine must see the claim to flag it.
    incident_date: date
    claimed_amount: Decimal = Field(gt=0, max_digits=14, decimal_places=2, examples=["42000.00"])
    description: str = Field(min_length=10, max_length=2000)


class ClaimAccepted(BaseModel):
    """202 response: the claim is stored; triage runs in the background."""

    claim_id: uuid.UUID
    claim_number: str
    status: ClaimStatus
    triage: Literal["queued", "unavailable"] = Field(
        description="unavailable when no LLM provider is configured; the claim is stored either way"
    )
    stream_url: str


class ClaimSummary(BaseModel):
    """One row of the claims queue."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
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
    final_outcome: FinalOutcome | None
    created_at: datetime


class ClaimPage(BaseModel):
    """A page of the claims queue."""

    items: list[ClaimSummary]
    page: int
    page_size: int
    total: int


class ClaimRunOut(BaseModel):
    """One node execution in the reasoning trace."""

    id: uuid.UUID
    run_id: uuid.UUID | None
    agent_name: str
    round: int
    input: dict[str, Any]
    output: dict[str, Any]
    latency_ms: int
    created_at: datetime


class AuditEntryOut(BaseModel):
    """One audit log row."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    actor: str
    action: str
    before: dict[str, Any] | None
    after: dict[str, Any] | None
    reason: str | None
    created_at: datetime


class ClaimDetail(ClaimSummary):
    """Everything the claim detail page shows."""

    description: str
    document_filename: str | None
    extracted_fields: dict[str, Any] | None
    decision_rationale: str | None
    decided_by: str | None
    decided_at: datetime | None
    override_reason: str | None
    run_id: uuid.UUID | None = Field(description="The latest triage run; the fields below come from it")
    runs: list[ClaimRunOut]
    tool_call_log: list[ToolCallRecord]
    evidence: list[EvidencePassage]
    risk_report: RiskReport | None
    recommendation: FinalRecommendation | None
    audit: list[AuditEntryOut]
    disclaimer: str = DISCLAIMER


class DecisionRequest(BaseModel):
    """A human reviewer's action on a triaged claim."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["approve", "reject", "request_info"]
    reviewer: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,59}$", examples=["priya.nair"])
    reason: str | None = Field(
        default=None,
        max_length=2000,
        description="Required (10+ characters) to override the AI recommendation, to decide an escalated claim, "
        "and for request_info (the question to ask)",
    )


class DecisionResponse(BaseModel):
    """Result of a reviewer action."""

    claim_id: uuid.UUID
    status: ClaimStatus
    final_outcome: FinalOutcome | None
    overridden: bool
    decided_by: str | None
    decided_at: datetime | None


class RunStarted(BaseModel):
    """202 response when triage is (re)started for a stored claim."""

    claim_id: uuid.UUID
    status: ClaimStatus
    stream_url: str

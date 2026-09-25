"""Typed contracts between the agents, and the LangGraph state that carries them.

Agents exchange these Pydantic models only, never free-form prose. A model that must be filled by an LLM
(``IntakeExtraction``, ``InvestigationFindings``, ``DecisionDraft``, ``Critique``) is validated the moment it
arrives, so a malformed answer fails at the boundary where it can be retried, instead of three nodes later.
Models filled by code are frozen; nothing downstream can edit an upstream result in place.

Money crosses the LLM boundary as a string of digits and is converted to Decimal by code. Confidence arrives
as a float from the LLM and is converted to a 3-place Decimal before any threshold is compared.
"""

import operator
import uuid
from datetime import date
from decimal import Decimal
from typing import Annotated, Any, Literal, TypedDict

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from app.domain.enums import IncidentType, ProductType, RecommendedOutcome
from app.domain.products import INCIDENT_TYPES_BY_PRODUCT
from app.rules.models import RiskReport

# --- loop limits ---------------------------------------------------------------------------------------------
# 6 tool-calling steps: the most complex seeded claim needs about 5 lookups; more steps mostly buys repetition.
MAX_INVESTIGATION_STEPS = 6
# Reflection may send the claim back for more evidence at most twice; a third failure escalates instead.
MAX_REFLECTION_RETRIES = 2

DISCLAIMER = (
    "AI-assisted recommendation for a human reviewer. It is not an automated determination; a person makes "
    "the final decision."
)

# Policy number prefix -> product, so intake can check the incident type fits the product without a lookup.
PRODUCT_BY_PREFIX = {"MOT": ProductType.MOTOR, "HLT": ProductType.HEALTH, "HOM": ProductType.HOME}


class _Frozen(BaseModel):
    """A result produced by code: immutable, and unknown fields are an error."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class _LlmSchema(BaseModel):
    """A schema an LLM fills in. Unknown fields are rejected, so a hallucinated field fails validation."""

    model_config = ConfigDict(extra="forbid")


# ---- input -----------------------------------------------------------------------------------------------------


class DocumentPage(_Frozen):
    """Text of one page of the uploaded claim document."""

    page: int = Field(ge=1)
    text: str


class ClaimDocument(_Frozen):
    """The claimant's supporting PDF, already extracted to text (the file itself is deleted after upload)."""

    name: str
    pages: tuple[DocumentPage, ...]


class ClaimInput(_Frozen):
    """What the graph is asked to triage: the stored claim, as submitted on the form."""

    claim_id: uuid.UUID
    claim_number: str
    policy_number: str
    product_type: ProductType
    incident_type: IncidentType
    incident_date: date
    claimed_amount: Decimal = Field(gt=0, max_digits=14, decimal_places=2)
    description: str
    submitted_at: AwareDatetime
    document: ClaimDocument | None = None


# ---- intake ----------------------------------------------------------------------------------------------------


class IntakeExtraction(_LlmSchema):
    """Structured fields of an insurance claim, extracted from the claim form and the claimant's document."""

    policy_number: str = Field(
        pattern=r"^(MOT|HLT|HOM)-\d{4}-\d{6}$", description="Policy number, e.g. MOT-2025-000123"
    )
    incident_type: IncidentType = Field(description="The type of loss")
    incident_date: date = Field(description="Date the incident happened, YYYY-MM-DD")
    claimed_amount: str = Field(
        pattern=r"^\d{1,12}(\.\d{1,2})?$",
        description="Amount claimed: digits with up to 2 decimals, no currency symbol or commas, e.g. 45000.00",
    )
    incident_summary: str = Field(
        min_length=10, max_length=600, description="Neutral one-paragraph account of what happened"
    )
    incident_location: str | None = Field(default=None, max_length=200, description="Where it happened, if stated")
    supporting_documents: list[str] = Field(
        default_factory=list, max_length=10, description="Documents mentioned, e.g. FIR, hospital bill, repair estimate"
    )
    red_flags: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Statements that are vague, inconsistent or unusual; empty if none",
    )

    @model_validator(mode="after")
    def _consistent(self) -> "IntakeExtraction":
        if Decimal(self.claimed_amount) <= 0:
            raise ValueError("claimed_amount must be greater than zero")
        product = PRODUCT_BY_PREFIX[self.policy_number[:3]]
        allowed = INCIDENT_TYPES_BY_PRODUCT[product]
        if self.incident_type not in allowed:
            raise ValueError(
                f"incident_type {self.incident_type.value!r} is not possible on a {product.value} policy; "
                f"allowed: {', '.join(t.value for t in allowed)}"
            )
        return self


class FieldMismatch(_Frozen):
    """A field where the extraction disagrees with the form. Computed by code, not judged by the LLM."""

    field: str
    form_value: str
    extracted_value: str


class IntakeResult(_Frozen):
    """Validated intake output."""

    extraction: IntakeExtraction
    claimed_amount: Decimal
    mismatches: tuple[FieldMismatch, ...]
    source: Literal["form_only", "form_and_document"]
    attempts: int


# ---- investigation ------------------------------------------------------------------------------------------------


class ToolCallRecord(_Frozen):
    """One tool call made by the investigator. The UI's nested trace is built from these."""

    call_id: str = Field(description="T1, T2, ... in call order across all rounds; cited by the decision agent")
    round: int = Field(ge=1)
    step: int = Field(ge=1)
    tool: str
    args: dict[str, Any]
    status: Literal["ok", "error"]
    result_summary: str
    result_data: dict[str, Any] | None = Field(
        default=None, description="Structured result (policy searches: evidence ids and confidence only)"
    )
    latency_ms: int = Field(ge=0)
    started_at: AwareDatetime


class EvidencePassage(_Frozen):
    """A retrieved policy passage the decision may cite."""

    evidence_id: str = Field(description="E1, E2, ... in order of first retrieval")
    chunk_id: str
    document: str
    page: int
    section: str | None
    text: str
    similarity: float
    retrieval_confidence: float = Field(ge=0, le=1, description="Calibrated confidence of this passage's match")
    found_by: str = Field(description="call_id of the search that first returned it")

    @property
    def label(self) -> str:
        """Citation label, e.g. ``Motor_Policy.pdf p.3``."""
        return f"{self.document} p.{self.page}"


class PolicyFinding(_LlmSchema):
    """A statement about the policy wording, with its source."""

    statement: str = Field(max_length=400)
    document: str = Field(description="Source document, e.g. Motor_Policy.pdf")
    page: int = Field(ge=1)


class InvestigationFindings(_LlmSchema):
    """Submit your investigation findings. Call this on its own, once you have enough evidence or cannot usefully gather more."""  # noqa: E501 — the LLM reads this as the tool description

    summary: str = Field(min_length=10, max_length=1000, description="What you found, in plain words")
    key_facts: list[str] = Field(
        default_factory=list, max_length=12, description="Facts from tool results, each naming the tool it came from"
    )
    policy_findings: list[PolicyFinding] = Field(
        default_factory=list, max_length=8, description="What the wording says that matters for this claim"
    )
    concerns: list[str] = Field(default_factory=list, max_length=8, description="Anything that needs attention")
    evidence_gaps: list[str] = Field(
        default_factory=list, max_length=8, description="What you could not establish, and why"
    )


class InvestigationRound(_Frozen):
    """One run of the investigator."""

    round: int = Field(ge=1)
    findings: InvestigationFindings
    steps_used: int
    stopped_early: bool
    stop_reason: str | None
    tool_call_ids: tuple[str, ...]
    brief: tuple[str, ...] = Field(default=(), description="What reflection asked this round to address")


# ---- decision -----------------------------------------------------------------------------------------------------


class RationalePoint(_LlmSchema):
    """One reason behind the decision, with where it comes from."""

    statement: str = Field(max_length=400)
    basis: Literal["policy_wording", "claim_data", "claim_form", "rule_result"] = Field(
        description=(
            "policy_wording: from a cited passage; claim_data: from a tool result; claim_form: what the claimant "
            "stated on the form or in their description/document; rule_result: a rule"
        )
    )
    evidence_ids: list[str] = Field(
        default_factory=list, description="For policy_wording: the E ids of the passages relied on"
    )
    quote: str | None = Field(
        default=None,
        max_length=300,
        description="For policy_wording: words copied exactly from the cited passage that support the statement",
    )
    tool_call_ids: list[str] = Field(default_factory=list, description="For claim_data: the T ids of tool results")


class DecisionDraft(_LlmSchema):
    """Your coverage judgment for the claim, with its reasons."""

    covered: bool = Field(description="Whether the policy wording covers this loss, as the cited evidence shows")
    confidence: float = Field(ge=0, le=1, description="Confidence in the covered judgment, 0 to 1")
    summary: str = Field(min_length=10, max_length=800, description="Two or three sentences for a human reviewer")
    rationale: list[RationalePoint] = Field(min_length=1, max_length=8)


# ---- reflection ---------------------------------------------------------------------------------------------------


# What the critic names as the tool for a fact that only the claimant, a surveyor or a document could supply.
NO_TOOL = "none"


class EvidenceRequest(_LlmSchema):
    """One lookup that could change the coverage judgment."""

    tool: str = Field(
        description=f'The tool that can answer it, from the list given; "{NO_TOOL}" when only the claimant, a surveyor '
        "or a document could"
    )
    lookup: str = Field(description="What to look up, e.g. 'waiting period for cataract surgery'")


class Critique(_LlmSchema):
    """Your review of whether the decision is grounded in the evidence."""

    grounded: bool = Field(description="True if every statement follows from the cited evidence or tool results")
    unsupported_statements: list[str] = Field(default_factory=list, max_length=8)
    missing_evidence: list[EvidenceRequest] = Field(default_factory=list, max_length=6)
    verdict: Literal["accept", "investigate_more"]


class ReflectionResult(_Frozen):
    """Outcome of one reflection round."""

    round: int
    citation_problems: tuple[str, ...] = Field(description="Found by code before any LLM review")
    critique: Critique | None = Field(description="None when the code check already failed, so no LLM call was made")
    action: Literal["accept", "retry_investigation", "escalate"]
    feedback: tuple[str, ...] = Field(description="What the next investigation round must address")
    open_questions: tuple[str, ...] = Field(
        default=(),
        description="Facts no tool can supply, or already looked for once: they go to the human reviewer instead of "
        "sending the claim back",
    )


# ---- final --------------------------------------------------------------------------------------------------------


class ResolvedCitation(_Frozen):
    """A citation resolved from an evidence id to its document, page and quoted text."""

    evidence_id: str
    document: str
    page: int
    section: str | None
    quote: str | None
    label: str


class FinalRecommendation(_Frozen):
    """What the pipeline recommends to the human reviewer. Produced by deterministic routing only."""

    outcome: RecommendedOutcome
    reasons: tuple[str, ...]
    covered: bool | None
    agent_confidence: Decimal | None
    retrieval_confidence: Decimal
    confidence: Decimal = Field(description="The lower of agent and retrieval confidence; what routing compared")
    risk_score: int
    hard_blocks: tuple[str, ...]
    triggered_rules: tuple[str, ...]
    summary: str
    citations: tuple[ResolvedCitation, ...]
    escalation_flags: tuple[str, ...]
    open_questions: tuple[str, ...] = Field(default=(), description="What the reviewer should check with the claimant")
    requires_human_review: Literal[True] = True
    disclaimer: str = DISCLAIMER


class NodeRun(_Frozen):
    """One node execution; becomes a claim_runs row in Phase 7."""

    agent_name: str
    round: int
    input: dict[str, Any]
    output: dict[str, Any]
    latency_ms: int = Field(ge=0)
    started_at: AwareDatetime


# ---- graph state --------------------------------------------------------------------------------------------------


class ClaimState(TypedDict, total=False):
    """The state LangGraph carries between nodes and checkpoints after each one.

    Lists annotated with ``operator.add`` are append-only: a node returns only its new items and LangGraph
    concatenates them, so a retry round can never overwrite round 1's trace.
    """

    claim: ClaimInput
    status: Literal["running", "failed", "completed"]
    failure_reason: str | None
    intake: IntakeResult | None
    investigation_round: int
    investigations: Annotated[list[InvestigationRound], operator.add]
    tool_call_log: Annotated[list[ToolCallRecord], operator.add]
    evidence: Annotated[list[EvidencePassage], operator.add]
    risk_report: RiskReport
    decision: DecisionDraft | None
    reflections: Annotated[list[ReflectionResult], operator.add]
    # Every reason the claim must go to a human regardless of what the agents concluded (loop limits, failures).
    escalation_flags: Annotated[list[str], operator.add]
    final: FinalRecommendation | None
    node_runs: Annotated[list[NodeRun], operator.add]

"""Typed inputs and outputs of the rules engine.

All models are frozen: once the rules node has produced a RiskReport, nothing downstream (an agent included)
can mutate it in place.
"""

import uuid
from datetime import date
from decimal import Decimal
from enum import StrEnum

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, computed_field, model_validator

from app.domain.enums import KycStatus
from app.rules.thresholds import MAX_RISK_SCORE

# Money in, money out: exact decimals with at most 2 places, matching NUMERIC(14, 2) in the database.
MoneyField = Field(gt=0, max_digits=14, decimal_places=2)


class Severity(StrEnum):
    """How bad a triggered rule is, for display and sorting. Routing uses weights and hard blocks, not this."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class _FrozenModel(BaseModel):
    """Immutable model that rejects unknown fields, so a typo can't silently drop data."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ClaimFacts(_FrozenModel):
    """The claim being evaluated."""

    claim_id: uuid.UUID
    policy_id: uuid.UUID
    incident_date: date
    claimed_amount: Decimal = MoneyField
    submitted_at: AwareDatetime
    submitted_on: date = Field(
        description=(
            "Submission date in the insurer's business timezone. Supplied by the caller because 'today' "
            "depends on the timezone, and rules never read the clock."
        )
    )


class PolicyFacts(_FrozenModel):
    """The policy the claim is made under."""

    policy_id: uuid.UUID
    start_date: date
    end_date: date
    lapsed_on: date | None = None
    sum_insured: Decimal = MoneyField

    @model_validator(mode="after")
    def _end_after_start(self) -> "PolicyFacts":
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        return self


class CustomerFacts(_FrozenModel):
    """The policyholder."""

    customer_id: uuid.UUID
    kyc_status: KycStatus


class PriorClaim(_FrozenModel):
    """Another claim by the same customer, on any of their policies."""

    claim_id: uuid.UUID
    policy_id: uuid.UUID
    incident_date: date
    claimed_amount: Decimal = MoneyField
    submitted_at: AwareDatetime


class RuleContext(_FrozenModel):
    """Everything the rules may look at. Rules see nothing else, which is what makes them reproducible."""

    claim: ClaimFacts
    policy: PolicyFacts
    customer: CustomerFacts
    other_claims: tuple[PriorClaim, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> "RuleContext":
        if self.claim.policy_id != self.policy.policy_id:
            raise ValueError("claim.policy_id does not match policy.policy_id")
        # Counting the claim against itself would make R04 over-count and R07 flag every claim as a duplicate.
        if any(other.claim_id == self.claim.claim_id for other in self.other_claims):
            raise ValueError("other_claims must not include the claim being evaluated")
        return self


class RuleResult(_FrozenModel):
    """The outcome of one rule. Produced for every rule, triggered or not, so the UI can show all checks."""

    rule_id: str = Field(pattern=r"^R\d{2}$")
    name: str
    triggered: bool
    severity: Severity
    hard_block: bool = Field(description="A triggered hard block makes the claim ineligible (routes to reject)")
    score_contribution: int = Field(ge=0, le=MAX_RISK_SCORE, description="Points added to risk_score")
    explanation: str = Field(description="Plain-English reason citing the actual values compared")
    evidence: dict[str, str] = Field(description="The values the rule compared, as strings (exact decimals)")


class RiskReport(_FrozenModel):
    """Results of all rules plus the derived score and blocks.

    risk_score and hard_blocks are computed from ``results`` rather than stored, so a report can never
    carry a score that disagrees with its own rule results.
    """

    results: tuple[RuleResult, ...]

    @model_validator(mode="after")
    def _unique_rules(self) -> "RiskReport":
        ids = [result.rule_id for result in self.results]
        if len(ids) != len(set(ids)):
            raise ValueError("each rule may appear only once in a RiskReport")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def triggered_rule_ids(self) -> list[str]:
        """IDs of every triggered rule, in rule order."""
        return [result.rule_id for result in self.results if result.triggered]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def hard_blocks(self) -> list[str]:
        """IDs of triggered eligibility rules; non-empty means the claim must be rejected."""
        return [result.rule_id for result in self.results if result.triggered and result.hard_block]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def risk_score(self) -> int:
        """Sum of triggered risk weights, capped at 100.

        Additive scoring is deliberately simple: an auditor can recompute it by hand from the rule list.
        """
        return min(MAX_RISK_SCORE, sum(result.score_contribution for result in self.results))

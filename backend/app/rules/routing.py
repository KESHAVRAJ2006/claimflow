"""Final routing: turn rule results and the decision agent's judgment into APPROVE / REJECT / ESCALATE.

This is the boundary between agentic and deterministic. The decision agent contributes exactly two values,
``confidence`` and ``covered``. Everything else, and the order in which conditions are applied, is fixed
code: an agent that reports ``covered=True`` with confidence 0.99 still cannot get past a hard block or a
high risk score, because those come from the RiskReport, which only the rules engine produces.
"""

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import RecommendedOutcome
from app.rules import thresholds as t
from app.rules.models import RiskReport


class RoutingInput(BaseModel):
    """Everything routing needs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    risk_report: RiskReport
    claimed_amount: Decimal = Field(gt=0, max_digits=14, decimal_places=2)
    confidence: Decimal = Field(ge=0, le=1, description="Decision agent confidence, or retrieval confidence if lower")
    covered: bool = Field(description="Decision agent's reading of whether the policy covers this loss")
    forced_escalation_reason: str | None = Field(
        default=None,
        description="Set when an agent loop limit was hit; an agent that cannot finish hands off, it does not guess",
    )


class RoutingDecision(BaseModel):
    """The routed outcome and every condition that applied, highest priority first."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome: RecommendedOutcome
    reasons: tuple[str, ...] = Field(min_length=1)


def route_final(routing: RoutingInput) -> RoutingDecision:
    """Apply the routing table in fixed priority order.

    Escalation conditions are checked before rejection conditions: when the system is unsure, or the stakes
    are high, a human decides even if a rule says reject.

    Args:
        routing: Rule results plus the decision agent's confidence and coverage judgment.

    Returns:
        The outcome, with all applicable reasons (the first one is decisive).
    """
    report = routing.risk_report
    escalate: list[str] = []
    if routing.forced_escalation_reason:
        escalate.append(f"Agent loop limit reached: {routing.forced_escalation_reason}")
    if routing.confidence < t.MIN_DECISION_CONFIDENCE:
        escalate.append(f"Confidence {routing.confidence} is below the {t.MIN_DECISION_CONFIDENCE} minimum.")
    if routing.claimed_amount > t.AUTO_DECISION_MAX_AMOUNT:
        escalate.append(
            f"Claimed amount {routing.claimed_amount:,.2f} exceeds the {t.AUTO_DECISION_MAX_AMOUNT:,.2f} "
            "auto-decision limit."
        )
    if report.risk_score >= t.RISK_ESCALATION_THRESHOLD:
        triggered = ", ".join(r.rule_id for r in report.results if r.triggered and r.score_contribution)
        escalate.append(f"Risk score {report.risk_score} is at or above {t.RISK_ESCALATION_THRESHOLD} ({triggered}).")

    reject: list[str] = []
    if report.hard_blocks:
        reject.append(f"Eligibility rule(s) failed: {', '.join(report.hard_blocks)}.")
    if not routing.covered:
        reject.append("The decision agent found that the policy does not cover this loss.")

    if escalate:
        return RoutingDecision(outcome=RecommendedOutcome.ESCALATE, reasons=(*escalate, *reject))
    if reject:
        return RoutingDecision(outcome=RecommendedOutcome.REJECT, reasons=tuple(reject))
    return RoutingDecision(
        outcome=RecommendedOutcome.APPROVE,
        reasons=(
            f"No hard blocks, risk score {report.risk_score}, confidence {routing.confidence}, "
            f"amount within the auto-decision limit, and the loss is covered.",
        ),
    )

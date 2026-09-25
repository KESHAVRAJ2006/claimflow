"""The eight deterministic claim rules.

Each rule is a pure function ``RuleContext -> Outcome`` registered with ``@rule``. Metadata (severity,
weight, hard block) lives in the decorator, so the rule body only answers "did this happen, and why?".

Two kinds of rule, kept separate on purpose:
- Hard blocks (R02, R05, R06) are eligibility failures: the claim cannot be paid under the contract.
  They contribute 0 to the risk score, because routing checks escalation conditions before reject
  conditions; if a hard block also raised the score, a clear rejection would be escalated instead.
- Risk signals (R01, R03, R04, R07, R08) suggest fraud or error. They add weighted points to the score.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal

from app.domain.enums import KycStatus
from app.rules import thresholds as t
from app.rules.models import RuleContext, RuleResult, Severity


@dataclass(frozen=True)
class Outcome:
    """What a rule body returns: whether it fired, why, and the values it compared."""

    triggered: bool
    explanation: str
    evidence: dict[str, str]


CheckFunction = Callable[[RuleContext], Outcome]


@dataclass(frozen=True)
class Rule:
    """A registered rule: metadata plus its check function."""

    rule_id: str
    name: str
    severity: Severity
    hard_block: bool
    weight: int
    check: CheckFunction

    def evaluate(self, context: RuleContext) -> RuleResult:
        """Run the rule and package the outcome with its metadata.

        Args:
            context: Facts about the claim.

        Returns:
            The RuleResult, with score_contribution set only when a risk signal fires.
        """
        outcome = self.check(context)
        return RuleResult(
            rule_id=self.rule_id,
            name=self.name,
            triggered=outcome.triggered,
            severity=self.severity,
            hard_block=self.hard_block,
            score_contribution=self.weight if outcome.triggered else 0,
            explanation=outcome.explanation,
            evidence=outcome.evidence,
        )


_REGISTRY: list[Rule] = []


def rule(
    rule_id: str, *, severity: Severity, weight: int = 0, hard_block: bool = False
) -> Callable[[CheckFunction], CheckFunction]:
    """Register a check function as a rule.

    Args:
        rule_id: Stable identifier such as ``"R01"``; referenced by the UI, audit log and evaluation.
        severity: Display severity.
        weight: Risk points added when triggered (risk signals only).
        hard_block: Whether a trigger makes the claim ineligible.

    Returns:
        A decorator that registers the function and returns it unchanged, so it stays directly testable.

    Raises:
        ValueError: If a hard block is given a weight, or an ID is registered twice.
    """
    if hard_block and weight:
        raise ValueError(f"{rule_id}: hard blocks must have weight 0 (see module docstring)")
    if any(existing.rule_id == rule_id for existing in _REGISTRY):
        raise ValueError(f"{rule_id} is already registered")

    def register(function: CheckFunction) -> CheckFunction:
        _REGISTRY.append(Rule(rule_id, function.__name__, severity, hard_block, weight, function))
        return function

    return register


def _money(amount: Decimal) -> str:
    """Format money for explanations, e.g. ``650,000.00``."""
    return f"{amount:,.2f}"


@rule("R01", severity=Severity.HIGH, weight=t.WEIGHT_AMOUNT_EXCEEDS_SUM_INSURED)
def amount_exceeds_sum_insured(context: RuleContext) -> Outcome:
    """R01: the claimed amount is more than the policy can ever pay."""
    amount, insured = context.claim.claimed_amount, context.policy.sum_insured
    evidence = {"claimed_amount": str(amount), "sum_insured": str(insured)}
    if amount > insured:
        return Outcome(
            True,
            f"Claimed {_money(amount)} exceeds the sum insured of {_money(insured)} by {_money(amount - insured)}.",
            evidence,
        )
    return Outcome(False, f"Claimed {_money(amount)} is within the sum insured of {_money(insured)}.", evidence)


@rule("R02", severity=Severity.CRITICAL, hard_block=True)
def policy_lapsed_at_incident_date(context: RuleContext) -> Outcome:
    """R02: the policy had lapsed on or before the incident date, so there was no cover."""
    lapsed_on, incident = context.policy.lapsed_on, context.claim.incident_date
    evidence = {"lapsed_on": str(lapsed_on) if lapsed_on else "", "incident_date": str(incident)}
    if lapsed_on is None:
        return Outcome(False, "The policy has not lapsed.", evidence)
    # lapsed_on is the first day without cover, so an incident on that very day is already uncovered.
    if lapsed_on <= incident:
        return Outcome(
            True, f"The policy lapsed on {lapsed_on}, on or before the incident on {incident}; no cover applied.",
            evidence,
        )  # fmt: skip
    return Outcome(
        False, f"The policy lapsed on {lapsed_on}, after the incident on {incident}; cover was in force.", evidence
    )


@rule("R03", severity=Severity.MEDIUM, weight=t.WEIGHT_NEW_POLICY_CLAIM)
def claim_within_30d_of_policy_start(context: RuleContext) -> Outcome:
    """R03: the incident happened within 30 days of the policy starting."""
    start, incident = context.policy.start_date, context.claim.incident_date
    days = (incident - start).days
    evidence = {"policy_start_date": str(start), "incident_date": str(incident), "days_after_start": str(days)}
    if days < 0:
        # Incidents before the start are R06's job; flagging them here too would double-report one problem.
        return Outcome(False, "The incident precedes the policy start date (see R06).", evidence)
    if days <= t.NEW_POLICY_WINDOW_DAYS:
        return Outcome(
            True,
            f"The incident occurred {days} days after the policy started, within the "
            f"{t.NEW_POLICY_WINDOW_DAYS}-day new-policy window.",
            evidence,
        )
    return Outcome(False, f"The incident occurred {days} days after the policy started.", evidence)


@rule("R04", severity=Severity.MEDIUM, weight=t.WEIGHT_FREQUENT_CLAIMANT)
def more_than_3_claims_in_12_months(context: RuleContext) -> Outcome:
    """R04: including this one, the customer has more than 3 claims with incidents in the preceding 12 months."""
    incident = context.claim.incident_date
    window_start = incident - timedelta(days=t.CLAIM_FREQUENCY_WINDOW_DAYS)
    # Only incidents up to this claim's date count: a later claim must not change an earlier claim's result.
    related = sorted(
        (other for other in context.other_claims if window_start <= other.incident_date <= incident),
        key=lambda other: other.incident_date,
    )
    total = len(related) + 1  # + this claim
    evidence = {
        "window_start": str(window_start),
        "window_end": str(incident),
        "claims_in_window": str(total),
        "related_claim_ids": ",".join(str(other.claim_id) for other in related),
    }
    if total > t.MAX_CLAIMS_IN_WINDOW:
        return Outcome(
            True,
            f"This is claim {total} by the customer with an incident between {window_start} and {incident} "
            f"(limit {t.MAX_CLAIMS_IN_WINDOW}).",
            evidence,
        )
    return Outcome(
        False, f"{total} claim(s) by the customer with an incident between {window_start} and {incident}.", evidence
    )


@rule("R05", severity=Severity.CRITICAL, hard_block=True)
def incident_date_in_future(context: RuleContext) -> Outcome:
    """R05: the incident date is after the date the claim was submitted."""
    incident, submitted_on = context.claim.incident_date, context.claim.submitted_on
    evidence = {"incident_date": str(incident), "submitted_on": str(submitted_on)}
    if incident > submitted_on:
        return Outcome(True, f"The incident date {incident} is after the submission date {submitted_on}.", evidence)
    return Outcome(False, f"The incident date {incident} is not after the submission date {submitted_on}.", evidence)


@rule("R06", severity=Severity.CRITICAL, hard_block=True)
def incident_date_before_policy_start(context: RuleContext) -> Outcome:
    """R06: the incident happened before the policy started."""
    start, incident = context.policy.start_date, context.claim.incident_date
    evidence = {"policy_start_date": str(start), "incident_date": str(incident)}
    if incident < start:
        return Outcome(
            True,
            f"The incident on {incident} is {(start - incident).days} days before the policy started on {start}.",
            evidence,
        )
    return Outcome(False, f"The incident on {incident} is on or after the policy start on {start}.", evidence)


@rule("R07", severity=Severity.HIGH, weight=t.WEIGHT_DUPLICATE_CLAIM)
def duplicate_claim_same_date_amount(context: RuleContext) -> Outcome:
    """R07: an earlier-submitted claim on the same policy has the same incident date and amount."""
    claim = context.claim
    # Only earlier submissions count, so the original claim is never flagged as a duplicate of its copy.
    duplicates = [
        other
        for other in context.other_claims
        if other.policy_id == claim.policy_id
        and other.incident_date == claim.incident_date
        and other.claimed_amount == claim.claimed_amount
        and other.submitted_at < claim.submitted_at
    ]
    evidence = {
        "incident_date": str(claim.incident_date),
        "claimed_amount": str(claim.claimed_amount),
        "duplicate_claim_ids": ",".join(str(other.claim_id) for other in duplicates),
    }
    if duplicates:
        return Outcome(
            True,
            f"{len(duplicates)} earlier claim(s) on this policy have the same incident date {claim.incident_date} "
            f"and amount {_money(claim.claimed_amount)}.",
            evidence,
        )
    return Outcome(False, "No earlier claim on this policy has the same incident date and amount.", evidence)


@rule("R08", severity=Severity.HIGH, weight=t.WEIGHT_KYC_INCOMPLETE)
def kyc_incomplete(context: RuleContext) -> Outcome:
    """R08: the customer's identity (KYC) is not verified."""
    status = context.customer.kyc_status
    evidence = {"kyc_status": status.value}
    if status is not KycStatus.VERIFIED:
        return Outcome(True, f"Customer KYC status is '{status.value}'; payment requires verified KYC.", evidence)
    return Outcome(False, "Customer KYC is verified.", evidence)


# Sorted by ID so the report order never depends on the order functions happen to be defined in.
RULES: tuple[Rule, ...] = tuple(sorted(_REGISTRY, key=lambda registered: registered.rule_id))

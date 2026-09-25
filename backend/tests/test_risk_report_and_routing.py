"""Tests for evaluate_all (scoring, hard blocks, report shape) and route_final (the routing table)."""

from datetime import date, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.domain.enums import KycStatus, RecommendedOutcome
from app.rules import RoutingInput, evaluate_all, route_final
from app.rules.models import RiskReport, RuleResult, Severity
from tests.test_rules import INCIDENT, START, SUBMITTED_AT, make_context, prior_claim

# ---- evaluate_all ---------------------------------------------------------------------------------


def test_report_contains_every_rule_in_order_even_when_nothing_triggers() -> None:
    report = evaluate_all(make_context())
    assert [r.rule_id for r in report.results] == [f"R0{n}" for n in range(1, 9)]
    assert report.triggered_rule_ids == [] and report.hard_blocks == [] and report.risk_score == 0


def test_single_serious_signal_reaches_escalation_threshold() -> None:
    assert evaluate_all(make_context(kyc_status=KycStatus.PENDING)).risk_score == 70


def test_weak_signals_escalate_only_in_combination() -> None:
    new_policy = make_context(incident_date=START + timedelta(days=12))
    assert evaluate_all(new_policy).risk_score == 40

    frequent = tuple(prior_claim(START + timedelta(days=d)) for d in (2, 5, 8))
    both = make_context(incident_date=START + timedelta(days=12), other_claims=frequent)
    report = evaluate_all(both)
    assert report.triggered_rule_ids == ["R03", "R04"]
    assert report.risk_score == 75


def test_hard_blocks_add_nothing_to_risk_score() -> None:
    report = evaluate_all(make_context(lapsed_on=date(2026, 3, 1)))
    assert report.hard_blocks == ["R02"]
    assert report.risk_score == 0


def test_risk_score_is_capped_at_100() -> None:
    original = prior_claim(INCIDENT, amount="650000.00", submitted_at=SUBMITTED_AT - timedelta(days=1))
    context = make_context(claimed_amount="650000.00", kyc_status=KycStatus.PENDING, other_claims=(original,))
    report = evaluate_all(context)
    assert {"R01", "R07", "R08"} <= set(report.triggered_rule_ids)  # 70 + 70 + 70
    assert report.risk_score == 100


def test_report_serialises_to_json_with_computed_fields_and_exact_decimals() -> None:
    payload = evaluate_all(make_context(claimed_amount="650000.00")).model_dump(mode="json")
    assert payload["risk_score"] == 70 and payload["triggered_rule_ids"] == ["R01"]
    assert payload["results"][0]["evidence"]["claimed_amount"] == "650000.00"


def test_evaluation_is_deterministic() -> None:
    context = make_context(incident_date=START + timedelta(days=3), kyc_status=KycStatus.PENDING)
    assert evaluate_all(context) == evaluate_all(context)


def test_report_rejects_duplicate_rule_ids() -> None:
    result = RuleResult(
        rule_id="R01", name="x", triggered=False, severity=Severity.LOW, hard_block=False,
        score_contribution=0, explanation="x", evidence={},
    )  # fmt: skip
    with pytest.raises(ValidationError, match="only once"):
        RiskReport(results=(result, result))


def test_report_is_immutable() -> None:
    report = evaluate_all(make_context())
    with pytest.raises(ValidationError):
        report.results = ()  # type: ignore[misc]


# ---- route_final ----------------------------------------------------------------------------------


def route(
    *,
    context_overrides: dict[str, object] | None = None,
    amount: str = "50000.00",
    confidence: str | float = "0.90",
    covered: bool = True,
    forced: str | None = None,
) -> RoutingInput:
    report = evaluate_all(make_context(claimed_amount=amount, **(context_overrides or {})))  # type: ignore[arg-type]
    return RoutingInput(
        risk_report=report,
        claimed_amount=Decimal(amount),
        confidence=confidence,  # type: ignore[arg-type]
        covered=covered,
        forced_escalation_reason=forced,
    )


def test_clean_covered_confident_claim_is_approved() -> None:
    assert route_final(route()).outcome is RecommendedOutcome.APPROVE


def test_low_confidence_escalates() -> None:
    decision = route_final(route(confidence="0.64"))
    assert decision.outcome is RecommendedOutcome.ESCALATE
    assert "below" in decision.reasons[0]


def test_confidence_exactly_at_threshold_is_not_escalated() -> None:
    # A float from an LLM's structured output must compare exactly at the boundary, not as 0.6499999.
    assert route_final(route(confidence=0.65)).outcome is RecommendedOutcome.APPROVE


def test_amount_above_limit_escalates_and_amount_at_limit_does_not() -> None:
    assert route_final(route(amount="100000.01")).outcome is RecommendedOutcome.ESCALATE
    assert route_final(route(amount="100000.00")).outcome is RecommendedOutcome.APPROVE


def test_risk_score_at_threshold_escalates() -> None:
    decision = route_final(route(context_overrides={"kyc_status": KycStatus.PENDING}))
    assert decision.outcome is RecommendedOutcome.ESCALATE
    assert "R08" in decision.reasons[0]


def test_hard_block_rejects_even_when_agent_says_covered_with_high_confidence() -> None:
    decision = route_final(route(context_overrides={"lapsed_on": date(2026, 3, 1)}, confidence="0.99", covered=True))
    assert decision.outcome is RecommendedOutcome.REJECT
    assert decision.reasons == ("Eligibility rule(s) failed: R02.",)


def test_not_covered_rejects() -> None:
    assert route_final(route(covered=False)).outcome is RecommendedOutcome.REJECT


def test_escalation_takes_priority_over_rejection_and_keeps_all_reasons() -> None:
    decision = route_final(route(context_overrides={"lapsed_on": date(2026, 3, 1)}, amount="150000.00"))
    assert decision.outcome is RecommendedOutcome.ESCALATE
    assert len(decision.reasons) == 2 and "R02" in decision.reasons[1]


def test_loop_limit_forces_escalation_over_an_otherwise_clean_approval() -> None:
    decision = route_final(route(forced="investigator reached 6 iterations"))
    assert decision.outcome is RecommendedOutcome.ESCALATE
    assert decision.reasons[0].startswith("Agent loop limit reached")


def test_routing_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValidationError):
        route(confidence="1.2")

"""Unit tests for each rule: a trigger case, a no-trigger case, and the boundaries in between.

Every test starts from a claim that triggers nothing and changes exactly one fact, so a failure points at
one rule.
"""

import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.domain.enums import KycStatus
from app.rules import checks
from app.rules.models import ClaimFacts, CustomerFacts, PolicyFacts, PriorClaim, RuleContext

POLICY_ID = uuid.UUID("11111111-1111-4111-8111-111111111111")
OTHER_POLICY_ID = uuid.UUID("22222222-2222-4222-8222-222222222222")
CLAIM_ID = uuid.UUID("33333333-3333-4333-8333-333333333333")
START = date(2026, 1, 1)
INCIDENT = date(2026, 6, 1)  # 151 days after start: clear of R03 and R06
SUBMITTED_AT = datetime(2026, 6, 5, 10, 0, tzinfo=UTC)


def make_context(
    *,
    incident_date: date = INCIDENT,
    claimed_amount: str = "50000.00",
    start_date: date = START,
    lapsed_on: date | None = None,
    sum_insured: str = "500000.00",
    kyc_status: KycStatus = KycStatus.VERIFIED,
    submitted_at: datetime = SUBMITTED_AT,
    submitted_on: date | None = None,
    other_claims: tuple[PriorClaim, ...] = (),
) -> RuleContext:
    return RuleContext(
        claim=ClaimFacts(
            claim_id=CLAIM_ID,
            policy_id=POLICY_ID,
            incident_date=incident_date,
            claimed_amount=Decimal(claimed_amount),
            submitted_at=submitted_at,
            submitted_on=submitted_on or submitted_at.date(),
        ),
        policy=PolicyFacts(
            policy_id=POLICY_ID,
            start_date=start_date,
            end_date=start_date + timedelta(days=364),
            lapsed_on=lapsed_on,
            sum_insured=Decimal(sum_insured),
        ),
        customer=CustomerFacts(customer_id=uuid.uuid4(), kyc_status=kyc_status),
        other_claims=other_claims,
    )


def prior_claim(
    incident_date: date,
    amount: str = "20000.00",
    policy_id: uuid.UUID = POLICY_ID,
    submitted_at: datetime | None = None,
) -> PriorClaim:
    return PriorClaim(
        claim_id=uuid.uuid4(),
        policy_id=policy_id,
        incident_date=incident_date,
        claimed_amount=Decimal(amount),
        submitted_at=submitted_at or datetime.combine(incident_date + timedelta(days=2), datetime.min.time(), UTC),
    )


def test_clean_claim_triggers_no_rule() -> None:
    context = make_context()
    for registered in checks.RULES:
        assert registered.check(context).triggered is False, registered.rule_id


# ---- R01 amount_exceeds_sum_insured ---------------------------------------------------------------


def test_r01_triggers_when_amount_exceeds_sum_insured() -> None:
    outcome = checks.amount_exceeds_sum_insured(make_context(claimed_amount="650000.00"))
    assert outcome.triggered
    assert "650,000.00" in outcome.explanation and "150,000.00" in outcome.explanation


def test_r01_does_not_trigger_within_sum_insured() -> None:
    assert not checks.amount_exceeds_sum_insured(make_context(claimed_amount="499999.99")).triggered


def test_r01_boundary_amount_equal_to_sum_insured_is_allowed() -> None:
    assert not checks.amount_exceeds_sum_insured(make_context(claimed_amount="500000.00")).triggered


def test_r01_one_paisa_over_triggers() -> None:
    assert checks.amount_exceeds_sum_insured(make_context(claimed_amount="500000.01")).triggered


# ---- R02 policy_lapsed_at_incident_date -----------------------------------------------------------


def test_r02_triggers_when_policy_lapsed_before_incident() -> None:
    assert checks.policy_lapsed_at_incident_date(make_context(lapsed_on=date(2026, 5, 1))).triggered


def test_r02_does_not_trigger_when_policy_never_lapsed() -> None:
    outcome = checks.policy_lapsed_at_incident_date(make_context(lapsed_on=None))
    assert not outcome.triggered
    assert outcome.evidence["lapsed_on"] == ""


def test_r02_boundary_lapse_on_incident_day_triggers() -> None:
    assert checks.policy_lapsed_at_incident_date(make_context(lapsed_on=INCIDENT)).triggered


def test_r02_lapse_after_incident_does_not_trigger() -> None:
    assert not checks.policy_lapsed_at_incident_date(make_context(lapsed_on=INCIDENT + timedelta(days=1))).triggered


# ---- R03 claim_within_30d_of_policy_start ---------------------------------------------------------


def test_r03_triggers_12_days_after_start() -> None:
    outcome = checks.claim_within_30d_of_policy_start(make_context(incident_date=START + timedelta(days=12)))
    assert outcome.triggered
    assert outcome.evidence["days_after_start"] == "12"


def test_r03_does_not_trigger_well_after_start() -> None:
    assert not checks.claim_within_30d_of_policy_start(make_context()).triggered


@pytest.mark.parametrize(("days", "expected"), [(0, True), (30, True), (31, False)])
def test_r03_boundaries(days: int, expected: bool) -> None:
    context = make_context(incident_date=START + timedelta(days=days))
    assert checks.claim_within_30d_of_policy_start(context).triggered is expected


def test_r03_leaves_incidents_before_start_to_r06() -> None:
    context = make_context(incident_date=START - timedelta(days=5))
    assert not checks.claim_within_30d_of_policy_start(context).triggered
    assert checks.incident_date_before_policy_start(context).triggered


# ---- R04 more_than_3_claims_in_12_months ----------------------------------------------------------


def test_r04_triggers_on_fourth_claim_in_six_months() -> None:
    others = tuple(prior_claim(INCIDENT - timedelta(days=d)) for d in (40, 90, 140))
    outcome = checks.more_than_3_claims_in_12_months(make_context(other_claims=others))
    assert outcome.triggered
    assert outcome.evidence["claims_in_window"] == "4"


def test_r04_does_not_trigger_on_third_claim() -> None:
    others = tuple(prior_claim(INCIDENT - timedelta(days=d)) for d in (40, 90))
    assert not checks.more_than_3_claims_in_12_months(make_context(other_claims=others)).triggered


def test_r04_counts_claims_on_other_policies_of_the_same_customer() -> None:
    others = tuple(prior_claim(INCIDENT - timedelta(days=d), policy_id=OTHER_POLICY_ID) for d in (10, 20, 30))
    assert checks.more_than_3_claims_in_12_months(make_context(other_claims=others)).triggered


def test_r04_boundary_claim_exactly_365_days_earlier_is_inside_window() -> None:
    others = tuple(prior_claim(INCIDENT - timedelta(days=d)) for d in (365, 100, 50))
    assert checks.more_than_3_claims_in_12_months(make_context(other_claims=others)).triggered


def test_r04_claim_366_days_earlier_is_outside_window() -> None:
    others = tuple(prior_claim(INCIDENT - timedelta(days=d)) for d in (366, 100, 50))
    assert not checks.more_than_3_claims_in_12_months(make_context(other_claims=others)).triggered


def test_r04_later_incidents_do_not_count() -> None:
    others = (prior_claim(INCIDENT - timedelta(days=10)), *(prior_claim(INCIDENT + timedelta(days=d)) for d in (5, 9)))
    assert not checks.more_than_3_claims_in_12_months(make_context(other_claims=others)).triggered


# ---- R05 incident_date_in_future ------------------------------------------------------------------


def test_r05_triggers_when_incident_is_after_submission() -> None:
    context = make_context(incident_date=SUBMITTED_AT.date() + timedelta(days=5))
    assert checks.incident_date_in_future(context).triggered


def test_r05_does_not_trigger_for_past_incident() -> None:
    assert not checks.incident_date_in_future(make_context()).triggered


def test_r05_boundary_incident_on_submission_day_is_allowed() -> None:
    assert not checks.incident_date_in_future(make_context(incident_date=SUBMITTED_AT.date())).triggered


def test_r05_uses_business_date_not_utc_date() -> None:
    # 20:30 UTC on 14 June is already 15 June in India; an incident on 15 June is not in the future there.
    late_evening_utc = datetime(2026, 6, 14, 20, 30, tzinfo=UTC)
    context = make_context(
        incident_date=date(2026, 6, 15), submitted_at=late_evening_utc, submitted_on=date(2026, 6, 15)
    )
    assert not checks.incident_date_in_future(context).triggered


# ---- R06 incident_date_before_policy_start --------------------------------------------------------


def test_r06_triggers_when_incident_precedes_start() -> None:
    outcome = checks.incident_date_before_policy_start(make_context(incident_date=START - timedelta(days=15)))
    assert outcome.triggered
    assert "15 days before" in outcome.explanation


def test_r06_does_not_trigger_after_start() -> None:
    assert not checks.incident_date_before_policy_start(make_context()).triggered


def test_r06_boundary_incident_on_start_day_is_covered() -> None:
    assert not checks.incident_date_before_policy_start(make_context(incident_date=START)).triggered


# ---- R07 duplicate_claim_same_date_amount ---------------------------------------------------------


def test_r07_triggers_for_earlier_claim_with_same_date_and_amount() -> None:
    original = prior_claim(INCIDENT, amount="50000.00", submitted_at=SUBMITTED_AT - timedelta(days=3))
    outcome = checks.duplicate_claim_same_date_amount(make_context(other_claims=(original,)))
    assert outcome.triggered
    assert outcome.evidence["duplicate_claim_ids"] == str(original.claim_id)


def test_r07_does_not_trigger_when_amount_differs() -> None:
    other = prior_claim(INCIDENT, amount="50000.01", submitted_at=SUBMITTED_AT - timedelta(days=3))
    assert not checks.duplicate_claim_same_date_amount(make_context(other_claims=(other,))).triggered


def test_r07_original_is_not_a_duplicate_of_its_later_copy() -> None:
    later_copy = prior_claim(INCIDENT, amount="50000.00", submitted_at=SUBMITTED_AT + timedelta(days=3))
    assert not checks.duplicate_claim_same_date_amount(make_context(other_claims=(later_copy,))).triggered


def test_r07_same_date_and_amount_on_a_different_policy_is_not_a_duplicate() -> None:
    other = prior_claim(INCIDENT, "50000.00", policy_id=OTHER_POLICY_ID, submitted_at=SUBMITTED_AT - timedelta(1))
    assert not checks.duplicate_claim_same_date_amount(make_context(other_claims=(other,))).triggered


# ---- R08 kyc_incomplete ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [KycStatus.PENDING, KycStatus.REJECTED])
def test_r08_triggers_when_kyc_not_verified(status: KycStatus) -> None:
    assert checks.kyc_incomplete(make_context(kyc_status=status)).triggered


def test_r08_does_not_trigger_when_kyc_verified() -> None:
    assert not checks.kyc_incomplete(make_context(kyc_status=KycStatus.VERIFIED)).triggered


# ---- input validation -----------------------------------------------------------------------------


def test_context_rejects_the_claim_listed_among_its_own_other_claims() -> None:
    itself = PriorClaim(
        claim_id=CLAIM_ID, policy_id=POLICY_ID, incident_date=INCIDENT,
        claimed_amount=Decimal("50000.00"), submitted_at=SUBMITTED_AT,
    )  # fmt: skip
    with pytest.raises(ValidationError, match="must not include"):
        make_context(other_claims=(itself,))


def test_context_rejects_naive_timestamps() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        make_context(submitted_at=datetime(2026, 6, 5, 10, 0))  # noqa: DTZ001 — deliberately naive


def test_context_rejects_money_with_more_than_two_decimals() -> None:
    with pytest.raises(ValidationError):
        make_context(claimed_amount="100.005")

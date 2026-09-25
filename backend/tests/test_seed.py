"""Tests for the seed generator. Pure in-memory: no database needed.

The rule checks below are deliberately simple reference versions of R01-R08. Phase 3 builds the real
rules engine; these tests pin down what the seed data must look like to it.
"""

from collections import defaultdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from app.db.models import Claim, Customer, Policy
from app.domain.enums import ClaimStatus, KycStatus, PaymentStatus, PolicyStatus
from app.domain.products import INCIDENT_TYPES_BY_PRODUCT
from scripts.seed import DESCRIPTIONS, SeedData, build_dataset

NOW = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def data() -> SeedData:
    return build_dataset(now=NOW, seed=42)


def _triggered_rules(claim: Claim, data: SeedData) -> set[str]:
    policies = {p.id: p for p in data.policies}
    customers = {c.id: c for c in data.customers}
    policy: Policy = policies[claim.policy_id]
    customer: Customer = customers[policy.customer_id]
    customer_claims = [c for c in data.claims if policies[c.policy_id].customer_id == customer.id]

    rules = set()
    if claim.claimed_amount > policy.sum_insured:
        rules.add("R01")
    if policy.lapsed_on is not None and policy.lapsed_on <= claim.incident_date:
        rules.add("R02")
    if 0 <= (claim.incident_date - policy.start_date).days <= 30:
        rules.add("R03")
    window_start = claim.incident_date - timedelta(days=365)
    if sum(1 for c in customer_claims if window_start <= c.incident_date <= claim.incident_date) > 3:
        rules.add("R04")
    if claim.incident_date > claim.created_at.date():
        rules.add("R05")
    if claim.incident_date < policy.start_date:
        rules.add("R06")
    if any(
        other.id != claim.id
        and other.policy_id == claim.policy_id
        and other.incident_date == claim.incident_date
        and other.claimed_amount == claim.claimed_amount
        and other.created_at < claim.created_at
        for other in data.claims
    ):
        rules.add("R07")
    if customer.kyc_status is not KycStatus.VERIFIED:
        rules.add("R08")
    return rules


def test_row_counts(data: SeedData) -> None:
    assert (len(data.customers), len(data.policies), len(data.claims)) == (50, 80, 200)


def test_output_is_deterministic(data: SeedData) -> None:
    again = build_dataset(now=NOW, seed=42)
    assert [c.claim_number for c in again.claims] == [c.claim_number for c in data.claims]
    assert [c.claimed_amount for c in again.claims] == [c.claimed_amount for c in data.claims]
    assert [c.id for c in again.customers] == [c.id for c in data.customers]


def test_includes_the_five_required_edge_cases(data: SeedData) -> None:
    rules = {case.rule_id for case in data.edge_cases}
    assert {"R01", "R02", "R03", "R04", "R06"} <= rules  # over sum insured, lapsed, 12 days, 4 claims, before start
    assert len(data.edge_cases) == 8


def test_each_edge_case_triggers_exactly_its_rule(data: SeedData) -> None:
    by_number = {c.claim_number: c for c in data.claims}
    for case in data.edge_cases:
        assert _triggered_rules(by_number[case.claim_number], data) == {case.rule_id}, case


def test_unlabelled_claims_trigger_no_rules(data: SeedData) -> None:
    labelled = {case.claim_number for case in data.edge_cases}
    for claim in data.claims:
        if claim.claim_number not in labelled:
            assert _triggered_rules(claim, data) == set(), claim.claim_number


def test_queue_has_clean_submitted_claims_to_triage(data: SeedData) -> None:
    labelled = {case.claim_number for case in data.edge_cases}
    backlog = [c for c in data.claims if c.status is ClaimStatus.SUBMITTED and c.claim_number not in labelled]
    assert len(backlog) >= 15
    for claim in backlog:
        assert claim.final_outcome is None and claim.decided_by is None


def test_money_is_decimal_and_timestamps_are_aware_and_not_in_future(data: SeedData) -> None:
    for claim in data.claims:
        assert isinstance(claim.claimed_amount, Decimal)
        assert claim.created_at.tzinfo is not None and claim.created_at <= NOW
        if claim.decided_at is not None:
            assert claim.created_at <= claim.decided_at <= NOW
    for entry in data.audit_entries:
        assert entry.created_at.tzinfo is not None


def test_policy_and_payment_invariants_match_db_constraints(data: SeedData) -> None:
    payments = defaultdict(list)
    for payment in data.payments:
        payments[payment.policy_id].append(payment)
        assert (payment.status is PaymentStatus.MISSED) == (payment.paid_date is None)

    for policy in data.policies:
        assert policy.end_date > policy.start_date
        assert (policy.status is PolicyStatus.LAPSED) == (policy.lapsed_on is not None)
        policy_payments = payments[policy.id]
        assert sum(p.amount for p in policy_payments) <= policy.annual_premium
        if policy.status is PolicyStatus.LAPSED:
            assert policy_payments[-1].status is PaymentStatus.MISSED
            assert policy.lapsed_on is not None and policy.lapsed_on <= NOW.date()


@pytest.mark.parametrize("seed", [1, 7, 99, 2024])
def test_other_seeds_also_produce_clean_data(seed: int) -> None:
    other = build_dataset(now=NOW, seed=seed)
    labelled = {case.claim_number for case in other.edge_cases}
    assert len(other.claims) == 200
    for claim in other.claims:
        expected = {case.rule_id for case in other.edge_cases if case.claim_number == claim.claim_number}
        assert _triggered_rules(claim, other) == (expected if claim.claim_number in labelled else set())


def test_naive_now_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        build_dataset(now=datetime(2026, 9, 15, 12, 0))  # noqa: DTZ001 — deliberately naive


def test_every_product_incident_pair_has_its_own_descriptions() -> None:
    # Natural calamity belongs to motor and home; a shared list once gave a motor claim a house's "roof and walls".
    valid = {(product, incident) for product, incidents in INCIDENT_TYPES_BY_PRODUCT.items() for incident in incidents}
    assert set(DESCRIPTIONS) == valid


def test_claim_descriptions_fit_the_product(data: SeedData) -> None:
    products = {p.id: p.product_type for p in data.policies}
    for claim in data.claims:
        assert claim.description in DESCRIPTIONS[(products[claim.policy_id], claim.incident_type)]

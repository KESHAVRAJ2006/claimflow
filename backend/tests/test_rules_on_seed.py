"""Run the real rules engine over all 200 seeded claims.

test_seed.py checks the seed against a hand-written reference; this checks the production engine against the
seed's labels. Two independent implementations agreeing is strong evidence both are right.
"""

from collections import defaultdict
from datetime import UTC, datetime

import pytest

from app.db.models import Claim
from app.rules import evaluate_all
from app.services.rule_context import build_rule_context
from scripts.seed import SeedData, build_dataset

NOW = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def data() -> SeedData:
    return build_dataset(now=NOW, seed=42)


def _reports(data: SeedData) -> dict[str, tuple[list[str], int, list[str]]]:
    policies = {policy.id: policy for policy in data.policies}
    customers = {customer.id: customer for customer in data.customers}
    claims_by_customer: dict[object, list[Claim]] = defaultdict(list)
    for claim in data.claims:
        claims_by_customer[policies[claim.policy_id].customer_id].append(claim)

    reports = {}
    for claim in data.claims:
        policy = policies[claim.policy_id]
        customer_id = policy.customer_id
        context = build_rule_context(claim, policy, customers[customer_id], claims_by_customer[customer_id])
        report = evaluate_all(context)
        reports[claim.claim_number] = (report.triggered_rule_ids, report.risk_score, report.hard_blocks)
    return reports


def test_engine_matches_every_seed_label(data: SeedData) -> None:
    reports = _reports(data)
    labels = {case.claim_number: case.rule_id for case in data.edge_cases}
    for claim_number, (triggered, _, _) in reports.items():
        expected = [labels[claim_number]] if claim_number in labels else []
        assert triggered == expected, claim_number


def test_edge_cases_score_and_block_as_designed(data: SeedData) -> None:
    reports = _reports(data)
    by_rule = {case.rule_id: reports[case.claim_number] for case in data.edge_cases}
    for rule_id in ("R02", "R05", "R06"):
        assert by_rule[rule_id][2] == [rule_id] and by_rule[rule_id][1] == 0
    for rule_id in ("R01", "R07", "R08"):
        assert by_rule[rule_id][1] == 70
    assert by_rule["R03"][1] == 40 and by_rule["R04"][1] == 35

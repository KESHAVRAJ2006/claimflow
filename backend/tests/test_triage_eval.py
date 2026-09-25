"""Triage evaluation: the labelled set, scoring, retry accounting, quota detection and the report."""

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from pydantic import ValidationError

from app.agents.evaluation import (
    CaseResult,
    CitedSection,
    ClaimSelector,
    EvalCase,
    ExpectedCitation,
    FreeTier,
    Usage,
    append_result,
    count_wasted_retries,
    is_daily_quota,
    load_eval_set,
    load_results,
    provider_failure,
    render_report,
    score_case,
    summarise,
)
from app.agents.graph import AgentDependencies, build_claim_graph, run_claim
from app.agents.llm import LlmClient
from app.agents.llm_errors import AllProvidersFailedError, explain_llm_error
from app.agents.state import EvidencePassage, ToolCallRecord
from app.domain.enums import RecommendedOutcome
from scripts.eval_triage import CASES_PATH, UsageCounter, to_result
from tests.agent_fakes import Brain, ScriptedChatModel, loader_for, make_claim, rule_context_for, stub_tools

APPROVE, REJECT, ESCALATE = RecommendedOutcome.APPROVE, RecommendedOutcome.REJECT, RecommendedOutcome.ESCALATE
TIER = FreeTier(name="Test tier", tokens_per_minute=8_000, requests_per_day=1_000)


def _case(expected: RecommendedOutcome = REJECT, **fields: Any) -> EvalCase:
    base = {
        "id": "C1", "title": "Keys left in the car", "kind": "coverage",
        "base": {"product_type": "motor", "incident_type": "theft"},
        "description": "Car stolen; keys were in the ignition.", "expected_outcome": expected,
        "also_acceptable": ["escalate"], "expected_covered": False,
        "expected_citation": {"document": "Motor_Policy.pdf", "section": "4.4"},
        "why": "Theft with keys in the car is excluded (4.4).",
    }  # fmt: skip
    return EvalCase.model_validate(base | fields)


def _result(outcome: RecommendedOutcome | None, **fields: Any) -> CaseResult:
    base: dict[str, Any] = {
        "case_id": "C1", "claim_number": "CLM-2020-000015", "ran_at": datetime(2026, 9, 25, tzinfo=UTC),
        "status": "completed" if outcome else "failed", "outcome": outcome, "covered": False, "latency_s": 30.0,
        "citations": [CitedSection(document="Motor_Policy.pdf", page=3,
                                   section="4.4 Theft from an unlocked or unattended vehicle")],
        "usage": Usage(llm_calls=10, input_tokens=9_000, output_tokens=1_000),
    }  # fmt: skip
    return CaseResult.model_validate(base | fields)


def test_the_labelled_set_is_valid_and_fully_labelled() -> None:
    cases = load_eval_set(CASES_PATH).cases
    assert len(cases) >= 20
    for case in cases:
        assert case.kind == "rule" or (case.expected_covered is not None and case.expected_citation is not None)
        assert case.acceptable >= {case.expected_outcome}
    assert {case.id for case in cases if case.kind == "rule"} == {f"R0{n}" for n in range(1, 9)}


def test_a_selector_names_an_edge_case_or_a_product_and_incident() -> None:
    assert ClaimSelector(edge_case=3).edge_case == 3
    with pytest.raises(ValidationError):
        ClaimSelector()
    with pytest.raises(ValidationError):
        ClaimSelector(edge_case=1, product_type="motor", incident_type="theft")


@pytest.mark.parametrize(
    ("document", "section", "hit"),
    [
        ("Motor_Policy.pdf", "4.4 Theft from an unlocked or unattended vehicle", True),
        ("Motor_Policy.pdf", "4.3 Flood and engine damage", False),
        ("Motor_Policy.pdf", "SECTION 4 - WHAT IS NOT COVERED (EXCLUSIONS)", False),
        ("Home_Policy.pdf", "4.4 Underinsurance", False),
        ("Motor_Policy.pdf", None, False),
    ],
)
def test_expected_citation_matches_the_clause_number(document: str, section: str | None, hit: bool) -> None:
    assert ExpectedCitation(document="Motor_Policy.pdf", section="4.4").matches(document, section) is hit


def test_approving_what_should_be_rejected_is_unsafe() -> None:
    score = score_case(_case(REJECT), _result(APPROVE, covered=True))
    assert score.unsafe_approval and not score.acceptable and score.covered_correct is False


def test_escalating_instead_of_rejecting_is_acceptable_but_not_exact() -> None:
    score = score_case(_case(REJECT), _result(ESCALATE))
    assert score.acceptable and not score.exact and not score.unsafe_approval
    assert score.unnecessary_escalation and score.citation_hit is True


def test_rejecting_what_should_be_approved_is_a_wrongful_rejection() -> None:
    score = score_case(_case(APPROVE, expected_covered=True), _result(REJECT))
    assert score.wrongful_rejection and not score.acceptable and score.covered_correct is False


def test_a_run_without_an_outcome_is_never_acceptable() -> None:
    score = score_case(_case(), _result(None))
    assert not score.acceptable and not score.unsafe_approval


def _call(call_id: str, round_number: int, tool: str, **args: Any) -> ToolCallRecord:
    return ToolCallRecord(call_id=call_id, round=round_number, step=1, tool=tool, args=args, status="ok",
                          result_summary="", latency_ms=1, started_at=datetime.now(UTC))  # fmt: skip


def _passage(evidence_id: str, found_by: str) -> EvidencePassage:
    return EvidencePassage(
        evidence_id=evidence_id, chunk_id=evidence_id, document="Motor_Policy.pdf", page=3, section="4.4",
        text="...", similarity=0.6, retrieval_confidence=0.9, found_by=found_by,
    )  # fmt: skip


def test_a_retry_that_repeats_old_calls_and_finds_nothing_is_wasted() -> None:
    calls = [_call("T1", 1, "check_exclusions", query="theft"), _call("T2", 2, "check_exclusions", query="theft"),
             _call("T3", 3, "get_waiting_period", query="theft")]  # fmt: skip
    evidence = [_passage("E1", "T1")]
    assert count_wasted_retries(calls, evidence, rounds=3) == 1  # round 2 repeated T1; round 3 made a new call
    assert count_wasted_retries(calls[:1], evidence, rounds=1) == 0
    assert count_wasted_retries(calls[:1], evidence, rounds=2) == 1  # a round with no calls at all


def _explained(*errors: tuple[str, Exception]) -> str:
    """The text a failed model call leaves in a run, exactly as invoke_structured writes it."""
    error = AllProvidersFailedError(errors) if len(errors) > 1 else errors[0][1]
    return f"LLM call failed: {explain_llm_error(error, single_provider=len(errors) == 1)}"


class RateLimitError(Exception):
    """Stands in for a provider SDK's rate-limit error (classified by class name)."""


DAILY = RateLimitError("Rate limit reached on tokens per day (TPD): Limit 200000")
PER_MINUTE = RateLimitError("Rate limit reached on tokens per minute. Please try again in 12s.")


def test_quota_detection_matches_the_real_error_messages() -> None:
    both_daily = _explained(("Groq", DAILY), ("Gemini", DAILY))
    one_recovers = _explained(("Groq", DAILY), ("Gemini", PER_MINUTE))
    only_provider = _explained(("Groq", DAILY))
    assert all(provider_failure([text]) == text for text in (both_daily, one_recovers, only_provider))
    assert is_daily_quota(both_daily) and is_daily_quota(only_provider)
    assert not is_daily_quota(one_recovers)  # Gemini is back within a minute: keep going
    assert provider_failure(["Risk score 70 is at or above 70 (R08)."]) is None


def test_results_append_and_the_latest_run_of_a_case_wins(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    append_result(path, _result(ESCALATE))
    append_result(path, _result(REJECT))
    assert load_results(path)["C1"].outcome is REJECT
    assert load_results(tmp_path / "missing.jsonl") == {}


def test_provider_limited_runs_are_left_out_of_the_metrics() -> None:
    cases = [_case(), _case(APPROVE, id="C2", expected_covered=True)]
    results = {"C1": _result(REJECT), "C2": _result(None, case_id="C2", status="provider_limited")}
    summary, scores = summarise(cases, results)
    assert (summary.scored, summary.missing, summary.provider_limited) == (1, ("C2",), ("C2",))
    assert [score.case.id for score in scores] == ["C1"]


def test_report_shows_the_safety_metric_first_and_every_case() -> None:
    cases = [_case(), _case(APPROVE, id="C2", title="Collision", expected_covered=True)]
    questioned = _result(APPROVE, case_id="C2", covered=True, open_questions=("Was the car locked?",))
    results = {"C1": _result(APPROVE, covered=True), "C2": questioned}
    summary, scores = summarise(cases, results)
    report = render_report(summary, scores, models="test-model", free_tier=TIER)
    assert summary.unsafe_approvals == ("C1",) and summary.claims_with_open_questions == 1
    metric_rows = [line for line in report.splitlines() if line.startswith("| ") and "Metric" not in line]
    assert metric_rows[0].startswith("| Unsafe approvals") and "**1** (C1)" in metric_rows[0]
    assert "| C1: Keys left in the car | reject | approve | UNSAFE |" in report
    assert "C2: Was the car locked?" in report and "0.8 claims a minute" in report  # 8,000 TPM / 10,000 tokens


async def test_usage_counter_adds_up_every_model_call() -> None:
    counter = UsageCounter()
    for tokens_in, tokens_out in ((1_000, 200), (500, 50)):
        message = AIMessage("ok", usage_metadata={"input_tokens": tokens_in, "output_tokens": tokens_out,
                                                  "total_tokens": tokens_in + tokens_out},
                            response_metadata={"model_name": "gpt-oss-120b"})  # fmt: skip
        await counter.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))
    assert counter.usage() == Usage(llm_calls=2, input_tokens=1_500, output_tokens=250,
                                    tokens_by_model={"gpt-oss-120b": 1_750})  # fmt: skip


async def test_a_scripted_run_becomes_a_scored_result() -> None:
    claim = make_claim("CLM-2026-000301")
    plan = [[("get_policy_status", {"policy_number": claim.policy_number, "on_date": claim.incident_date.isoformat()})],
            [("get_coverage_section", {"product_type": "motor", "incident_type": "collision"})]]  # fmt: skip
    deps = AgentDependencies(
        llm=LlmClient([ScriptedChatModel(brain=Brain(plans={claim.claim_number: plan}))]),
        tools=stub_tools(), load_rule_context=loader_for({claim.claim_number: rule_context_for(claim)}),
    )  # fmt: skip
    state = await run_claim(build_claim_graph(deps), claim, "t-eval")
    result = to_result(_case(APPROVE, expected_covered=True), claim.claim_number, state, Usage(), latency_s=1.23)
    assert (result.status, result.outcome, result.covered) == ("completed", APPROVE, True)
    assert (result.investigation_rounds, result.retries, result.wasted_retries, result.tool_calls) == (1, 0, 0, 2)
    assert result.citations and result.confidence == Decimal("0.900") and result.latency_s == 1.2

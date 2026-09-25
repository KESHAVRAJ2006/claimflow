"""Reflection: the deterministic citation check, the weakest-link confidence, and the retry/escalate decision."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from app.agents.llm import LlmClient
from app.agents.reflection import (
    build_messages,
    check_citations,
    cited_retrieval_confidence,
    quote_supported,
    run_reflection,
    same_request,
)
from app.agents.state import Critique, DecisionDraft, EvidencePassage, EvidenceRequest, ReflectionResult, ToolCallRecord
from app.rules import evaluate_all
from tests.agent_fakes import COVERAGE_TEXT, Brain, ScriptedChatModel, make_claim, rule_context_for

TOOLS = {
    "check_exclusions": "Find exclusions that may apply to an incident.",
    "get_waiting_period": "Find the waiting period for a condition or event.",
    "get_claim_history": "List the policy's earlier claims.",
}

EVIDENCE = [
    EvidencePassage(evidence_id="E1", chunk_id="c1", document="Motor_Policy.pdf", page=2, section="Coverage",
                    text=COVERAGE_TEXT, similarity=0.6, retrieval_confidence=0.9, found_by="T2"),
    EvidencePassage(evidence_id="E2", chunk_id="c2", document="Motor_Policy.pdf", page=5, section="Claims",
                    text="Notify the insurer within 7 days.", similarity=0.3, retrieval_confidence=0.286,
                    found_by="T3"),
]  # fmt: skip
CALLS = [
    ToolCallRecord(call_id="T1", round=1, step=1, tool="get_policy_status", args={}, status="ok",
                   result_summary="in force", latency_ms=3, started_at=datetime.now(UTC)),
]  # fmt: skip
GOOD_QUOTE = "caused by accidental external means, including collision"


def _draft(*points: dict[str, Any], covered: bool = True) -> DecisionDraft:
    return DecisionDraft.model_validate(
        {"covered": covered, "confidence": 0.9, "summary": "Covered collision damage.", "rationale": list(points)}
    )


WORDING = {"statement": "Collision is covered.", "basis": "policy_wording", "evidence_ids": ["E1"], "quote": GOOD_QUOTE}
DATA = {"statement": "Cover was in force.", "basis": "claim_data", "tool_call_ids": ["T1"]}


@pytest.mark.parametrize(
    ("quote", "expected"),
    [
        (GOOD_QUOTE, True),
        ("CAUSED  BY accidental\nexternal means", True),  # case and whitespace don't matter
        ("“the insurer will indemnify the insured”", True),  # curly quotes stripped
        ("The insurer will indemnify ... including collision and overturning", True),  # each fragment present
        ("including collision", False),  # too short to prove anything
        ("the insurer will pay for any collision damage", False),  # paraphrase, not a quote
        ("The insurer will indemnify ... excluding collision", False),  # one fragment invented
    ],
)
def test_quote_supported(quote: str, expected: bool) -> None:
    assert quote_supported(quote, COVERAGE_TEXT) is expected


def test_clean_decision_has_no_citation_problems() -> None:
    assert check_citations(_draft(WORDING, DATA), EVIDENCE, CALLS) == ()


@pytest.mark.parametrize(
    ("point", "problem"),
    [
        (WORDING | {"evidence_ids": ["E9"]}, "never retrieved: E9"),
        (WORDING | {"evidence_ids": []}, "cites no evidence id"),
        (WORDING | {"quote": None}, "no supporting quote"),
        (WORDING | {"quote": "the insurer pays for all collision damage"}, "does not appear in the cited passage"),
        (WORDING | {"evidence_ids": ["E2"]}, "does not appear"),  # real quote, wrong passage
        (DATA | {"tool_call_ids": []}, "cites no tool call"),
        (DATA | {"tool_call_ids": ["T7"]}, "do not exist: T7"),
    ],
)
def test_citation_problems_are_caught_by_code(point: dict[str, Any], problem: str) -> None:
    problems = check_citations(_draft(point, WORDING if point["basis"] == "claim_data" else DATA), EVIDENCE, CALLS)
    assert any(problem in item for item in problems), problems


def test_claimant_statements_need_no_citation() -> None:
    form = {"statement": "The claimant says the bumper was damaged.", "basis": "claim_form"}
    assert check_citations(_draft(WORDING, form), EVIDENCE, CALLS) == ()


def test_covered_without_any_wording_is_a_problem() -> None:
    assert any("without citing any policy wording" in p for p in check_citations(_draft(DATA), EVIDENCE, CALLS))
    assert check_citations(_draft(DATA, covered=False), EVIDENCE, CALLS) == ()


def test_retrieval_confidence_is_the_weakest_cited_passage() -> None:
    both = WORDING | {"evidence_ids": ["E1", "E2"]}
    assert cited_retrieval_confidence(_draft(WORDING), EVIDENCE) == Decimal("0.900")
    assert cited_retrieval_confidence(_draft(both), EVIDENCE) == Decimal("0.286")
    assert cited_retrieval_confidence(_draft(DATA, covered=False), EVIDENCE) == Decimal("0.000")
    assert cited_retrieval_confidence(None, EVIDENCE) == Decimal("0.000")


async def _reflect(
    brain: Brain, draft: DecisionDraft | None, retries_used: int = 0, previous: tuple[ReflectionResult, ...] = ()
) -> Any:
    claim = make_claim()
    return await run_reflection(
        LlmClient([ScriptedChatModel(brain=brain)]), reflection_round=retries_used + 1, retries_used=retries_used,
        claim=claim, draft=draft, rounds=[], tool_calls=CALLS, evidence=EVIDENCE,
        risk_report=evaluate_all(rule_context_for(claim)), tools=TOOLS, previous=previous,
    )  # fmt: skip


def _critique(*lookups: tuple[str, str], grounded: bool = False, **fields: Any) -> dict[str, Any]:
    requests = [{"tool": tool, "lookup": lookup} for tool, lookup in lookups]
    return {"grounded": grounded, "verdict": "investigate_more", "missing_evidence": requests} | fields


async def test_citation_failure_sends_back_without_calling_the_critic() -> None:
    brain = Brain()
    result = await _reflect(brain, _draft(WORDING | {"evidence_ids": ["E9"]}, DATA))
    assert result.action == "retry_investigation" and result.critique is None and brain.calls == []
    assert any("E9" in item for item in result.feedback)


async def test_critic_accepts_a_grounded_decision() -> None:
    result = await _reflect(Brain(), _draft(WORDING, DATA))
    assert result.action == "accept" and result.critique is not None and result.feedback == ()


async def test_critic_requests_specific_evidence() -> None:
    critique = _critique(("check_exclusions", "exclusions for racing"))
    result = await _reflect(Brain(critiques=[critique]), _draft(WORDING, DATA))
    assert result.action == "retry_investigation" and result.open_questions == ()
    assert "Look up with check_exclusions: exclusions for racing" in result.feedback


async def test_retry_limit_escalates_instead_of_looping() -> None:
    critique = _critique(("check_exclusions", "anything"))
    result = await _reflect(Brain(critiques=[critique]), _draft(WORDING, DATA), retries_used=2)
    assert result.action == "escalate"


# The lookups that cost early runs both retries: facts only the claimant knows.
async def test_claimant_only_facts_go_to_the_reviewer_instead_of_a_retry() -> None:
    brain = Brain(critiques=[_critique(("none", "Whether the vehicle was locked at the time of theft"), grounded=True)])
    result = await _reflect(brain, _draft(WORDING, DATA))
    assert result.action == "accept" and result.feedback == ()
    assert result.open_questions == ("Whether the vehicle was locked at the time of theft",)


async def test_a_lookup_naming_no_real_tool_is_an_open_question() -> None:
    critique = _critique(("call_the_claimant", "keys left inside the vehicle"), grounded=True)
    result = await _reflect(Brain(critiques=[critique]), _draft(WORDING, DATA))
    assert result.action == "accept" and result.open_questions == ("keys left inside the vehicle",)


async def test_ungrounded_decision_missing_only_claimant_facts_escalates_without_a_retry() -> None:
    critique = _critique(("none", "whether the engine was running"))
    result = await _reflect(Brain(critiques=[critique]), _draft(WORDING, DATA))
    assert result.action == "escalate" and result.open_questions == ("whether the engine was running",)
    assert result.feedback == ("More investigation cannot answer: whether the engine was running",)


async def test_unsupported_statements_still_send_the_claim_back() -> None:
    critique = _critique(("none", "whether the car was locked"), unsupported_statements=["The car was locked."])
    result = await _reflect(Brain(critiques=[critique]), _draft(WORDING, DATA))
    assert result.action == "retry_investigation" and result.feedback == ("Unsupported: The car was locked.",)
    assert result.open_questions == ("whether the car was locked",)


def _sent_back(tool: str, lookup: str) -> ReflectionResult:
    critique = Critique.model_validate(_critique((tool, lookup)))
    return ReflectionResult(round=1, citation_problems=(), critique=critique, action="retry_investigation",
                            feedback=(f"Look up with {tool}: {lookup}",))  # fmt: skip


async def test_a_lookup_already_sent_back_is_not_sent_back_again() -> None:
    # Reworded from the first request, as the critic did in a real run of CLM-2026-900003.
    earlier = _sent_back("get_claim_history", "claim form details showing that the loss pertains to roof and walls")
    again = _critique(("get_claim_history", "claim form details confirming that the loss pertains to roof and walls"))
    result = await _reflect(Brain(critiques=[again]), _draft(WORDING, DATA), retries_used=1, previous=(earlier,))
    assert result.action == "escalate" and len(result.open_questions) == 1


@pytest.mark.parametrize(
    ("first", "second", "same"),
    [
        (("check_exclusions", "exclusions for racing"), ("check_exclusions", "Exclusions for  RACING"), True),
        (("check_exclusions", "exclusions for racing"), ("get_waiting_period", "exclusions for racing"), False),
        (("check_exclusions", "exclusions for racing"), ("check_exclusions", "theft from an unlocked car"), False),
        (
            ("get_waiting_period", "waiting period for cataract"),
            ("get_waiting_period", "cataract waiting period"),
            True,
        ),
    ],
)
def test_same_request_allows_rewording_but_not_a_new_question(
    first: tuple[str, str], second: tuple[str, str], same: bool
) -> None:
    make = lambda pair: EvidenceRequest(tool=pair[0], lookup=pair[1])  # noqa: E731
    assert same_request(make(first), make(second)) is same


def test_the_critic_sees_which_tools_exist() -> None:
    claim = make_claim()
    report = evaluate_all(rule_context_for(claim))
    prompt = str(build_messages(claim, _draft(WORDING, DATA), [], CALLS, EVIDENCE, report, TOOLS)[1].content)
    assert all(f"- {name}: {description}" in prompt for name, description in TOOLS.items())


async def test_no_decision_or_no_critic_escalates() -> None:
    assert (await _reflect(Brain(), None)).action == "escalate"
    assert (await _reflect(Brain(raise_on={"Critique"}), _draft(WORDING, DATA))).action == "escalate"

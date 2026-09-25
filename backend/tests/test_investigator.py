"""Investigator ReAct loop: tool logging, evidence ids, error handling, the step limit and graceful stop."""

from typing import Any

import pytest

from app.agents.investigator import SUBMIT_FINDINGS, Investigator
from app.agents.llm import LlmClient
from app.db.readonly import ReadOnlyViolationError
from tests.agent_fakes import Brain, ScriptedChatModel, make_claim, stub_tools

POLICY = {"policy_number": "MOT-2025-000011"}
CLAIM = make_claim("CLM-2026-000101")


def _investigator(brain: Brain, tools: list[Any] | None = None, max_steps: int = 6) -> Investigator:
    return Investigator(LlmClient([ScriptedChatModel(brain=brain)]), tools or stub_tools(), max_steps=max_steps)


async def test_every_tool_call_is_logged_and_streamed() -> None:
    brain = Brain(plans={CLAIM.claim_number: [
        [("get_policy_status", POLICY)],
        [("get_coverage_section", {"product_type": "motor", "incident_type": "collision"}),
         ("check_similar_claims", {"claim_number": CLAIM.claim_number})],  # two calls in one step
    ]})  # fmt: skip
    events: list[dict[str, Any]] = []
    output = await _investigator(brain).run(CLAIM, None, emit=events.append)

    records = output.tool_calls
    assert [r.tool for r in records] == ["get_policy_status", "get_coverage_section", "check_similar_claims"]
    assert [r.call_id for r in records] == ["T1", "T2", "T3"] and [r.step for r in records] == [1, 2, 2]
    assert all(r.status == "ok" and r.latency_ms >= 0 and r.result_summary for r in records)
    assert records[0].args == POLICY and "in force" in records[0].result_summary
    assert records[1].result_data == {"evidence_ids": ["E1"], "retrieval_confidence": pytest.approx(1.0)}
    assert [e["type"] for e in events] == ["tool_call_started", "tool_call_finished"] * 3
    assert output.round.stopped_early is False and output.round.steps_used == 3
    assert [p.evidence_id for p in output.new_evidence] == ["E1"]


async def test_later_rounds_continue_ids_and_reuse_evidence() -> None:
    step = [("get_coverage_section", {"product_type": "motor", "incident_type": "collision"})]
    brain = Brain(plans={CLAIM.claim_number: [step]})
    first = await _investigator(brain).run(CLAIM, None)
    second = await _investigator(brain).run(
        CLAIM, None, round_number=2, prior_calls=len(first.tool_calls), evidence=first.new_evidence,
        previous=[first.round], feedback=["Look up: exclusions for collision"],
    )  # fmt: skip
    assert second.tool_calls[0].call_id == "T2" and second.tool_calls[0].round == 2
    assert second.new_evidence == ()  # same passage: E1 is reused, not duplicated
    assert second.tool_calls[0].result_data["evidence_ids"] == ["E1"]  # type: ignore[index]
    round_two_brief = brain.calls[-1][1][1].content
    assert "investigation round 2" in round_two_brief and "Look up: exclusions" in round_two_brief


async def test_unknown_and_failing_tools_become_error_records_not_crashes() -> None:
    brain = Brain(plans={CLAIM.claim_number: [[("delete_claim", {}), ("get_policy_status", POLICY)]]})
    output = await _investigator(brain, stub_tools(failing={"get_policy_status"})).run(CLAIM, None)
    unknown, failed = output.tool_calls
    assert unknown.status == "error" and "Unknown tool" in unknown.result_summary
    assert failed.status == "error" and "failed" in failed.result_summary
    assert output.round.stopped_early is False  # the model still submitted findings


async def test_a_read_only_violation_stops_the_run() -> None:
    tools = stub_tools()

    async def violate(*_args: Any, **_kwargs: Any) -> Any:
        raise ReadOnlyViolationError("INSERT on claims")

    policy_tool = next(t for t in tools if t.name == "get_policy_status")
    object.__setattr__(policy_tool, "coroutine", violate)
    brain = Brain(plans={CLAIM.claim_number: [[("get_policy_status", POLICY)]]})
    with pytest.raises(ReadOnlyViolationError):
        await _investigator(brain, tools).run(CLAIM, None)


async def test_submit_with_other_calls_in_the_same_step_is_rejected() -> None:
    brain = Brain(
        plans={CLAIM.claim_number: [[("get_policy_status", POLICY), (SUBMIT_FINDINGS, {"summary": "x" * 20})]]}
    )
    output = await _investigator(brain).run(CLAIM, None)
    # Step 1's submit is refused; the brain submits alone in step 2.
    assert output.round.steps_used == 2 and output.round.findings.summary.startswith("Investigated")
    step_one_replies = [m for m in brain.calls[1][1] if getattr(m, "name", None) == SUBMIT_FINDINGS]
    assert "must be called on its own" in step_one_replies[0].content


async def test_step_limit_forces_partial_findings_and_marks_early_stop() -> None:
    brain = Brain(never_submit=True)
    output = await _investigator(brain, max_steps=3).run(CLAIM, None)
    assert output.round.stopped_early and "3-step limit" in (output.round.stop_reason or "")
    assert output.round.steps_used == 3 and len(output.tool_calls) == 3
    assert output.round.findings.summary == "Partial findings at the limit."
    assert brain.calls[-1][0] == "forced_submit" and brain.calls[-1][2] == SUBMIT_FINDINGS


async def test_when_even_forced_findings_fail_code_builds_them_from_the_log() -> None:
    brain = Brain(never_submit=True, fail_forced_submit=True)
    output = await _investigator(brain, max_steps=2).run(CLAIM, None)
    findings = output.round.findings
    assert output.round.stopped_early and findings.key_facts[0].startswith("T1 get_policy_status")
    assert "Investigation incomplete" in findings.evidence_gaps[0]


async def test_llm_outage_hands_off_without_a_forced_call() -> None:
    brain = Brain(raise_on={"investigator"})
    output = await _investigator(brain).run(CLAIM, None)
    assert output.round.stopped_early and "unavailable" in (output.round.stop_reason or "")
    assert all(role != "forced_submit" for role, _, _ in brain.calls)


async def test_different_claims_take_different_paths() -> None:
    simple, complex_ = make_claim("CLM-2026-000201"), make_claim("CLM-2026-000202", amount="98000.00")
    brain = Brain(plans={
        simple.claim_number: [[("get_policy_status", POLICY)]],
        complex_.claim_number: [
            [("get_policy_status", POLICY), ("get_claim_history", POLICY)],
            [("check_exclusions", {"product_type": "motor", "incident_description": "collision at night"})],
            [("check_similar_claims", {"claim_number": complex_.claim_number})],
        ],
    })  # fmt: skip
    a = await _investigator(brain).run(simple, None)
    b = await _investigator(brain).run(complex_, None)
    assert [r.tool for r in a.tool_calls] != [r.tool for r in b.tool_calls]


def test_duplicate_tool_names_are_rejected() -> None:
    tools = stub_tools()
    with pytest.raises(ValueError, match="unique"):
        Investigator(LlmClient([ScriptedChatModel(brain=Brain())]), [*tools, tools[0]])

"""End-to-end graph runs with a scripted model: routing, retries, loop limits, streaming and checkpointing."""

import logging
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from app.agents.graph import AgentDependencies, build_claim_graph, finalise, open_checkpointer, run_claim, stream_claim
from app.agents.llm import LlmClient
from app.domain.enums import RecommendedOutcome
from tests.agent_fakes import Brain, ScriptedChatModel, loader_for, make_claim, rule_context_for, stub_tools

CLAIM = make_claim("CLM-2026-000301")
POLICY = {"policy_number": CLAIM.policy_number}
COVERAGE = ("get_coverage_section", {"product_type": "motor", "incident_type": "collision"})
PLAN = [[("get_policy_status", POLICY | {"on_date": CLAIM.incident_date.isoformat()})], [COVERAGE]]


def _investigate(tool: str, lookup: str, *, grounded: bool = False) -> dict[str, Any]:
    return {"grounded": grounded, "verdict": "investigate_more", "missing_evidence": [{"tool": tool, "lookup": lookup}]}


INVESTIGATE_MORE = _investigate("check_exclusions", "exclusions")


def _graph(brain: Brain, *, context: Any = None, checkpointer: Any = None, policy_score: float = 0.6, steps: int = 6):  # type: ignore[no-untyped-def]
    deps = AgentDependencies(
        llm=LlmClient([ScriptedChatModel(brain=brain)]),
        tools=stub_tools(policy_score=policy_score),
        load_rule_context=loader_for({CLAIM.claim_number: context or rule_context_for(CLAIM)}),
        max_investigation_steps=steps,
    )
    return build_claim_graph(deps, checkpointer)


def _nodes(state: Any) -> list[tuple[str, int]]:
    return [(run.agent_name, run.round) for run in state["node_runs"]]


async def test_clean_claim_is_recommended_for_approval() -> None:
    state = await run_claim(_graph(Brain(plans={CLAIM.claim_number: PLAN})), CLAIM, "t-approve")
    assert _nodes(state) == [
        (n, 1) for n in ("intake", "investigator", "rules", "decision", "reflection", "route_final")
    ]
    final = state["final"]
    assert final.outcome is RecommendedOutcome.APPROVE and final.requires_human_review is True
    assert [c.label for c in final.citations] == ["Motor_Policy.pdf p.2"]
    assert (final.agent_confidence, final.retrieval_confidence) == (Decimal("0.900"), Decimal("1.000"))
    assert final.confidence == Decimal("0.900")  # routing uses the lower of the two
    assert [r.tool for r in state["tool_call_log"]] == ["get_policy_status", "get_coverage_section"]
    assert "AI-assisted recommendation" in final.disclaimer


async def test_stream_emits_node_events_with_tool_calls_nested_inside_the_investigator() -> None:
    events = [e async for e in stream_claim(_graph(Brain(plans={CLAIM.claim_number: PLAN})), CLAIM, "t-stream")]
    kinds = [(e["type"], e.get("node") or e.get("tool")) for e in events]
    start, end = kinds.index(("node_started", "investigator")), kinds.index(("node_finished", "investigator"))
    assert kinds[start + 1 : end] == [
        ("tool_call_started", "get_policy_status"), ("tool_call_finished", "get_policy_status"),
        ("tool_call_started", "get_coverage_section"), ("tool_call_finished", "get_coverage_section"),
    ]  # fmt: skip
    assert kinds[-1] == ("node_finished", "route_final")
    finished = next(e for e in events if e["type"] == "tool_call_finished")
    assert {"tool", "args", "result_summary", "latency_ms", "call_id", "round"} <= finished.keys()


async def test_reflection_retry_runs_a_distinct_second_round() -> None:
    brain = Brain(plans={CLAIM.claim_number: PLAN}, critiques=[INVESTIGATE_MORE])  # second review accepts
    state = await run_claim(_graph(brain), CLAIM, "t-retry")
    rounds = [r.round for r in state["tool_call_log"]]
    assert rounds == [1, 1, 2, 2] and [r.call_id for r in state["tool_call_log"]] == ["T1", "T2", "T3", "T4"]
    assert ("investigator", 2) in _nodes(state)
    assert state["investigations"][1].brief == ("Look up with check_exclusions: exclusions",)
    assert state["final"].outcome is RecommendedOutcome.APPROVE


async def test_reflection_limit_forces_escalation() -> None:
    critiques = [
        _investigate("check_exclusions", "exclusions for racing"),
        _investigate("get_waiting_period", "waiting period for engine damage"),
        _investigate("get_claim_history", "earlier claims for the same vehicle"),
    ]
    state = await run_claim(_graph(Brain(plans={CLAIM.claim_number: PLAN}, critiques=critiques)), CLAIM, "t-limit")
    assert [i.round for i in state["investigations"]] == [1, 2, 3]  # the first try plus 2 retries
    assert state["final"].outcome is RecommendedOutcome.ESCALATE
    assert any("Reflection could not confirm" in reason for reason in state["final"].reasons)


async def test_questions_only_the_claimant_can_answer_reach_the_reviewer_without_a_retry() -> None:
    critique = _investigate("none", "Whether the keys were left in the vehicle", grounded=True)
    state = await run_claim(_graph(Brain(plans={CLAIM.claim_number: PLAN}, critiques=[critique])), CLAIM, "t-open")
    assert [i.round for i in state["investigations"]] == [1]
    assert state["final"].outcome is RecommendedOutcome.APPROVE
    assert state["final"].open_questions == ("Whether the keys were left in the vehicle",)


async def test_the_same_lookup_is_not_sent_back_twice() -> None:
    critiques = [INVESTIGATE_MORE, INVESTIGATE_MORE]
    state = await run_claim(_graph(Brain(plans={CLAIM.claim_number: PLAN}, critiques=critiques)), CLAIM, "t-repeat")
    assert [i.round for i in state["investigations"]] == [1, 2]  # one retry, then a person takes over
    assert state["final"].outcome is RecommendedOutcome.ESCALATE


async def test_investigator_step_limit_forces_escalation() -> None:
    state = await run_claim(_graph(Brain(never_submit=True), steps=2), CLAIM, "t-steps")
    assert state["investigations"][0].stopped_early
    assert state["final"].outcome is RecommendedOutcome.ESCALATE
    assert any("stopped early" in flag for flag in state["final"].escalation_flags)


async def test_agent_cannot_override_a_hard_block() -> None:
    lapsed = rule_context_for(CLAIM, lapsed_on=CLAIM.incident_date - timedelta(days=10))
    state = await run_claim(_graph(Brain(plans={CLAIM.claim_number: PLAN}), context=lapsed), CLAIM, "t-block")
    final = state["final"]
    assert state["decision"].covered is True and state["decision"].confidence == 0.9  # the agent says "covered"
    assert final.outcome is RecommendedOutcome.REJECT and final.hard_blocks == ("R02",)


async def test_weak_retrieval_escalates_even_when_the_agent_is_confident() -> None:
    state = await run_claim(_graph(Brain(plans={CLAIM.claim_number: PLAN}), policy_score=0.35), CLAIM, "t-weak")
    final = state["final"]
    assert final.agent_confidence == Decimal("0.900") and final.retrieval_confidence < final.agent_confidence
    assert final.outcome is RecommendedOutcome.ESCALATE and "below the 0.65 minimum" in final.reasons[0]


async def test_high_amount_escalates() -> None:
    big = make_claim(CLAIM.claim_number, amount="150000.00", claim_id=CLAIM.claim_id)
    state = await run_claim(_graph(Brain(plans={CLAIM.claim_number: PLAN})), big, "t-big")
    assert state["final"].outcome is RecommendedOutcome.ESCALATE


async def test_intake_failure_ends_the_run_as_failed() -> None:
    bad = lambda text: {"policy_number": "nonsense"}  # noqa: E731
    state = await run_claim(_graph(Brain(intake=[bad, bad])), CLAIM, "t-fail")
    assert state["status"] == "failed" and "intake failed after 2 attempts" in state["failure_reason"]
    assert _nodes(state) == [("intake", 1)] and state.get("final") is None


async def test_decision_failure_escalates() -> None:
    brain = Brain(plans={CLAIM.claim_number: PLAN}, raise_on={"DecisionDraft"})
    state = await run_claim(_graph(brain), CLAIM, "t-decision")
    assert state["decision"] is None and state["final"].outcome is RecommendedOutcome.ESCALATE
    assert "No decision was produced" in state["final"].summary


def test_finalise_is_deterministic() -> None:
    state: Any = {"claim": CLAIM, "risk_report": _report(), "decision": None, "evidence": [], "escalation_flags": ["x"]}
    assert finalise(state) == finalise(state)


def _report() -> Any:
    from app.rules import evaluate_all

    return evaluate_all(rule_context_for(CLAIM, policy_start=date(2025, 1, 1)))


async def test_sqlite_checkpoint_round_trips_state_without_unregistered_types(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    async with open_checkpointer(tmp_path / "checkpoints.sqlite") as saver:
        graph = _graph(Brain(plans={CLAIM.claim_number: PLAN}), checkpointer=saver)
        await run_claim(graph, CLAIM, "t-checkpoint")
        snapshot = await graph.aget_state({"configurable": {"thread_id": "t-checkpoint"}})
    assert snapshot.values["final"].outcome is RecommendedOutcome.APPROVE
    assert snapshot.values["tool_call_log"][0].call_id == "T1"
    assert "unregistered type" not in caplog.text

"""The claim triage graph.

    intake -> investigator -> rules -> decision -> reflection -> retry_investigation | route_final
    (intake failure -> END with status "failed")

Agentic nodes: intake (reading documents), investigator (choosing tools), decision (judging coverage) and
reflection (judging grounding). Deterministic nodes: rules (fraud and eligibility checks) and route_final (the
outcome). The outcome is always computed by route_final from the RiskReport plus two agent values; no agent
output can override a rule result.

Every node is wrapped by ``_traced``, which streams node_started / node_finished events and records a NodeRun
(the future claim_runs row). The investigator additionally streams one event per tool call. This module has no
FastAPI imports: the API streams these events to the browser in Phase 7.
"""

import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Literal, get_args, get_type_hints

import aiosqlite
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from app.agents.decision import run_decision
from app.agents.intake import IntakeFailedError, run_intake
from app.agents.investigator import Emit, Investigator
from app.agents.llm import LlmClient
from app.agents.reflection import cited_retrieval_confidence, run_reflection
from app.agents.state import (
    MAX_INVESTIGATION_STEPS,
    ClaimInput,
    ClaimState,
    FinalRecommendation,
    NodeRun,
    ResolvedCitation,
)
from app.rules import RoutingInput, evaluate_all, route_final
from app.rules.models import RuleContext

NodeName = Literal["intake", "investigator", "rules", "decision", "reflection", "route_final"]


@dataclass(frozen=True)
class AgentDependencies:
    """Everything the graph needs from outside, injected so tests can swap in fakes."""

    llm: LlmClient
    tools: Sequence[BaseTool]
    load_rule_context: Callable[[ClaimInput], Awaitable[RuleContext]]
    max_investigation_steps: int = MAX_INVESTIGATION_STEPS


@dataclass(frozen=True)
class NodeResult:
    """What a node function returns: the state update plus what to record as the node's input and output."""

    update: dict[str, Any]
    input: dict[str, Any]
    output: dict[str, Any]


NodeFunction = Callable[[ClaimState, Emit], Awaitable[NodeResult]]


def _traced(name: NodeName, function: NodeFunction) -> Callable[[ClaimState], Awaitable[dict[str, Any]]]:
    """Wrap a node: stream start/finish events, time it, and append its NodeRun.

    Args:
        name: Node name, as shown in the UI trace.
        function: The node's logic.

    Returns:
        A LangGraph node.
    """

    async def node(state: ClaimState) -> dict[str, Any]:
        # A no-op unless the caller streams with stream_mode "custom"; then each dict becomes one SSE event.
        emit = get_stream_writer()
        round_number = state.get("investigation_round", 1)
        started_at = datetime.now(UTC)
        emit({"type": "node_started", "node": name, "round": round_number, "at": started_at.isoformat()})
        start = time.perf_counter()
        result = await function(state, emit)
        latency_ms = round((time.perf_counter() - start) * 1000)
        run = NodeRun(
            agent_name=name, round=round_number, input=result.input, output=result.output,
            latency_ms=latency_ms, started_at=started_at,
        )  # fmt: skip
        emit({"type": "node_finished", "node": name, "round": round_number, "latency_ms": latency_ms,
              "output": result.output})  # fmt: skip
        return {**result.update, "node_runs": [run]}

    return node


def _first_line(text: str, limit: int = 200) -> str:
    line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return line if len(line) <= limit else f"{line[: limit - 3]}..."


def finalise(state: ClaimState) -> FinalRecommendation:
    """Deterministic final routing. Pure function of the state; no LLM.

    Confidence given to routing is the LOWER of the agent's confidence and the retrieval confidence of the
    weakest cited passage, so the spec's "retrieval confidence < 0.65 escalates" holds even for a confident agent.

    Args:
        state: Graph state after reflection.

    Returns:
        The recommendation for the human reviewer.
    """
    claim, report = state["claim"], state["risk_report"]
    decision, evidence = state.get("decision"), state.get("evidence", [])
    flags = state.get("escalation_flags", [])
    retrieval = cited_retrieval_confidence(decision, evidence)
    # str() first so the float's binary error never reaches a threshold; 3 places matches NUMERIC(4,3).
    agent = Decimal(str(decision.confidence)).quantize(Decimal("0.001")) if decision is not None else None
    confidence = min(agent, retrieval) if agent is not None else Decimal("0.000")
    routing = route_final(
        RoutingInput(
            risk_report=report,
            claimed_amount=claim.claimed_amount,
            confidence=confidence,
            # With no decision there is no coverage judgment; a decision failure always adds an escalation flag,
            # so this placeholder can never produce an approval. True keeps a misleading "not covered" reason out.
            covered=decision.covered if decision is not None else True,
            forced_escalation_reason="; ".join(flags) or None,
        )
    )
    passages = {passage.evidence_id: passage for passage in evidence}
    citations: dict[str, ResolvedCitation] = {}
    for point in decision.rationale if decision is not None else ():
        if point.basis != "policy_wording":
            continue
        for evidence_id in point.evidence_ids:
            passage = passages.get(evidence_id)
            if passage is not None and evidence_id not in citations:
                citations[evidence_id] = ResolvedCitation(
                    evidence_id=evidence_id, document=passage.document, page=passage.page,
                    section=passage.section, quote=point.quote, label=passage.label,
                )  # fmt: skip
    # The latest review's questions: earlier ones were either answered by the retry or asked again.
    reflections = state.get("reflections", [])
    open_questions = reflections[-1].open_questions if reflections else ()
    return FinalRecommendation(
        outcome=routing.outcome,
        reasons=routing.reasons,
        covered=decision.covered if decision is not None else None,
        agent_confidence=agent,
        retrieval_confidence=retrieval,
        confidence=confidence,
        risk_score=report.risk_score,
        hard_blocks=tuple(report.hard_blocks),
        triggered_rules=tuple(report.triggered_rule_ids),
        summary=decision.summary if decision is not None else "No decision was produced; see the escalation reasons.",
        citations=tuple(citations.values()),
        escalation_flags=tuple(flags),
        open_questions=open_questions,
    )


def build_claim_graph(
    deps: AgentDependencies, checkpointer: BaseCheckpointSaver[Any] | None = None
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Assemble and compile the triage graph.

    Args:
        deps: LLM, tools and the rule-context loader.
        checkpointer: Persists state after every node (SQLite in dev); None for no persistence.

    Returns:
        The compiled graph. Invoke with ``{"claim": ClaimInput}`` and a ``thread_id``.
    """
    investigator = Investigator(deps.llm, deps.tools, max_steps=deps.max_investigation_steps)
    # What the critic may ask the investigator to use: a lookup naming anything else goes to the reviewer instead.
    tool_catalogue = {tool.name: _first_line(tool.description) for tool in deps.tools}

    async def intake_node(state: ClaimState, emit: Emit) -> NodeResult:
        claim = state["claim"]
        run_input = {"claim_number": claim.claim_number, "has_document": claim.document is not None}
        try:
            result = await run_intake(deps.llm, claim)
        except IntakeFailedError as error:
            return NodeResult(
                {"status": "failed", "failure_reason": str(error), "intake": None},
                run_input,
                {"failed": True, "attempts": error.attempts, "errors": list(error.errors)},
            )
        return NodeResult(
            {"status": "running", "intake": result, "investigation_round": 1},
            run_input,
            result.model_dump(mode="json"),
        )

    async def investigator_node(state: ClaimState, emit: Emit) -> NodeResult:
        round_number = state.get("investigation_round", 1)
        reflections = state.get("reflections", [])
        feedback = reflections[-1].feedback if round_number > 1 and reflections else ()
        output = await investigator.run(
            state["claim"],
            state.get("intake"),
            round_number=round_number,
            prior_calls=len(state.get("tool_call_log", [])),
            evidence=state.get("evidence", []),
            previous=state.get("investigations", []),
            feedback=feedback,
            emit=emit,
        )
        item = output.round
        flags = [f"Investigation round {item.round} stopped early: {item.stop_reason}"] if item.stopped_early else []
        return NodeResult(
            {
                "investigations": [item],
                "tool_call_log": list(output.tool_calls),
                "evidence": list(output.new_evidence),
                "escalation_flags": flags,
            },
            {"round": round_number, "feedback": list(feedback)},
            {
                "findings": item.findings.model_dump(mode="json"),
                "steps_used": item.steps_used,
                "stopped_early": item.stopped_early,
                "stop_reason": item.stop_reason,
                # Stored in full (claim_runs.output): the UI's tool log and citation sheet are rebuilt from this row.
                "tool_calls": [record.model_dump(mode="json") for record in output.tool_calls],
                "new_evidence": [passage.model_dump(mode="json") for passage in output.new_evidence],
            },
        )

    async def rules_node(state: ClaimState, emit: Emit) -> NodeResult:
        # DETERMINISTIC: plain Python over database facts. Nothing an agent produced is an input here.
        context = await deps.load_rule_context(state["claim"])
        report = evaluate_all(context)
        return NodeResult({"risk_report": report}, context.model_dump(mode="json"), report.model_dump(mode="json"))

    async def decision_node(state: ClaimState, emit: Emit) -> NodeResult:
        outcome = await run_decision(
            deps.llm,
            state["claim"],
            state.get("intake"),
            state.get("investigations", []),
            state.get("tool_call_log", []),
            state.get("evidence", []),
            state["risk_report"],
        )
        flags = (
            [] if outcome.parsed else [f"The decision agent could not produce a valid decision: {outcome.errors[-1]}"]
        )
        return NodeResult(
            {"decision": outcome.parsed, "escalation_flags": flags},
            {"evidence_ids": [p.evidence_id for p in state.get("evidence", [])], "attempts": outcome.attempts},
            outcome.parsed.model_dump(mode="json")
            if outcome.parsed
            else {"failed": True, "errors": list(outcome.errors)},
        )

    async def reflection_node(state: ClaimState, emit: Emit) -> NodeResult:
        round_number = state.get("investigation_round", 1)
        result = await run_reflection(
            deps.llm,
            reflection_round=len(state.get("reflections", [])) + 1,
            retries_used=round_number - 1,
            claim=state["claim"],
            draft=state.get("decision"),
            rounds=state.get("investigations", []),
            tool_calls=state.get("tool_call_log", []),
            evidence=state.get("evidence", []),
            risk_report=state["risk_report"],
            tools=tool_catalogue,
            previous=state.get("reflections", []),
        )
        update: dict[str, Any] = {"reflections": [result]}
        if result.action == "retry_investigation":
            update["investigation_round"] = round_number + 1
        elif result.action == "escalate":
            update["escalation_flags"] = [
                f"Reflection could not confirm the decision is grounded: {'; '.join(result.feedback)}"
            ]
        return NodeResult(update, {"round": result.round}, result.model_dump(mode="json"))

    async def route_final_node(state: ClaimState, emit: Emit) -> NodeResult:
        # DETERMINISTIC: the routing table from app/rules/routing.py decides the outcome.
        final = finalise(state)
        return NodeResult({"final": final, "status": "completed"}, {}, final.model_dump(mode="json"))

    def after_intake(state: ClaimState) -> Literal["failed", "investigator"]:
        return "failed" if state.get("status") == "failed" else "investigator"

    def after_reflection(state: ClaimState) -> Literal["retry_investigation", "route_final"]:
        return "retry_investigation" if state["reflections"][-1].action == "retry_investigation" else "route_final"

    graph = StateGraph(ClaimState)
    graph.add_node("intake", _traced("intake", intake_node))
    graph.add_node("investigator", _traced("investigator", investigator_node))
    graph.add_node("rules", _traced("rules", rules_node))
    graph.add_node("decision", _traced("decision", decision_node))
    graph.add_node("reflection", _traced("reflection", reflection_node))
    graph.add_node("route_final", _traced("route_final", route_final_node))
    graph.add_edge(START, "intake")
    graph.add_conditional_edges("intake", after_intake, {"failed": END, "investigator": "investigator"})
    graph.add_edge("investigator", "rules")
    graph.add_edge("rules", "decision")
    graph.add_edge("decision", "reflection")
    graph.add_conditional_edges(
        "reflection", after_reflection, {"retry_investigation": "investigator", "route_final": "route_final"}
    )
    graph.add_edge("route_final", END)
    return graph.compile(checkpointer=checkpointer)


# ---- checkpointing ------------------------------------------------------------------------------------------------


def _state_types() -> set[type]:
    """Every Pydantic model and enum that can appear in ClaimState, found by walking the type hints."""
    found: set[type] = set()

    def visit(annotation: Any) -> None:
        for argument in get_args(annotation):
            visit(argument)
        if isinstance(annotation, type) and annotation not in found:
            if issubclass(annotation, BaseModel):
                found.add(annotation)
                for model_field in annotation.model_fields.values():
                    visit(model_field.annotation)
            elif issubclass(annotation, Enum):
                found.add(annotation)

    for hint in get_type_hints(ClaimState, include_extras=True).values():
        visit(hint)
    return found


def checkpoint_serializer() -> JsonPlusSerializer:
    """A serializer that will only rebuild our own state types from a checkpoint.

    LangGraph's default rebuilds any class named in a checkpoint. A tampered checkpoint file could then create
    arbitrary objects; an explicit allowlist closes that door (and silences LangGraph's deprecation warning).

    Returns:
        The configured serializer.
    """
    return JsonPlusSerializer(allowed_msgpack_modules=[(t.__module__, t.__qualname__) for t in _state_types()])


@asynccontextmanager
async def open_checkpointer(path: Path) -> AsyncIterator[AsyncSqliteSaver]:
    """Open the SQLite checkpointer.

    SQLite (per the spec) suits one API process. Several workers would need the Postgres checkpointer instead,
    because SQLite allows only one writer at a time.

    Args:
        path: Database file; created if missing.

    Yields:
        A checkpointer that closes its connection on exit.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(str(path)) as connection:
        yield AsyncSqliteSaver(connection, serde=checkpoint_serializer())


async def stream_claim(
    graph: CompiledStateGraph[Any, Any, Any, Any], claim: ClaimInput, thread_id: str
) -> AsyncIterator[dict[str, Any]]:
    """Run the graph for one claim, yielding node and tool-call events as they happen.

    Args:
        graph: The compiled graph.
        claim: The claim to triage.
        thread_id: Checkpoint thread; use a new one per run.

    Yields:
        Event dicts: node_started, tool_call_started, tool_call_finished, node_finished.
    """
    config = {"configurable": {"thread_id": thread_id}}
    async for event in graph.astream({"claim": claim}, config, stream_mode="custom"):
        yield event


async def run_claim(graph: CompiledStateGraph[Any, Any, Any, Any], claim: ClaimInput, thread_id: str) -> ClaimState:
    """Run the graph to completion and return the final state.

    Args:
        graph: The compiled graph.
        claim: The claim to triage.
        thread_id: Checkpoint thread; use a new one per run.

    Returns:
        The final ClaimState.
    """
    result: ClaimState = await graph.ainvoke({"claim": claim}, {"configurable": {"thread_id": thread_id}})
    return result

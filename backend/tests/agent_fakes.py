"""Test doubles for the agent layer: a scripted chat model, stub tools and claim fixtures.

ScriptedChatModel is a real LangChain BaseChatModel, so bind_tools, with_structured_output (include_raw) and
with_fallbacks run through LangChain's own code; only the model's replies are scripted. The Brain decides each
reply from the tools it was offered, the same signal a real model's behaviour depends on.
"""

import itertools
import re
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, tool
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.agents.state import ClaimInput
from app.domain.enums import IncidentType, KycStatus, PolicyStatus, ProductType
from app.retrieval.chunking import SectionKind
from app.retrieval.retriever import Citation, RetrievedChunk, SearchResult
from app.rules.models import ClaimFacts, CustomerFacts, PolicyFacts, PriorClaim, RuleContext
from app.tools.records import (
    PolicyStatusResult,
    SimilarClaimsResult,
    build_claim_history,
    build_payment_history,
)

INCIDENT = date(2026, 8, 1)
_ids = itertools.count(1)

COVERAGE_TEXT = (
    "The insurer will indemnify the insured against loss of or damage to the insured vehicle caused by "
    "accidental external means, including collision and overturning."
)
EXCLUSION_TEXT = (
    "The insurer shall not be liable for any loss or damage sustained while the vehicle is driven by a person "
    "under the influence of intoxicating liquor or drugs."
)
WAITING_TEXT = "Claims arising within the first 30 days of cover are subject to additional verification."


# ---- claims and rule contexts ----------------------------------------------------------------------------------


def make_claim(number: str = "CLM-2026-000101", amount: str = "42000.00", **overrides: Any) -> ClaimInput:
    values: dict[str, Any] = {
        "claim_id": uuid.uuid4(),
        "claim_number": number,
        "policy_number": "MOT-2025-000011",
        "product_type": ProductType.MOTOR,
        "incident_type": IncidentType.COLLISION,
        "incident_date": INCIDENT,
        "claimed_amount": Decimal(amount),
        "description": "Rear-ended at a traffic signal; bumper and tail lamp damaged.",
        "submitted_at": datetime(2026, 8, 3, 10, 0, tzinfo=UTC),
    }
    return ClaimInput(**(values | overrides))


def rule_context_for(
    claim: ClaimInput, *, policy_start: date = date(2025, 1, 1), lapsed_on: date | None = None, prior: int = 0
) -> RuleContext:
    """A rule context around ``claim``; by default no rule triggers."""
    policy_id = uuid.uuid4()
    return RuleContext(
        claim=ClaimFacts(
            claim_id=claim.claim_id, policy_id=policy_id, incident_date=claim.incident_date,
            claimed_amount=claim.claimed_amount, submitted_at=claim.submitted_at,
            submitted_on=claim.submitted_at.date(),
        ),
        policy=PolicyFacts(
            policy_id=policy_id, start_date=policy_start, end_date=policy_start + timedelta(days=729),
            lapsed_on=lapsed_on, sum_insured=Decimal("500000.00"),
        ),
        customer=CustomerFacts(customer_id=uuid.uuid4(), kyc_status=KycStatus.VERIFIED),
        other_claims=tuple(
            PriorClaim(
                claim_id=uuid.uuid4(), policy_id=policy_id, incident_date=claim.incident_date - timedelta(days=30 * n),
                claimed_amount=Decimal("10000.00"), submitted_at=claim.submitted_at - timedelta(days=30 * n),
            )
            for n in range(1, prior + 1)
        ),
    )  # fmt: skip


def loader_for(contexts: dict[str, RuleContext]) -> Callable[[ClaimInput], Any]:
    async def load(claim: ClaimInput) -> RuleContext:
        return contexts[claim.claim_number]

    return load


# ---- stub tools ---------------------------------------------------------------------------------------------------


def _search(query: str, kind: SectionKind, text: str, score: float) -> SearchResult:
    chunk = RetrievedChunk(
        chunk_id=f"{kind.value}-{abs(hash(text)) % 10_000}",
        text=text,
        citation=Citation(document="Motor_Policy.pdf", page={"coverage": 2, "exclusions": 3}.get(kind.value, 4),
                          section=kind.value),
        section_kind=kind, product_type=ProductType.MOTOR, score=score, char_start=0, char_end=len(text),
    )  # fmt: skip
    return SearchResult(
        query=query, product_type=ProductType.MOTOR, section_kinds=(kind,), chunks=(chunk,),
        retrieval_confidence=min(1.0, max(0.0, (score - 0.2) / 0.35)),
    )  # fmt: skip


def stub_tools(*, policy_score: float = 0.6, failing: set[str] | None = None) -> list[BaseTool]:
    """The 9 investigator tools with canned results. ``failing`` names tools that raise."""
    failing = failing or set()

    def maybe_fail(name: str) -> None:
        if name in failing:
            raise ConnectionError(f"{name} backend down")

    @tool(response_format="content_and_artifact")
    async def search_policy(query: str, product_type: str | None = None) -> tuple[str, SearchResult]:
        """Search policy wording. Use this for general questions. Do NOT use it for exclusions."""
        maybe_fail("search_policy")
        result = _search(query, SectionKind.GENERAL, WAITING_TEXT, policy_score)
        return f"<context>{result.chunks[0].text}</context>", result

    @tool(response_format="content_and_artifact")
    async def check_exclusions(product_type: str, incident_description: str) -> tuple[str, SearchResult]:
        """Find exclusions. Use this for exclusions. Do NOT use it for coverage."""
        maybe_fail("check_exclusions")
        result = _search(incident_description, SectionKind.EXCLUSIONS, EXCLUSION_TEXT, policy_score)
        return f"<context>{result.chunks[0].text}</context>", result

    @tool(response_format="content_and_artifact")
    async def get_waiting_period(product_type: str, condition_or_event: str) -> tuple[str, SearchResult]:
        """Find waiting periods. Use this for new policies. Do NOT use it for exclusions."""
        maybe_fail("get_waiting_period")
        result = _search(condition_or_event, SectionKind.WAITING_PERIODS, WAITING_TEXT, policy_score)
        return f"<context>{result.chunks[0].text}</context>", result

    @tool(response_format="content_and_artifact")
    async def get_coverage_section(product_type: str, incident_type: str) -> tuple[str, SearchResult]:
        """Find coverage. Use this to confirm cover. Do NOT use it for exclusions."""
        maybe_fail("get_coverage_section")
        result = _search(incident_type, SectionKind.COVERAGE, COVERAGE_TEXT, policy_score)
        return f"<context>{result.chunks[0].text}</context>", result

    @tool(response_format="content_and_artifact")
    async def get_policy_status(policy_number: str, on_date: str | None = None) -> tuple[str, PolicyStatusResult]:
        """Policy record. Use this first. Do NOT use it for payments."""
        maybe_fail("get_policy_status")
        result = PolicyStatusResult(
            policy_number=policy_number, product_type=ProductType.MOTOR, status=PolicyStatus.ACTIVE,
            start_date=date(2025, 1, 1), end_date=date(2026, 12, 31), lapsed_on=None,
            sum_insured=Decimal("500000.00"), deductible=Decimal("5000.00"), policy_document="Motor_Policy.pdf",
            on_date=INCIDENT, in_force_on_date=True, days_since_start=577,
        )  # fmt: skip
        return result.model_dump_json(), result

    @tool(response_format="content_and_artifact")
    async def get_claim_history(policy_number: str, exclude_claim_number: str | None = None) -> tuple[str, Any]:
        """Claim history. Use this for frequency. Do NOT use it for duplicates."""
        maybe_fail("get_claim_history")
        result = build_claim_history("CUS-00011", [], INCIDENT, exclude_claim_number)
        return result.model_dump_json(), result

    @tool(response_format="content_and_artifact")
    async def get_customer_profile(policy_number: str) -> tuple[str, Any]:
        """Customer facts. Use this for KYC context. Do NOT use it for claims."""
        maybe_fail("get_customer_profile")
        return '{"kyc_status":"verified"}', None

    @tool(response_format="content_and_artifact")
    async def check_similar_claims(claim_number: str) -> tuple[str, SimilarClaimsResult]:
        """Similar claims. Use this for duplicates. Do NOT use it for history."""
        maybe_fail("check_similar_claims")
        result = SimilarClaimsResult(
            claim_number=claim_number, incident_type=IncidentType.COLLISION, incident_date=INCIDENT,
            claimed_amount=Decimal("42000.00"), exact_duplicates=0, same_customer_matches=0,
            other_customer_matches=0, matches=(),
        )  # fmt: skip
        return result.model_dump_json(), result

    @tool(response_format="content_and_artifact")
    async def get_payment_history(policy_number: str, on_date: str | None = None) -> tuple[str, Any]:
        """Payments. Use this for lapses. Do NOT use it for active policies."""
        maybe_fail("get_payment_history")
        policy = {"policy_number": policy_number, "status": PolicyStatus.ACTIVE, "lapsed_on": None}
        result = build_payment_history(policy, [], None)
        return result.model_dump_json(), result

    tools = [
        search_policy, check_exclusions, get_waiting_period, get_coverage_section, get_policy_status,
        get_claim_history, get_customer_profile, check_similar_claims, get_payment_history,
    ]  # fmt: skip
    for item in tools:
        item.handle_tool_error = True
    return tools


# ---- the scripted brain -------------------------------------------------------------------------------------------

Step = list[tuple[str, dict[str, Any]]]
_CLAIM_NUMBER = re.compile(r"claim_number: (CLM-\d{4}-\d{6})")
_EVIDENCE = re.compile(r"\[(E\d+)\] [^\n]*\n([^\n]+)")
_TOOL_CALL = re.compile(r"\[(T\d+)\] ")


def _call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "args": args, "id": f"call_{next(_ids)}", "type": "tool_call"}


def _text(messages: Sequence[BaseMessage]) -> str:
    return "\n".join(str(message.content) for message in messages)


@dataclass
class Brain:
    """Scripted model behaviour, keyed by what the model is being asked to do.

    plans: per claim number, the investigator's steps (each a list of tool calls); after the plan, it submits.
    intake: queue of extraction-args factories; empty queue means "extract the claim correctly".
    critiques: queue of Critique args; empty means accept.
    """

    plans: dict[str, list[Step]] = field(default_factory=dict)
    default_plan: list[Step] = field(default_factory=list)  # for claims without their own plan (API tests)
    intake: list[Callable[[str], dict[str, Any]] | None] = field(default_factory=list)
    critiques: list[dict[str, Any]] = field(default_factory=list)
    decision: Callable[[str], dict[str, Any]] | None = None
    never_submit: bool = False
    fail_forced_submit: bool = False
    raise_on: set[str] = field(default_factory=set)
    calls: list[tuple[str, list[BaseMessage], Any]] = field(default_factory=list)
    _progress: dict[str, int] = field(default_factory=dict)

    def respond(self, messages: list[BaseMessage], tools: list[dict[str, Any]], tool_choice: Any) -> AIMessage:
        names = {spec["function"]["name"] for spec in tools}
        role = next((n for n in ("IntakeExtraction", "DecisionDraft", "Critique") if n in names), "investigator")
        if role == "investigator" and tool_choice:
            role = "forced_submit"
        self.calls.append((role, list(messages), tool_choice))
        if role in self.raise_on:
            raise RuntimeError(f"provider down ({role})")
        text = _text(messages)
        if role == "IntakeExtraction":
            factory = self.intake.pop(0) if self.intake else None
            return AIMessage("", tool_calls=[_call(role, (factory or default_extraction)(text))])
        if role == "DecisionDraft":
            return AIMessage("", tool_calls=[_call(role, (self.decision or default_decision)(text))])
        if role == "Critique":
            args = self.critiques.pop(0) if self.critiques else {"grounded": True, "verdict": "accept"}
            return AIMessage("", tool_calls=[_call(role, args)])
        if role == "forced_submit":
            if self.fail_forced_submit:
                return AIMessage("I cannot.")
            return AIMessage("", tool_calls=[_call("submit_findings", {"summary": "Partial findings at the limit."})])
        claim_number = _CLAIM_NUMBER.search(text).group(1)  # type: ignore[union-attr]
        # Count this claim's investigator turns in the current conversation (messages grow within a round).
        turns = sum(1 for message in messages if isinstance(message, AIMessage))
        plan = self.plans.get(claim_number, self.default_plan)
        if self.never_submit or turns < len(plan):
            step = plan[turns % len(plan)] if plan else [("get_policy_status", {"policy_number": "MOT-2025-000011"})]
            return AIMessage("", tool_calls=[_call(name, args) for name, args in step])
        return AIMessage("", tool_calls=[_call("submit_findings", {
            "summary": f"Investigated {claim_number} with {len(plan)} steps.",
            "key_facts": ["get_policy_status: in force on the incident date"],
        })])  # fmt: skip


def default_extraction(text: str) -> dict[str, Any]:
    return {
        "policy_number": "MOT-2025-000011",
        "incident_type": "collision",
        "incident_date": INCIDENT.isoformat(),
        "claimed_amount": _amount(text),
        "incident_summary": "The insured vehicle was rear-ended at a signal.",
    }


def _amount(text: str) -> str:
    match = re.search(r"claimed_amount: ([\d.]+)", text)
    return match.group(1) if match else "42000.00"


def default_decision(text: str) -> dict[str, Any]:
    """Cite the first evidence passage with a real quote, and the first tool call as claim data."""
    evidence = _EVIDENCE.search(text)
    tool_call = _TOOL_CALL.search(text)
    rationale = []
    if evidence:
        quote = " ".join(evidence.group(2).split()[:8])
        rationale.append({"statement": "The wording covers accidental collision damage.", "basis": "policy_wording",
                          "evidence_ids": [evidence.group(1)], "quote": quote})  # fmt: skip
    if tool_call:
        rationale.append({"statement": "Cover was in force on the incident date.", "basis": "claim_data",
                          "tool_call_ids": [tool_call.group(1)]})  # fmt: skip
    return {"covered": True, "confidence": 0.9, "summary": "Collision damage is covered and cover was in force.",
            "rationale": rationale or [{"statement": "No evidence.", "basis": "rule_result"}]}  # fmt: skip


class ScriptedChatModel(BaseChatModel):
    """A chat model whose replies come from a Brain."""

    brain: Any

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], *, tool_choice: Any = None, **kwargs: Any) -> Any:  # type: ignore[override]
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], tool_choice=tool_choice, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        reply = self.brain.respond(messages, kwargs.get("tools", []), kwargs.get("tool_choice"))
        return ChatResult(generations=[ChatGeneration(message=reply)])

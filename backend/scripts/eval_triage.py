"""Evaluate the triage pipeline end to end on labelled claims, and write the report.

Each case borrows a seeded claim's facts (policy, customer, dates, amount) and replaces its description with the
case's story, then runs the real graph: real LLM, real tools, real rules. Nothing is written to the database: the
graph only reads, and the run happens in memory, so the claims in the console are untouched.

Results are appended to a JSON Lines file as each case finishes, so an interrupted run loses nothing and the next
run continues where it stopped. The free tiers allow only a few dozen claims a day; when every provider is out of
its daily quota the run stops instead of recording failures.

Usage (inside the backend container):
    python -m scripts.eval_triage                 # run every case not yet scored, then write the report
    python -m scripts.eval_triage --only M2 H2    # just these cases (again, even if already scored)
    python -m scripts.eval_triage --kind rule     # just the rule edge cases
    python -m scripts.eval_triage --list          # show which seeded claim each case uses; no LLM calls
    python -m scripts.eval_triage --report        # rebuild the report from saved results; no LLM calls
Exit code 1 if any claim that should be rejected or escalated was approved.
"""

import argparse
import asyncio
import sys
import time
import uuid
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from sqlalchemy import func, select

from app.agents.evaluation import (
    CaseResult,
    CitedSection,
    ClaimSelector,
    EvalCase,
    FreeTier,
    Usage,
    append_result,
    count_wasted_retries,
    is_daily_quota,
    load_eval_set,
    load_results,
    provider_failure,
    render_report,
    summarise,
)
from app.agents.graph import AgentDependencies, build_claim_graph
from app.agents.llm import BATCH_WAIT, LlmNotConfiguredError, create_llm_client
from app.agents.loaders import build_investigator_tools, load_claim_input, load_rule_context, rule_context_loader
from app.agents.state import ClaimState
from app.core.config import Settings, get_settings
from app.db.models import Claim, Policy
from app.db.readonly import ReadOnlyDatabase, create_readonly_engine
from app.db.vector import create_qdrant_client
from app.retrieval.embeddings import get_embedder
from app.retrieval.retriever import PolicyRetriever
from app.rules import evaluate_all

EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"
CASES_PATH = EVAL_DIR / "triage_cases.json"
RESULTS_PATH = EVAL_DIR / "triage_results.jsonl"
REPORT_PATH = EVAL_DIR / "triage_report.md"
# Groq's free tier as measured for this project (response headers, September 2026).
GROQ_FREE_TIER = FreeTier(name="Groq free tier", tokens_per_minute=8_000, requests_per_day=1_000)
# A pause between claims lets the per-minute token budget refill, so one claim's retries don't spill into the next.
DEFAULT_PAUSE_S = 30.0


class UsageCounter(AsyncCallbackHandler):
    """Counts LLM calls and tokens for one claim (every chat model call ends in on_llm_end)."""

    def __init__(self) -> None:
        """Start from zero."""
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.by_model: Counter[str] = Counter()

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Add one call's usage.

        Args:
            response: The model's result.
            **kwargs: Run metadata (unused).
        """
        self.calls += 1
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None) or {}
                tokens_in, tokens_out = int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
                self.input_tokens += tokens_in
                self.output_tokens += tokens_out
                metadata = getattr(message, "response_metadata", None) or {}
                model = metadata.get("model_name") or metadata.get("model") or "unknown"
                self.by_model[str(model)] += tokens_in + tokens_out

    def usage(self) -> Usage:
        """Totals so far."""
        return Usage(
            llm_calls=self.calls, input_tokens=self.input_tokens, output_tokens=self.output_tokens,
            tokens_by_model=dict(self.by_model),
        )  # fmt: skip


async def find_base_claim(database: ReadOnlyDatabase, selector: ClaimSelector, exclude: set[str]) -> str:
    """Pick the seeded claim whose facts a case uses.

    Args:
        database: Read-only database.
        selector: The case's criteria.
        exclude: Claims already used by earlier cases, so each case gets its own policy and customer.

    Returns:
        The claim number.

    Raises:
        LookupError: If the seed has no matching claim.
    """
    if selector.edge_case is not None:
        statement = select(Claim.claim_number).where(Claim.claim_number.like(f"CLM-____-9{selector.edge_case:05d}"))
        async with database.connect() as connection:
            number = await connection.scalar(statement)
        if number is None:
            raise LookupError("seeded edge cases not found; run scripts.seed")
        return str(number)

    statement = (
        select(Claim.claim_number, Claim.id, Claim.incident_date, Policy.start_date, Policy.customer_id)
        .join(Policy, Policy.id == Claim.policy_id)
        .where(
            Policy.product_type == selector.product_type,
            Claim.incident_type == selector.incident_type,
            Claim.claimed_amount <= selector.max_amount,
            Claim.claim_number.not_like("CLM-____-9%"),  # not a labelled edge case
        )
        .order_by(Claim.claim_number)
    )
    async with database.connect() as connection:
        rows = (await connection.execute(statement)).all()
        for row in rows:
            age = (row.incident_date - row.start_date).days
            if row.claim_number in exclude or age < (selector.min_policy_age_days or 0):
                continue
            if selector.max_policy_age_days is not None and age > selector.max_policy_age_days:
                continue
            if selector.first_policy:
                earlier = select(func.count()).where(
                    Policy.customer_id == row.customer_id, Policy.start_date < row.start_date
                )
                if await connection.scalar(earlier):
                    continue
            # The case's story must decide the outcome, so no rule may trigger on the borrowed facts.
            if not evaluate_all(await load_rule_context(database, row.id)).triggered_rule_ids:
                return str(row.claim_number)
    raise LookupError(f"no seeded claim matches {selector.model_dump(exclude_defaults=True)}")


def _texts(state: ClaimState) -> Iterable[str]:
    """Everything a run wrote that could name a failed model call."""
    if state.get("failure_reason"):
        yield str(state["failure_reason"])
    yield from state.get("escalation_flags", [])
    for reflection in state.get("reflections", []):
        yield from reflection.feedback
    final = state.get("final")
    if final is not None:
        yield from final.reasons


def to_result(case: EvalCase, claim_number: str, state: ClaimState, usage: Usage, latency_s: float) -> CaseResult:
    """Record what the pipeline did with one case.

    Args:
        case: The case.
        claim_number: The seeded claim that supplied the facts.
        state: Final graph state.
        usage: LLM calls and tokens.
        latency_s: Wall-clock time.

    Returns:
        The result to store.
    """
    final = state.get("final")
    rounds = len(state.get("investigations", []))
    limited = provider_failure(_texts(state))
    status = "provider_limited" if limited else ("completed" if final is not None else "failed")
    return CaseResult(
        case_id=case.id,
        claim_number=claim_number,
        ran_at=datetime.now(UTC),
        status=status,
        outcome=final.outcome if final else None,
        covered=final.covered if final else None,
        confidence=final.confidence if final else None,
        reasons=final.reasons if final else (),
        citations=tuple(CitedSection(document=c.document, page=c.page, section=c.section) for c in final.citations)
        if final
        else (),
        investigation_rounds=rounds,
        retries=max(rounds - 1, 0),
        wasted_retries=count_wasted_retries(state.get("tool_call_log", []), state.get("evidence", []), rounds),
        open_questions=final.open_questions if final else (),
        tool_calls=len(state.get("tool_call_log", [])),
        latency_s=round(latency_s, 1),
        usage=usage,
        note=limited or (None if final else state.get("failure_reason")),
    )


def write_report(cases: Sequence[EvalCase], results_path: Path, report_path: Path, settings: Settings) -> int:
    """Rebuild the report from saved results.

    Args:
        cases: The labelled cases.
        results_path: Saved results.
        report_path: Where to write the Markdown report.
        settings: For the model names.

    Returns:
        Number of unsafe approvals.
    """
    summary, scores = summarise(cases, load_results(results_path))
    models = f"{settings.groq_model} (Groq), falling back to {settings.gemini_model} (Gemini)"
    report_path.write_text(render_report(summary, scores, models=models, free_tier=GROQ_FREE_TIER), encoding="utf-8")
    print(
        f"\nScored {summary.scored}/{summary.total_cases}: {summary.acceptable} acceptable, {summary.exact} exact, "
        f"{len(summary.unsafe_approvals)} unsafe approval(s), {summary.retries} retries ({summary.wasted_retries} "
        f"wasted). Report: {report_path}"
    )
    return len(summary.unsafe_approvals)


async def run_cases(cases: Sequence[EvalCase], args: argparse.Namespace, settings: Settings) -> None:
    """Run the pipeline on each case and append the results.

    Args:
        cases: Cases to run.
        args: Command line options.
        settings: Application settings.
    """
    database = ReadOnlyDatabase(create_readonly_engine(settings))
    qdrant = create_qdrant_client(settings)
    try:
        used: set[str] = set()
        bases: list[tuple[EvalCase, str]] = []
        for case in cases:
            number = await find_base_claim(database, case.base, used)
            used.add(number)
            bases.append((case, number))
        if args.list:
            for case, number in bases:
                print(f"{case.id:<4} {number}  {case.title}")
            return

        try:
            llm = create_llm_client(settings, wait=BATCH_WAIT)
        except LlmNotConfiguredError as error:
            print(f"eval_triage: {error}", file=sys.stderr)
            raise SystemExit(2) from error
        embedder = await asyncio.to_thread(get_embedder, settings.embedding_model_name)
        retriever = PolicyRetriever(qdrant, embedder, settings.qdrant_collection)
        deps = AgentDependencies(
            llm=llm,
            tools=build_investigator_tools(retriever, database),
            load_rule_context=rule_context_loader(database),
        )
        graph = build_claim_graph(deps)  # no checkpointer: evaluation runs leave nothing behind
        done = load_results(args.results)
        todo = [(case, number) for case, number in bases if args.only or not (case.id in done and done[case.id].scored)]
        print(f"{len(todo)} case(s) to run ({len(bases) - len(todo)} already scored).", flush=True)
        for index, (case, number) in enumerate(todo, start=1):
            if index > 1:
                await asyncio.sleep(args.pause)
            claim = (await load_claim_input(database, number)).model_copy(update={"description": case.description})
            counter = UsageCounter()
            started = time.perf_counter()
            thread = f"eval-{case.id}-{uuid.uuid4().hex[:8]}"
            config: Any = {"configurable": {"thread_id": thread}, "callbacks": [counter]}
            state: ClaimState = await graph.ainvoke({"claim": claim}, config)
            result = to_result(case, number, state, counter.usage(), time.perf_counter() - started)
            await asyncio.to_thread(append_result, args.results, result)
            got = result.outcome.value.upper() if result.outcome else result.status
            verdict = "ok" if result.outcome in case.acceptable else "MISS"
            print(
                f"[{index}/{len(todo)}] {case.id:<4} {got:<9} expected {case.expected_outcome.value:<8} {verdict:<4} "
                f"{result.latency_s:5.0f} s {result.usage.total_tokens:>7,} tokens {result.usage.llm_calls:>3} calls "
                f"{result.retries} retries  {case.title}",
                flush=True,
            )
            if result.status == "provider_limited":
                print(f"      provider limit: {result.note}", flush=True)
                if result.note and is_daily_quota(result.note):
                    print("Stopped: the free daily quota is used up. Run the same command tomorrow to continue.")
                    return
    finally:
        await qdrant.close()
        await database.dispose()


def main() -> None:
    """Parse arguments, run the cases, write the report."""
    parser = argparse.ArgumentParser(description="Evaluate the triage pipeline on labelled claims.")
    parser.add_argument("--cases", type=Path, default=CASES_PATH, help="labelled cases (JSON)")
    parser.add_argument("--results", type=Path, default=RESULTS_PATH, help="results file (JSON Lines, appended)")
    parser.add_argument("--output", type=Path, default=REPORT_PATH, help="where to write the Markdown report")
    parser.add_argument("--only", nargs="+", metavar="ID", help="run just these cases, even if already scored")
    parser.add_argument("--kind", choices=("coverage", "rule"), help="run just one kind of case")
    parser.add_argument("--pause", type=float, default=DEFAULT_PAUSE_S, help="seconds between claims")
    parser.add_argument("--list", action="store_true", help="show the seeded claim each case uses; no LLM calls")
    parser.add_argument("--report", action="store_true", help="only rebuild the report from saved results")
    args = parser.parse_args()

    settings = get_settings()
    cases = list(load_eval_set(args.cases).cases)
    if args.only:
        unknown = set(args.only) - {case.id for case in cases}
        if unknown:
            parser.error(f"unknown case id(s): {', '.join(sorted(unknown))}")
    selected = [c for c in cases if (not args.only or c.id in args.only) and (not args.kind or c.kind == args.kind)]
    if not args.report:
        asyncio.run(run_cases(selected, args, settings))
        if args.list:
            return
    sys.exit(1 if write_report(cases, args.results, args.output, settings) else 0)


if __name__ == "__main__":
    main()

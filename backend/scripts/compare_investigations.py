"""Phase 6 acceptance check: a simple and a complex claim must produce DIFFERENT tool call logs.

If both claims made the same tool calls in the same order, the "agent" would be a fixed workflow wearing an
agent's name. This script runs the real graph (real LLM, real tools, seeded database) on both claims, prints the
two logs side by side, and exits 1 if the tool sequences are identical.

Defaults:
- simple:  the lowest-value clean motor collision claim, on a policy running for 6+ months, no rule triggered
- complex: the seeded R02 edge case (incident after the policy lapsed for a missed premium)

Usage:
    python -m scripts.compare_investigations
    python -m scripts.compare_investigations --simple CLM-2026-000012 --complex CLM-2026-900004 --output out.json
"""

import argparse
import asyncio
import json
import sys
import textwrap
import uuid
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.agents.graph import AgentDependencies, build_claim_graph, open_checkpointer, run_claim
from app.agents.llm import LlmNotConfiguredError, create_llm_client
from app.agents.loaders import build_investigator_tools, load_claim_input, load_rule_context, rule_context_loader
from app.agents.state import ClaimState
from app.core.config import get_settings
from app.db.models import Claim, Policy
from app.db.readonly import ReadOnlyDatabase, create_readonly_engine
from app.db.vector import create_qdrant_client
from app.domain.enums import IncidentType
from app.retrieval.embeddings import get_embedder
from app.retrieval.retriever import PolicyRetriever
from app.rules import evaluate_all

COLUMN = 56
# A "simple" claim: small, on a policy that has been running long enough that no new-policy concern applies.
SIMPLE_MAX_AMOUNT = Decimal("30000.00")
SIMPLE_MIN_POLICY_AGE_DAYS = 180


async def pick_simple_claim(database: ReadOnlyDatabase) -> str:
    """Find the lowest-value clean motor collision claim on an established policy.

    The policy's current status is not filtered: most seeded policies have since expired or renewed, and what
    matters is that cover was in force on the incident date, which the rules engine confirms below.

    Args:
        database: Read-only database.

    Returns:
        Its claim number.

    Raises:
        LookupError: If the seed has no such claim.
    """
    statement = (
        select(Claim.claim_number, Claim.id, Claim.incident_date, Policy.start_date)
        .join(Policy, Policy.id == Claim.policy_id)
        .where(
            Claim.incident_type == IncidentType.COLLISION,
            Claim.claimed_amount <= SIMPLE_MAX_AMOUNT,
            Claim.claim_number.not_like("CLM-____-9%"),  # not a labelled edge case
        )
        .order_by(Claim.claimed_amount, Claim.claim_number)
    )
    async with database.connect() as connection:
        rows = (await connection.execute(statement)).all()
    for row in rows:
        if row.incident_date - row.start_date < timedelta(days=SIMPLE_MIN_POLICY_AGE_DAYS):
            continue
        # Confirm with the real rules engine that nothing triggers, rather than trusting the filter alone.
        if not evaluate_all(await load_rule_context(database, row.id)).triggered_rule_ids:
            return str(row.claim_number)
    raise LookupError("no clean low-value motor collision claim found; reseed with scripts.seed --reset")


async def find_edge_claim(database: ReadOnlyDatabase, sequence: int) -> str:
    """Find a labelled edge-case claim (CLM-<year>-9000NN; NN = 1..8 for rules R01..R08).

    Args:
        database: Read-only database.
        sequence: Edge-case number.

    Returns:
        Its claim number.
    """
    statement = select(Claim.claim_number).where(Claim.claim_number.like(f"CLM-____-9{sequence:05d}"))
    async with database.connect() as connection:
        number = await connection.scalar(statement)
    if number is None:
        raise LookupError("seeded edge cases not found; run scripts.seed")
    return str(number)


def _cell(record: Any) -> str:
    args = ", ".join(f"{key}={value}" for key, value in record.args.items())
    return f"{record.call_id} r{record.round} {record.tool}({args})"


def side_by_side(simple: ClaimState, complex_: ClaimState) -> str:
    """Format two tool call logs as columns.

    Args:
        simple: Final state of the simple claim.
        complex_: Final state of the complex claim.

    Returns:
        A printable table.
    """
    left = [_cell(r) for r in simple.get("tool_call_log", [])]
    right = [_cell(r) for r in complex_.get("tool_call_log", [])]
    lines = [
        f"{'#':>2}  {'SIMPLE ' + simple['claim'].claim_number:<{COLUMN}}  COMPLEX {complex_['claim'].claim_number}",
        "-" * (COLUMN * 2 + 6),
    ]
    for index in range(max(len(left), len(right))):
        a = textwrap.shorten(left[index], COLUMN) if index < len(left) else ""
        b = textwrap.shorten(right[index], COLUMN) if index < len(right) else ""
        lines.append(f"{index + 1:>2}  {a:<{COLUMN}}  {b}")
    lines.append("-" * (COLUMN * 2 + 6))
    for label, getter in (
        ("tool calls", lambda s: str(len(s.get("tool_call_log", [])))),
        ("investigation rounds", lambda s: str(len(s.get("investigations", [])))),
        ("outcome", lambda s: s["final"].outcome.value.upper() if s.get("final") else s.get("status", "?")),
        ("confidence (agent/retrieval)", lambda s: f"{s['final'].agent_confidence}/{s['final'].retrieval_confidence}"
         if s.get("final") else "-"),
        ("citations", lambda s: ", ".join(c.label for c in s["final"].citations) if s.get("final") else "-"),
    ):  # fmt: skip
        lines.append(f"    {label + ': ' + getter(simple):<{COLUMN}}  {label}: {getter(complex_)}")
    return "\n".join(lines)


def write_output(states: dict[str, ClaimState], output: Path) -> None:
    """Save both tool call logs and outcomes as JSON (evidence for the README and the evaluation).

    Args:
        states: Final states by label.
        output: Destination file.
    """
    payload = {
        label: {
            "claim_number": state["claim"].claim_number,
            "tool_call_log": [r.model_dump(mode="json") for r in state.get("tool_call_log", [])],
            "final": state["final"].model_dump(mode="json") if state.get("final") else None,
        }
        for label, state in states.items()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


async def run(simple_number: str | None, complex_number: str | None, output: Path | None) -> int:
    """Run both claims through the graph and compare their tool logs.

    Args:
        simple_number: Simple claim, or None to pick one.
        complex_number: Complex claim, or None for the R02 edge case.
        output: Optional JSON file for both final states.

    Returns:
        0 if the logs differ, 1 if they are identical, 2 if the LLM is not configured.
    """
    settings = get_settings()
    try:
        llm = create_llm_client(settings)
    except LlmNotConfiguredError as error:
        print(f"compare_investigations: {error}", file=sys.stderr)
        return 2
    database = ReadOnlyDatabase(create_readonly_engine(settings))
    qdrant = create_qdrant_client(settings)
    try:
        embedder = await asyncio.to_thread(get_embedder, settings.embedding_model_name)
        retriever = PolicyRetriever(qdrant, embedder, settings.qdrant_collection)
        deps = AgentDependencies(
            llm=llm,
            tools=build_investigator_tools(retriever, database),
            load_rule_context=rule_context_loader(database),
        )
        simple_number = simple_number or await pick_simple_claim(database)
        complex_number = complex_number or await find_edge_claim(database, 2)
        states: dict[str, ClaimState] = {}
        async with open_checkpointer(settings.checkpoint_path) as saver:
            graph = build_claim_graph(deps, saver)
            for label, number in (("simple", simple_number), ("complex", complex_number)):
                print(f"Running {label} claim {number} ...", flush=True)
                claim = await load_claim_input(database, number)
                states[label] = await run_claim(graph, claim, f"{number}-{uuid.uuid4().hex[:8]}")
    finally:
        await qdrant.close()
        await database.dispose()

    print()
    print(side_by_side(states["simple"], states["complex"]))
    if output is not None:
        # File writes block, so they run in a worker thread rather than on the event loop.
        await asyncio.to_thread(write_output, states, output)
        print(f"\nWrote {output}")

    sequence = {label: [r.tool for r in state.get("tool_call_log", [])] for label, state in states.items()}
    if sequence["simple"] == sequence["complex"]:
        print("\nFAIL: both claims made the same tool calls in the same order - a workflow, not an agent.")
        return 1
    print("\nPASS: the investigator chose different tools for the two claims.")
    return 0


def main() -> None:
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(description="Prove the investigator's tool use differs between claims.")
    parser.add_argument("--simple", help="claim number of the simple claim (default: auto-picked)")
    parser.add_argument("--complex", dest="complex_", help="claim number of the complex claim (default: R02 edge case)")
    parser.add_argument("--output", type=Path, help="write both tool call logs and outcomes to this JSON file")
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args.simple, args.complex_, args.output)))


if __name__ == "__main__":
    main()

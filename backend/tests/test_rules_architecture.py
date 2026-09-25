"""Enforce that the rules package stays pure: no I/O libraries, no other app layers, no clock or randomness."""

import ast
from pathlib import Path

import pytest

RULES_DIR = Path(__file__).resolve().parent.parent / "app" / "rules"
RULE_FILES = sorted(RULES_DIR.glob("*.py"))

FORBIDDEN_IMPORT_PREFIXES = (
    "sqlalchemy", "asyncpg", "fastapi", "starlette", "httpx", "requests", "qdrant_client",
    "langchain", "langgraph", "groq", "google", "openai", "sentence_transformers",
    "app.db", "app.api", "app.services", "random", "os", "socket", "subprocess",
)  # fmt: skip
# Reading the clock inside a rule would make the same claim evaluate differently on different days.
FORBIDDEN_CALLS = {"now", "today", "utcnow", "time", "perf_counter"}


def _imports(tree: ast.AST) -> list[str]:
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_rule_files_exist() -> None:
    assert {path.name for path in RULE_FILES} >= {"checks.py", "engine.py", "models.py", "routing.py"}


@pytest.mark.parametrize("path", RULE_FILES, ids=lambda path: path.name)
def test_rules_import_no_io_or_other_layers(path: Path) -> None:
    for name in _imports(ast.parse(path.read_text(encoding="utf-8"))):
        assert not any(name == p or name.startswith(p + ".") for p in FORBIDDEN_IMPORT_PREFIXES), f"{path.name}: {name}"


@pytest.mark.parametrize("path", RULE_FILES, ids=lambda path: path.name)
def test_rules_never_read_the_clock(path: Path) -> None:
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in FORBIDDEN_CALLS, f"{path.name}:{node.lineno} calls .{node.func.attr}()"


def test_every_rule_is_registered_once_with_spec_names() -> None:
    from app.rules.checks import RULES

    assert [(r.rule_id, r.name) for r in RULES] == [
        ("R01", "amount_exceeds_sum_insured"),
        ("R02", "policy_lapsed_at_incident_date"),
        ("R03", "claim_within_30d_of_policy_start"),
        ("R04", "more_than_3_claims_in_12_months"),
        ("R05", "incident_date_in_future"),
        ("R06", "incident_date_before_policy_start"),
        ("R07", "duplicate_claim_same_date_amount"),
        ("R08", "kyc_incomplete"),
    ]
    assert {r.rule_id for r in RULES if r.hard_block} == {"R02", "R05", "R06"}
    assert all(r.weight == 0 for r in RULES if r.hard_block)

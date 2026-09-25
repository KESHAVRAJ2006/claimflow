"""Static guard against SQL built from strings anywhere in the backend.

SQL injection needs a value to be pasted into SQL text. These checks make that impossible to merge unnoticed:
every ``text()`` statement must be a string literal (values go in ``:params``), and nothing may execute a
computed string except the one reviewed DDL helper, whose values Postgres quotes itself.
"""

import ast
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
PYTHON_FILES = sorted([*(BACKEND / "app").rglob("*.py"), *(BACKEND / "scripts").rglob("*.py")])
# The only file allowed to call exec_driver_sql: role DDL cannot take bind parameters (see _execute_ddl).
DRIVER_SQL_ALLOWED = {BACKEND / "scripts" / "provision_readonly_role.py"}


def _calls(path: Path) -> list[ast.Call]:
    return [node for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))) if isinstance(node, ast.Call)]


def _name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _is_string_literal(node: ast.expr) -> bool:
    # Adjacent literals ("SELECT " "1") are merged by the parser into one Constant, so they pass too.
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


@pytest.mark.parametrize("path", PYTHON_FILES, ids=lambda path: str(path.relative_to(BACKEND)))
def test_text_statements_are_string_literals(path: Path) -> None:
    for call in _calls(path):
        if _name(call) == "text" and call.args:
            assert _is_string_literal(call.args[0]), f"{path.name}:{call.lineno} text() of a computed string"


@pytest.mark.parametrize("path", PYTHON_FILES, ids=lambda path: str(path.relative_to(BACKEND)))
def test_nothing_executes_a_computed_string(path: Path) -> None:
    for call in _calls(path):
        name = _name(call)
        if name == "exec_driver_sql":
            assert path in DRIVER_SQL_ALLOWED, f"{path.name}:{call.lineno} exec_driver_sql outside the DDL helper"
        if name in {"execute", "scalar", "scalars"} and call.args:
            first = call.args[0]
            # f-strings, "+" concatenation, % formatting and .format() are the ways a value sneaks into SQL.
            built = isinstance(first, ast.JoinedStr | ast.BinOp) or (
                isinstance(first, ast.Call) and _name(first) == "format"
            )
            assert not built and not _is_string_literal(first), f"{path.name}:{call.lineno} executes a raw string"


def test_sql_tools_never_use_text() -> None:
    source = (BACKEND / "app" / "tools" / "sql_tools.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names}
    assert "text" not in imported, "sql_tools.py must build queries with Core select(), not text()"

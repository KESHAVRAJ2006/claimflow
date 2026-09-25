"""Layer boundaries of the orchestration package."""

import ast
from pathlib import Path

import pytest

from app.agents.graph import AgentDependencies, build_claim_graph
from app.agents.investigator import INVESTIGATOR_TOOL_NAMES
from app.agents.llm import LlmClient
from tests.agent_fakes import Brain, ScriptedChatModel, loader_for, stub_tools

AGENT_FILES = sorted((Path(__file__).resolve().parent.parent / "app" / "agents").glob("*.py"))


@pytest.mark.parametrize("path", AGENT_FILES, ids=lambda path: path.name)
def test_orchestration_never_imports_the_api_layer(path: Path) -> None:
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else []
        if isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
        for name in names:
            assert not name.startswith(("fastapi", "starlette", "app.api")), f"{path.name} imports {name}"


def test_graph_has_the_specified_nodes() -> None:
    deps = AgentDependencies(LlmClient([ScriptedChatModel(brain=Brain())]), stub_tools(), loader_for({}))
    nodes = set(build_claim_graph(deps).get_graph().nodes) - {"__start__", "__end__"}
    assert nodes == {"intake", "investigator", "rules", "decision", "reflection", "route_final"}


def test_stub_tools_match_the_specified_tool_set() -> None:
    assert {tool.name for tool in stub_tools()} == INVESTIGATOR_TOOL_NAMES

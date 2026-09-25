"""Orchestration layer: the LangGraph agents and the graph that connects them to the deterministic core.

No FastAPI imports here (enforced by tests/test_agents_architecture.py): the API layer drives the graph, never the
other way round, so the same graph runs from a script, a test, or the MCP server.
"""

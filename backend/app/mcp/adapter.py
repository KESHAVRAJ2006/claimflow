"""Expose the investigator's LangChain tools as MCP tools, without a second implementation.

WHY AN ADAPTER, NOT NEW TOOLS. The 9 tools already exist, with tested queries, a SELECT-only database login, the
<context> fence around retrieved text, and docstrings written for an LLM. Re-declaring them with ``@mcp.tool``
would copy all of that, and the copies would drift. The adapter reuses each LangChain tool as-is: its name, its
description, its argument schema (so MCP clients get the same argument descriptions) and its error handling.

What an MCP client receives is exactly what our own investigator receives: the same text, with policy passages
inside the same <context> fence. The structured artifact is deliberately not sent as ``structuredContent``: for
policy searches it holds the raw passages outside the fence, and a client could feed that to its model.
"""

import logging
import time
import uuid
from typing import Any

from fastmcp.exceptions import ToolError
from fastmcp.tools import Tool, ToolResult
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from mcp.types import TextContent, ToolAnnotations
from pydantic import ConfigDict, PrivateAttr

logger = logging.getLogger("claimflow.mcp")

# Every tool only reads: the SQL tools through a SELECT-only login, the policy tools from a vector index.
# Clients use these hints to skip confirmation prompts for safe tools; ``open_world_hint=False`` says the tools
# touch only ClaimFlow's own data, never the internet.
READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)

# A database statement times out after 5s (TOOLS_STATEMENT_TIMEOUT_MS) and a search takes well under 1s;
# 30s leaves room for a cold embedding model and still stops a hung call from holding a client forever.
TOOL_TIMEOUT_S = 30.0


def summarize_artifact(artifact: object, content: str) -> str:
    """One line describing a tool result, for the log.

    Args:
        artifact: The tool's structured result; every ClaimFlow result model has a ``summary()`` method.
        content: The text result, used when there is no artifact (for example an error message).

    Returns:
        A short summary, at most 200 characters.
    """
    summary = getattr(artifact, "summary", None)
    text = summary() if callable(summary) else content
    return text if len(text) <= 200 else f"{text[:197]}..."


class LangChainToolAdapter(Tool):
    """An MCP tool that runs one LangChain tool and returns its text result."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    # PrivateAttr keeps the LangChain tool out of the MCP tool's serialized definition.
    _tool: BaseTool = PrivateAttr()

    @classmethod
    def wrap(cls, tool: BaseTool) -> "LangChainToolAdapter":
        """Build the MCP definition from the LangChain tool's own name, description and argument schema.

        Args:
            tool: A ClaimFlow investigator tool.

        Returns:
            The MCP tool.
        """
        schema = tool.tool_call_schema.model_json_schema()
        adapter = cls(
            name=tool.name,
            description=tool.description,
            # The docstring's one-line summary is already the description, so it is dropped from the schema.
            parameters={key: value for key, value in schema.items() if key not in {"title", "description"}},
            annotations=READ_ONLY,
            timeout=TOOL_TIMEOUT_S,
        )
        adapter._tool = tool
        return adapter

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Invoke the LangChain tool and turn its result into an MCP result.

        A bad identifier or a missing record comes back as an error result the client's model can read and
        correct, exactly as in our own ReAct loop. Anything else (database down, a failed read-only check) is
        logged in full here and reported to the client without details, so connection strings and stack traces
        never leave the server.

        Args:
            arguments: The client's arguments, already checked against the input schema by FastMCP.

        Returns:
            The text result, flagged as an error when the tool reported one.

        Raises:
            ToolError: On an unexpected failure.
        """
        started = time.perf_counter()
        # Invoking with a ToolCall (not bare args) returns a ToolMessage, which carries both the error status
        # and the artifact; bare args would return only the text.
        call = {"name": self.name, "args": arguments, "id": f"mcp-{uuid.uuid4().hex[:12]}", "type": "tool_call"}
        try:
            message = await self._tool.ainvoke(call)
        except Exception as error:
            logger.exception(
                "mcp tool failed",
                extra={"tool": self.name, "tool_args": arguments, "latency_ms": _elapsed_ms(started)},
            )
            raise ToolError(f"{self.name} is unavailable right now; the failure was logged on the server.") from error
        if not isinstance(message, ToolMessage):  # a tool built without response_format returns plain text
            message = ToolMessage(content=str(message), tool_call_id=call["id"])
        content = message.text
        is_error = message.status == "error"
        # The same {tool, args, result_summary, latency_ms} record the investigator keeps for every call.
        logger.info(
            "mcp tool call",
            extra={
                "tool": self.name,
                "tool_args": arguments,  # "args" is reserved by logging.LogRecord
                "status": "error" if is_error else "ok",
                "result_summary": summarize_artifact(message.artifact, content),
                "latency_ms": _elapsed_ms(started),
            },
        )
        return ToolResult(content=[TextContent(type="text", text=content)], is_error=is_error)


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)

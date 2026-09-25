"""Run the ClaimFlow MCP server.

    python -m scripts.run_mcp                     # stdio, for Claude Desktop (started by the client itself)
    python -m scripts.run_mcp --transport http    # streamable HTTP on 127.0.0.1:8765, bearer token required

Over stdio, stdout carries the MCP protocol, so every log line goes to stderr.
"""

import argparse
import asyncio
import logging
import sys

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.mcp.server import SharedTokenVerifier, build_mcp_server, open_mcp_dependencies

logger = logging.getLogger("claimflow.mcp")


async def serve(settings: Settings, transport: str, host: str, port: int) -> None:
    """Create the dependencies, then serve until the client disconnects or the process is stopped.

    Args:
        settings: Application settings.
        transport: "stdio" or "http".
        host: HTTP bind address.
        port: HTTP port.

    Raises:
        SystemExit: If HTTP is requested without a valid MCP_AUTH_TOKEN.
    """
    auth = None
    if transport == "http":
        if settings.mcp_auth_token is None:
            raise SystemExit("MCP_AUTH_TOKEN must be set for the HTTP transport (32+ random characters)")
        try:
            auth = SharedTokenVerifier(settings.mcp_auth_token.get_secret_value())
        except ValueError as error:
            raise SystemExit(str(error)) from None
    async with open_mcp_dependencies(settings) as deps:
        server = build_mcp_server(deps.tools, deps.database, auth=auth)
        if transport == "stdio":
            # show_banner=False: the banner is decoration, and stdio clients read nothing but protocol messages.
            await server.run_stdio_async(show_banner=False)
        else:
            logger.info("mcp http listening", extra={"host": host, "port": port, "path": "/mcp"})
            await server.run_http_async(transport="http", host=host, port=port, show_banner=False)


def main() -> None:
    """Parse arguments and run the server."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="ClaimFlow MCP server (read-only claim investigation tools)")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default=settings.mcp_http_host, help="HTTP bind address (default: loopback)")
    parser.add_argument("--port", type=int, default=settings.mcp_http_port)
    args = parser.parse_args()
    configure_logging(settings.log_level, stream=sys.stderr)
    # FastMCP installs its own (Rich, stderr) handlers at import; send its records through our JSON handler too.
    fastmcp_logger = logging.getLogger("fastmcp")
    fastmcp_logger.handlers = []
    fastmcp_logger.propagate = True
    asyncio.run(serve(settings, args.transport, args.host, args.port))


if __name__ == "__main__":
    main()

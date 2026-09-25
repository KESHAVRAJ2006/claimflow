"""Read-only database access for the agent tools.

Three independent barriers stop an agent tool from writing, so no single mistake opens a write path:

1. Grants. The tools log in as a role that holds SELECT on four tables and nothing else
   (created by ``scripts/provision_readonly_role.py``). This is the real barrier.
2. Session default. Every connection starts its transactions READ ONLY, so even a wrongly granted role cannot
   write by accident. A session could switch this off with SET, which is why it is only the second barrier.
3. Privilege probe. Before the first query, ``verify_read_only_privileges`` asks Postgres what the logged-in
   role can actually do and refuses to run if it could write anything. This fails closed when someone
   points TOOLS_DATABASE_URL at the owner account.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from app.core.config import Settings

# NOLOGIN group role that holds the grants; the login user is a member. Keeping grants on a group means
# rotating the login user (or adding a second one for the MCP server) never touches the grants.
READONLY_GROUP_ROLE = "claimflow_agent_readonly"
# Least privilege: the tools need these four tables. claim_runs, audit_log and alembic_version stay invisible.
AGENT_READABLE_TABLES = ("customers", "policies", "premium_payments", "claims")
WRITE_PRIVILEGES = ("INSERT", "UPDATE", "DELETE", "TRUNCATE", "REFERENCES", "TRIGGER")

_TABLE_PRIVILEGE_PROBE = text(
    """
    SELECT c.relname AS table_name, p.privilege
    FROM pg_catalog.pg_class AS c
    JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
    CROSS JOIN unnest(CAST(:privileges AS text[])) AS p(privilege)
    WHERE n.nspname = 'public'
      AND c.relkind IN ('r', 'p', 'v', 'm')
      AND has_table_privilege(c.oid, p.privilege)
    ORDER BY c.relname, p.privilege
    """
)
_ROLE_PROBE = text(
    """
    SELECT r.rolname, r.rolsuper, r.rolcreaterole, r.rolcreatedb, r.rolbypassrls, r.rolreplication,
           has_schema_privilege('public', 'CREATE') AS can_create_in_schema,
           has_database_privilege(current_database(), 'CREATE') AS can_create_schema
    FROM pg_catalog.pg_roles AS r
    WHERE r.rolname = current_user
    """
)
_DANGEROUS_ROLE_FLAGS = (
    "rolsuper", "rolcreaterole", "rolcreatedb", "rolbypassrls", "rolreplication",
    "can_create_in_schema", "can_create_schema",
)  # fmt: skip


class ReadOnlyViolationError(RuntimeError):
    """The tools' database role can do more than read the allowed tables."""


def create_readonly_engine(settings: Settings) -> AsyncEngine:
    """Create the connection pool the agent tools use.

    Args:
        settings: Application settings; ``tools_database_url`` must be set.

    Returns:
        An AsyncEngine whose sessions default to read-only transactions with a statement timeout.

    Raises:
        RuntimeError: If TOOLS_DATABASE_URL is not configured.
    """
    if settings.tools_database_url is None:
        raise RuntimeError("TOOLS_DATABASE_URL is not set; the agent tools need their own read-only login")
    return create_async_engine(
        settings.tools_database_url.get_secret_value(),
        pool_pre_ping=True,
        # The investigator calls tools one at a time per claim; a small pool also respects Render's connection cap.
        pool_size=3,
        max_overflow=2,
        connect_args={
            # asyncpg sends these at connection start, so they apply before our first statement runs.
            "server_settings": {
                "default_transaction_read_only": "on",
                # A slow query must not stall the agent loop; the tool fails and the agent reports it instead.
                "statement_timeout": str(settings.tools_statement_timeout_ms),
                "application_name": "claimflow-agent-tools",
            }
        },
    )


async def verify_read_only_privileges(connection: AsyncConnection) -> None:
    """Check that the connected role can only read the allowed tables.

    Args:
        connection: A connection logged in as the role to check.

    Raises:
        ReadOnlyViolationError: Listing every privilege the role should not have.
    """
    problems: list[str] = []
    role = (await connection.execute(_ROLE_PROBE)).mappings().one()
    problems += [f"role {role['rolname']!r} has {flag}" for flag in _DANGEROUS_ROLE_FLAGS if role[flag]]

    rows = await connection.execute(_TABLE_PRIVILEGE_PROBE, {"privileges": [*WRITE_PRIVILEGES, "SELECT"]})
    for table_name, privilege in rows:
        if privilege != "SELECT":
            problems.append(f"{privilege} on {table_name}")
        elif table_name not in AGENT_READABLE_TABLES:
            problems.append(f"SELECT on {table_name}, which the agent tools do not need")
    if problems:
        raise ReadOnlyViolationError(
            "Refusing to run agent tools: the tools database role is not read-only ("
            + "; ".join(problems)
            + "). Point TOOLS_DATABASE_URL at the login created by scripts.provision_readonly_role."
        )


class ReadOnlyDatabase:
    """Hands out connections for the agent tools, after verifying once that the role cannot write."""

    def __init__(self, engine: AsyncEngine) -> None:
        """Wrap an engine created by ``create_readonly_engine``.

        Args:
            engine: The tools' connection pool.
        """
        self._engine = engine
        self._verified = False
        # Serialises the first check, so concurrent first calls don't each run the probe.
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[AsyncConnection]:
        """Open a pooled connection for one tool call.

        Yields:
            A connection inside a read-only transaction; it is rolled back when the block exits.

        Raises:
            ReadOnlyViolationError: If the role could write (checked before the first connection only).
        """
        await self._ensure_verified()
        # engine.connect() returns the connection to the pool with a rollback, so nothing a tool does persists.
        async with self._engine.connect() as connection:
            yield connection

    async def dispose(self) -> None:
        """Close every pooled connection. Call on application shutdown."""
        await self._engine.dispose()

    async def _ensure_verified(self) -> None:
        """Run the privilege probe once per process; lazily, so startup never depends on Postgres being up."""
        if self._verified:
            return
        async with self._lock:
            if not self._verified:
                async with self._engine.connect() as connection:
                    await verify_read_only_privileges(connection)
                self._verified = True

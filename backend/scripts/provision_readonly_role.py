"""Create or update the read-only database login the agent tools use. Idempotent; runs on every startup.

Roles live here rather than in an Alembic migration because they are cluster-wide (not part of one database's
schema) and carry a password, which does not belong in migration history. Running on every startup also means
that neither ``scripts.migrate`` adopting existing tables nor ``scripts.reset_db`` dropping the schema can leave
the grants missing for long: the next start puts them back.

What it does, in one transaction:
1. Ensures the NOLOGIN group role exists, strips every table privilege from it, then grants SELECT on exactly
   the tables in ``AGENT_READABLE_TABLES`` (converging, so a stray manual GRANT is removed).
2. Ensures the login user from TOOLS_DATABASE_URL exists with that password, is a member of the group, holds no
   direct table privileges, and defaults to read-only transactions.
3. Logs in as that user and runs the same privilege probe the tools run, so a bad setup fails here, loudly.

Usage:
    python -m scripts.provision_readonly_role
"""

import asyncio
import re
import sys

from sqlalchemy import pool, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from app.core.config import get_settings
from app.db.readonly import AGENT_READABLE_TABLES, READONLY_GROUP_ROLE, verify_read_only_privileges

# Lower-case unquoted Postgres identifier, so the name means the same thing quoted or not.
USERNAME_PATTERN = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
# Long enough not to be guessable by brute force. Base64 characters are allowed because Render's generateValue
# produces base64; Settings percent-encodes the password when it builds TOOLS_DATABASE_URL, and the DDL quotes it
# with %L, so none of these characters can break out of either.
PASSWORD_PATTERN = re.compile(r"^[A-Za-z0-9_+/=-]{12,128}$")

_ROLE_EXISTS = text("SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = :name)")
_FORMAT_DDL = text("SELECT format(:template, VARIADIC CAST(:args AS text[]))")


class ProvisioningError(RuntimeError):
    """The configuration cannot be provisioned safely."""


async def _execute_ddl(connection: AsyncConnection, template: str, *args: str) -> None:
    """Run a DDL statement whose identifiers and literals are quoted by Postgres itself.

    Postgres cannot bind parameters into DDL (CREATE ROLE, GRANT), so ``:param`` placeholders don't work there.
    Rather than building the statement with Python string formatting, Postgres's own ``format()`` quotes each
    value (%I = identifier, %L = literal), which is the escaping Postgres guarantees. The templates are
    constants in this file; only the quoted values come from configuration. This is the one place in the
    codebase allowed to execute a computed statement (see tests/test_sql_safety.py).

    Args:
        connection: Owner connection inside the provisioning transaction.
        template: Constant statement template with %I / %L placeholders.
        *args: Values for the placeholders.
    """
    statement = await connection.scalar(_FORMAT_DDL, {"template": template, "args": list(args)})
    # exec_driver_sql, not text(): text() would treat a ':' inside the quoted password as a bind parameter.
    await connection.exec_driver_sql(statement)


async def _role_exists(connection: AsyncConnection, name: str) -> bool:
    return bool(await connection.scalar(_ROLE_EXISTS, {"name": name}))


def check_configuration(owner_url: str, tools_url: str) -> tuple[str, str]:
    """Validate the tools login before touching the database.

    Args:
        owner_url: DATABASE_URL (the schema owner).
        tools_url: TOOLS_DATABASE_URL (the login to provision).

    Returns:
        (username, password) of the tools login.

    Raises:
        ProvisioningError: If the login would be unsafe or unusable.
    """
    owner, tools = make_url(owner_url), make_url(tools_url)
    username, password = tools.username or "", tools.password or ""
    if not USERNAME_PATTERN.fullmatch(username):
        raise ProvisioningError(f"tools username {username!r} must be lower-case letters, digits and _")
    # Provisioning the owner as the tools user would strip the owner's own privileges.
    if username in (owner.username, READONLY_GROUP_ROLE):
        raise ProvisioningError(f"tools username {username!r} must differ from the owner and the group role")
    if not PASSWORD_PATTERN.fullmatch(password):
        raise ProvisioningError("tools password must be 12-128 characters of letters, digits, _ - + / or =")
    # Grants are per database; a login pointed at another database would find no tables.
    if tools.database != owner.database:
        raise ProvisioningError(f"TOOLS_DATABASE_URL database {tools.database!r} != DATABASE_URL {owner.database!r}")
    return username, password


async def provision(owner_url: str, tools_url: str) -> str:
    """Create or update the group role and tools login, then verify them.

    Args:
        owner_url: DATABASE_URL (must be allowed to create roles; the dev container's user is a superuser).
        tools_url: TOOLS_DATABASE_URL.

    Returns:
        The provisioned username.

    Raises:
        ProvisioningError: If the configuration is unsafe.
        ReadOnlyViolationError: If the login can still do more than read after provisioning.
    """
    username, password = check_configuration(owner_url, tools_url)
    owner_engine = create_async_engine(owner_url, poolclass=pool.NullPool)
    try:
        await _grant(owner_engine, username, password)
    except DBAPIError as error:
        # 42501 insufficient_privilege: the owner may not create roles or grant on the tables. Say what to do
        # instead of printing a raw traceback.
        if getattr(error.orig, "sqlstate", None) == "42501":
            raise ProvisioningError(
                "the DATABASE_URL user may not create roles or grant access (needs CREATEROLE and table "
                "ownership). Run this script once as a user that may (for example the provider's admin user), or "
                "create the login in the provider's console and grant it only SELECT on "
                + ", ".join(AGENT_READABLE_TABLES)
                + "; the privilege check at startup then confirms it is read-only"
            ) from error
        raise
    finally:
        await owner_engine.dispose()

    tools_engine = create_async_engine(tools_url, poolclass=pool.NullPool)
    try:
        async with tools_engine.connect() as connection:
            await verify_read_only_privileges(connection)
    finally:
        await tools_engine.dispose()
    return username


async def _grant(owner_engine: AsyncEngine, username: str, password: str) -> None:
    """Create or update the group role and the login, with SELECT on the agent tables only."""
    # One transaction (role DDL is transactional in Postgres): a failure halfway leaves nothing half-granted.
    async with owner_engine.begin() as connection:
        if not await _role_exists(connection, READONLY_GROUP_ROLE):
            await _execute_ddl(connection, "CREATE ROLE %I NOLOGIN", READONLY_GROUP_ROLE)
        await _execute_ddl(connection, "REVOKE ALL ON ALL TABLES IN SCHEMA public FROM %I", READONLY_GROUP_ROLE)
        await _execute_ddl(connection, "REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM %I", READONLY_GROUP_ROLE)
        await _execute_ddl(connection, "GRANT USAGE ON SCHEMA public TO %I", READONLY_GROUP_ROLE)
        for table in AGENT_READABLE_TABLES:
            await _execute_ddl(connection, "GRANT SELECT ON TABLE %I TO %I", table, READONLY_GROUP_ROLE)

        if await _role_exists(connection, username):
            # Re-applying the password keeps the database in step when it is rotated in the environment.
            await _execute_ddl(connection, "ALTER ROLE %I WITH LOGIN PASSWORD %L", username, password)
        else:
            # CREATE ROLE defaults to NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS.
            await _execute_ddl(connection, "CREATE ROLE %I WITH LOGIN INHERIT PASSWORD %L", username, password)
        await _execute_ddl(connection, "REVOKE ALL ON ALL TABLES IN SCHEMA public FROM %I", username)
        await _execute_ddl(connection, "GRANT %I TO %I", READONLY_GROUP_ROLE, username)
        # Also protects anyone who logs in as this user with psql, not just our connection pool.
        await _execute_ddl(connection, "ALTER ROLE %I SET default_transaction_read_only = on", username)


def main() -> int:
    """Provision the tools login if TOOLS_DATABASE_URL is configured.

    Returns:
        Process exit code: 0 on success or when not configured, 1 on failure.
    """
    settings = get_settings()
    if settings.tools_database_url is None:
        print("provision_readonly_role: TOOLS_DATABASE_URL is not set; skipping (agent tools will be unavailable).")
        return 0
    try:
        username = asyncio.run(
            provision(settings.database_url.get_secret_value(), settings.tools_database_url.get_secret_value())
        )
    except Exception as error:  # noqa: BLE001 — report any failure as one clear line and a non-zero exit
        print(f"provision_readonly_role: FAILED: {error}", file=sys.stderr)
        return 1
    print(
        f"provision_readonly_role: {username!r} can SELECT from {', '.join(AGENT_READABLE_TABLES)} and write nothing."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

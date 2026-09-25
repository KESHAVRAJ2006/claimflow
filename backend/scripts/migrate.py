"""Bring the database schema to the latest migration before the API starts.

Plain ``alembic upgrade head`` fails with a confusing "relation already exists" error when tables exist that
Alembic has no record of (created by hand, by ``create_all``, or by an older checkout). This script inspects
the database first and picks a safe action:

- empty database, or one Alembic already tracks  -> apply pending migrations
- untracked tables that exactly match the models -> record them as current (``alembic stamp head``); no data lost
- untracked tables that differ from the models   -> stop and explain how to reset; never guess

Usage:
    python -m scripts.migrate
"""

import asyncio
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import inspect, pool, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

import app.db.models  # noqa: F401 — registers every table on Base.metadata
from app.core.config import get_settings
from app.db.base import Base

ALEMBIC_INI = Path(__file__).resolve().parent.parent / "alembic.ini"
# Created by migration 0001. Autogenerate cannot see triggers, so adoption checks for it explicitly.
APPEND_ONLY_TRIGGER = "audit_log_append_only"
CONNECT_ATTEMPTS = 10
CONNECT_RETRY_DELAY_S = 2.0
MAX_DIFFERENCES_SHOWN = 10


class MigrationAction(StrEnum):
    """What to do with the database schema."""

    UPGRADE = "upgrade"
    ADOPT = "adopt"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class SchemaState:
    """What the database looks like before migrating."""

    current_revision: str | None  # None when Alembic has never recorded a migration here
    existing_app_tables: frozenset[str]
    differences: tuple[str, ...]  # schema differences from the models; only computed for untracked tables


def choose_action(state: SchemaState) -> MigrationAction:
    """Pick the safe migration action for a schema state.

    Args:
        state: The inspected database state.

    Returns:
        UPGRADE when Alembic can proceed normally, ADOPT when untracked tables match the models exactly,
        otherwise CONFLICT.
    """
    if state.current_revision is not None or not state.existing_app_tables:
        return MigrationAction.UPGRADE
    if not state.differences:
        return MigrationAction.ADOPT
    return MigrationAction.CONFLICT


def _inspect_sync(connection: Connection) -> SchemaState:
    """Read the schema state over a synchronous connection facade.

    Args:
        connection: Connection provided by ``AsyncConnection.run_sync``.

    Returns:
        The current SchemaState.
    """
    migration_context = MigrationContext.configure(connection, opts={"compare_type": True})
    current_revision = migration_context.get_current_revision()
    existing = frozenset(set(inspect(connection).get_table_names()) & set(Base.metadata.tables))

    differences: tuple[str, ...] = ()
    if current_revision is None and existing:
        # Same comparison `alembic check` performs: every table, column, type, index and constraint.
        differences = tuple(str(diff) for diff in compare_metadata(migration_context, Base.metadata))
        trigger = connection.execute(
            text("SELECT 1 FROM pg_trigger WHERE tgname = :name"), {"name": APPEND_ONLY_TRIGGER}
        ).first()
        if trigger is None:
            differences += (f"missing trigger {APPEND_ONLY_TRIGGER} on audit_log",)
    return SchemaState(current_revision, existing, differences)


async def inspect_schema(database_url: str, attempts: int = CONNECT_ATTEMPTS) -> SchemaState:
    """Connect to the database, retrying briefly while it starts, and inspect its schema.

    Args:
        database_url: asyncpg connection URL.
        attempts: Connection attempts before giving up.

    Returns:
        The current SchemaState.

    Raises:
        OSError: If the database stays unreachable after all attempts.
    """
    engine = create_async_engine(database_url, poolclass=pool.NullPool)
    try:
        for attempt in range(1, attempts + 1):
            try:
                async with engine.connect() as connection:
                    return await connection.run_sync(_inspect_sync)
            except OSError:  # connection refused / host not resolvable yet
                if attempt == attempts:
                    raise
                print(f"migrate: database not reachable (attempt {attempt}/{attempts}); retrying...", file=sys.stderr)
                await asyncio.sleep(CONNECT_RETRY_DELAY_S)
        raise AssertionError("unreachable")
    finally:
        await engine.dispose()


def _conflict_message(state: SchemaState) -> str:
    """Explain a schema conflict and how to resolve it.

    Args:
        state: The conflicting schema state.

    Returns:
        A multi-line, human-readable message.
    """
    shown = "\n".join(f"  - {diff}" for diff in state.differences[:MAX_DIFFERENCES_SHOWN])
    hidden = len(state.differences) - MAX_DIFFERENCES_SHOWN
    more = f"\n  ... and {hidden} more" if hidden > 0 else ""
    return (
        "migrate: SCHEMA CONFLICT — the database has tables Alembic has no record of, and they do not match\n"
        "the current models (usually tables created by hand or by an older version of the code).\n"
        f"Differences:\n{shown}{more}\n\n"
        "To wipe the development database and rebuild it (deletes all Postgres data; Qdrant is untouched):\n"
        "  docker compose run --rm backend python -m scripts.reset_db --yes\n"
        "then start again with:\n"
        "  docker compose up -d"
    )


def main() -> int:
    """Inspect the schema and apply the safe action.

    Returns:
        Process exit code: 0 on success, 1 on a schema conflict.
    """
    state = asyncio.run(inspect_schema(get_settings().database_url.get_secret_value()))
    action = choose_action(state)
    config = Config(str(ALEMBIC_INI))

    if action is MigrationAction.CONFLICT:
        print(_conflict_message(state), file=sys.stderr)
        return 1
    if action is MigrationAction.ADOPT:
        # The tables already match the latest migration exactly, so recording that fact is lossless.
        print("migrate: found untracked tables that match the models exactly; marking them as current.")
        command.stamp(config, "head")
        return 0
    # Called after asyncio.run has returned: env.py starts its own event loop, which cannot nest.
    command.upgrade(config, "head")
    print("migrate: schema is at the latest migration.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Tests for the startup safety logic: migration action selection and seed action selection.

The pure decision functions are tested directly. The schema inspection is tested against real temporary
Postgres databases (integration; skipped when Postgres is unavailable).
"""

import asyncio
from collections.abc import AsyncIterator
from datetime import date

import pytest
from sqlalchemy import make_url, pool, text
from sqlalchemy.ext.asyncio import create_async_engine

import app.db.models  # noqa: F401 — registers tables
from app.core.config import get_settings
from app.db.base import Base
from scripts.migrate import MigrationAction, SchemaState, choose_action, inspect_schema
from scripts.seed import DatabaseState, SeedAction, decide_seed_action

TODAY = date(2026, 9, 15)
YESTERDAY = date(2026, 9, 14)
APP_TABLES = frozenset(Base.metadata.tables)


# ---- migration decisions --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (SchemaState(None, frozenset(), ()), MigrationAction.UPGRADE),  # empty database
        (SchemaState("0001", APP_TABLES, ()), MigrationAction.UPGRADE),  # tracked by Alembic
        (SchemaState(None, APP_TABLES, ()), MigrationAction.ADOPT),  # untracked but identical
        (SchemaState(None, APP_TABLES, ("remove column x",)), MigrationAction.CONFLICT),
        (SchemaState(None, frozenset({"customers"}), ("add table claims",)), MigrationAction.CONFLICT),
    ],
)
def test_choose_action(state: SchemaState, expected: MigrationAction) -> None:
    assert choose_action(state) is expected


# ---- seed decisions -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "reset", "expected"),
    [
        (DatabaseState(False, None, False), False, SeedAction.SEED),
        (DatabaseState(True, TODAY, False), False, SeedAction.UP_TO_DATE),
        (DatabaseState(True, YESTERDAY, False), False, SeedAction.REFRESH),  # stale and untouched: safe
        (DatabaseState(True, YESTERDAY, True), False, SeedAction.KEEP_USED_DATA),  # stale but used: keep
        (DatabaseState(True, TODAY, True), False, SeedAction.KEEP_USED_DATA),
        (DatabaseState(True, None, False), False, SeedAction.KEEP_UNKNOWN_DATA),  # not ours: keep
        (DatabaseState(True, YESTERDAY, True), True, SeedAction.RESET),  # only --reset deletes used data
    ],
)
def test_decide_seed_action(state: DatabaseState, reset: bool, expected: SeedAction) -> None:
    assert decide_seed_action(state, today=TODAY, reset=reset) is expected


# ---- schema inspection against real temporary databases -------------------------------------------

TEMP_DATABASE = "claimflow_startup_safety_test"
APPEND_ONLY_TRIGGER_SQL = (
    "CREATE FUNCTION audit_log_reject_mutation() RETURNS trigger LANGUAGE plpgsql AS $$ "
    "BEGIN RAISE EXCEPTION 'append-only'; END; $$",
    "CREATE TRIGGER audit_log_append_only BEFORE UPDATE OR DELETE ON audit_log "
    "FOR EACH ROW EXECUTE FUNCTION audit_log_reject_mutation()",
)


@pytest.fixture
async def temp_database_url() -> AsyncIterator[str]:
    main_url = get_settings().database_url.get_secret_value()
    admin = create_async_engine(main_url, poolclass=pool.NullPool, isolation_level="AUTOCOMMIT")
    try:
        async with asyncio.timeout(3):
            async with admin.connect() as connection:
                # CREATE/DROP DATABASE cannot take bind parameters; the name is a fixed constant, never input.
                await connection.execute(text(f'DROP DATABASE IF EXISTS "{TEMP_DATABASE}" WITH (FORCE)'))
                await connection.execute(text(f'CREATE DATABASE "{TEMP_DATABASE}"'))
    except Exception:  # noqa: BLE001 — no reachable Postgres with CREATE DATABASE rights
        await admin.dispose()
        pytest.skip("Postgres with CREATE DATABASE rights is not reachable (run inside docker compose)")

    yield make_url(main_url).set(database=TEMP_DATABASE).render_as_string(hide_password=False)

    async with admin.connect() as connection:
        await connection.execute(text(f'DROP DATABASE IF EXISTS "{TEMP_DATABASE}" WITH (FORCE)'))
    await admin.dispose()


async def _create_tables(url: str, with_trigger: bool) -> None:
    engine = create_async_engine(url, poolclass=pool.NullPool)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        if with_trigger:
            for statement in APPEND_ONLY_TRIGGER_SQL:
                await connection.execute(text(statement))
    await engine.dispose()


@pytest.mark.integration
async def test_empty_database_is_upgraded(temp_database_url: str) -> None:
    state = await inspect_schema(temp_database_url, attempts=1)
    assert choose_action(state) is MigrationAction.UPGRADE


@pytest.mark.integration
async def test_untracked_identical_schema_is_adopted(temp_database_url: str) -> None:
    await _create_tables(temp_database_url, with_trigger=True)
    state = await inspect_schema(temp_database_url, attempts=1)
    assert state.differences == ()
    assert choose_action(state) is MigrationAction.ADOPT


@pytest.mark.integration
async def test_untracked_schema_without_trigger_is_a_conflict(temp_database_url: str) -> None:
    await _create_tables(temp_database_url, with_trigger=False)
    state = await inspect_schema(temp_database_url, attempts=1)
    assert any("audit_log_append_only" in diff for diff in state.differences)
    assert choose_action(state) is MigrationAction.CONFLICT

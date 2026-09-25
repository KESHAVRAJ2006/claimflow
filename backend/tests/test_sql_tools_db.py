"""Integration tests for the SQL tools against the seeded Postgres, including proof that writes raise.

Need a migrated, seeded database and the read-only login (``python -m scripts.provision_readonly_role``, which
docker compose runs on startup). Skipped automatically when any of that is missing.
"""

import asyncio
from collections.abc import AsyncIterator
from datetime import date

import pytest
from langchain_core.messages import ToolMessage
from sqlalchemy import func, pool, select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from app.core.config import get_settings
from app.db.models import Claim, Policy
from app.db.readonly import (
    ReadOnlyDatabase,
    ReadOnlyViolationError,
    create_readonly_engine,
    verify_read_only_privileges,
)
from app.tools.records import MatchReason
from app.tools.sql_tools import build_sql_tools, fetch_policy

pytestmark = pytest.mark.integration

# Statements an attacker (or a bug) might try. Each must fail for the tools' login.
WRITES = {
    "insert": "INSERT INTO audit_log (actor, action) VALUES ('agent', 'agent.tampered')",
    "update": "UPDATE claims SET status = 'approved'",
    "delete": "DELETE FROM premium_payments",
    "truncate": "TRUNCATE claims CASCADE",
    "create": "CREATE TABLE agent_scratch (id integer)",
    "drop": "DROP TABLE customers",
}


@pytest.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    settings = get_settings()
    if settings.tools_database_url is None:
        pytest.skip("TOOLS_DATABASE_URL is not set")
    engine = create_readonly_engine(settings)
    try:
        async with asyncio.timeout(3), engine.connect() as connection:
            await connection.scalar(select(func.count()).select_from(Claim))
    except Exception as error:  # noqa: BLE001 — unreachable, not provisioned, or not migrated: all mean "skip"
        await engine.dispose()
        pytest.skip(f"read-only login unavailable ({type(error).__name__}); run scripts.provision_readonly_role")
    yield engine
    await engine.dispose()


@pytest.fixture
async def conn(engine: AsyncEngine) -> AsyncIterator[AsyncConnection]:
    async with engine.connect() as connection:
        yield connection


@pytest.fixture
def tools(engine: AsyncEngine) -> dict[str, object]:
    return {t.name: t for t in build_sql_tools(ReadOnlyDatabase(engine))}


async def _edge_claim(conn: AsyncConnection, sequence: int) -> tuple[str, str, date]:
    """Find a seeded edge-case claim: CLM-<year>-9000NN, where NN is 1-8 for rules R01-R08 (see scripts/seed.py)."""
    statement = (
        select(Claim.claim_number, Policy.policy_number, Claim.incident_date)
        .join(Policy, Policy.id == Claim.policy_id)
        .where(Claim.claim_number.like(f"CLM-____-9{sequence:05d}"))  # a bound LIKE pattern, not SQL text
    )
    row = (await conn.execute(statement)).one_or_none()
    if row is None:
        pytest.skip("seeded edge cases not found (run scripts.seed)")
    return row.claim_number, row.policy_number, row.incident_date


async def _run(tools: dict[str, object], name: str, **args: object) -> ToolMessage:
    call = {"type": "tool_call", "id": "call-1", "name": name, "args": args}
    return await tools[name].ainvoke(call)  # type: ignore[attr-defined,no-any-return]


# ---- writes must raise ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("statement", WRITES.values(), ids=WRITES.keys())
async def test_writes_raise_in_the_default_read_only_transaction(conn: AsyncConnection, statement: str) -> None:
    with pytest.raises(DBAPIError, match="read-only transaction"):
        await conn.execute(text(statement))


@pytest.mark.parametrize("statement", WRITES.values(), ids=WRITES.keys())
async def test_writes_raise_from_grants_even_in_a_read_write_transaction(conn: AsyncConnection, statement: str) -> None:
    # Switching the session back to read-write is possible for any role, so the grants must stop the write.
    await conn.execute(text("SET TRANSACTION READ WRITE"))
    with pytest.raises(DBAPIError, match="permission denied|must be owner"):
        await conn.execute(text(statement))


@pytest.mark.parametrize("table", ["audit_log", "claim_runs", "alembic_version"])
async def test_tables_the_tools_do_not_need_are_unreadable(conn: AsyncConnection, table: str) -> None:
    statements = {
        "audit_log": "SELECT count(*) FROM audit_log",
        "claim_runs": "SELECT count(*) FROM claim_runs",
        "alembic_version": "SELECT count(*) FROM alembic_version",
    }
    with pytest.raises(DBAPIError, match="permission denied"):
        await conn.execute(text(statements[table]))


async def test_tools_login_passes_the_privilege_probe(conn: AsyncConnection) -> None:
    await verify_read_only_privileges(conn)


@pytest.fixture
async def owner() -> AsyncIterator[AsyncEngine]:
    """The schema owner's engine; skips when DATABASE_URL is the placeholder conftest sets outside Docker."""
    engine = create_async_engine(get_settings().database_url.get_secret_value(), poolclass=pool.NullPool)
    try:
        async with asyncio.timeout(3), engine.connect():
            pass
    except Exception:  # noqa: BLE001 — unreachable or wrong credentials both mean "no owner login here"
        await engine.dispose()
        pytest.skip("owner login unavailable (run inside docker compose)")
    yield engine
    await engine.dispose()


async def test_owner_login_fails_the_privilege_probe(owner: AsyncEngine) -> None:
    async with owner.connect() as connection:
        with pytest.raises(ReadOnlyViolationError, match="INSERT on claims"):
            await verify_read_only_privileges(connection)


async def test_injection_shaped_value_is_just_data(conn: AsyncConnection) -> None:
    # Bypassing the tool's format check on purpose: even then, the value is a bound parameter, not SQL.
    assert await fetch_policy(conn, "' OR '1'='1") is None


# ---- tools on the labelled seed edge cases --------------------------------------------------------------------


async def test_policy_status_reports_the_r02_lapse(conn: AsyncConnection, tools: dict[str, object]) -> None:
    _, policy_number, incident = await _edge_claim(conn, 2)
    message = await _run(tools, "get_policy_status", policy_number=policy_number, on_date=incident.isoformat())
    assert message.artifact.status == "lapsed" and message.artifact.lapsed_on < incident
    assert message.artifact.in_force_on_date is False
    payments = await _run(tools, "get_payment_history", policy_number=policy_number, on_date=incident.isoformat())
    assert payments.artifact.missed == 1 and payments.artifact.missed_on_or_before_date == 1


async def test_claim_history_shows_the_r04_frequency(conn: AsyncConnection, tools: dict[str, object]) -> None:
    claim_number, policy_number, incident = await _edge_claim(conn, 4)
    message = await _run(
        tools, "get_claim_history",
        policy_number=policy_number, exclude_claim_number=claim_number, reference_date=incident.isoformat(),
    )  # fmt: skip
    assert message.artifact.other_claims_in_prior_12_months == 3
    assert claim_number not in {claim.claim_number for claim in message.artifact.claims}


async def test_similar_claims_finds_the_r07_duplicate(conn: AsyncConnection, tools: dict[str, object]) -> None:
    claim_number, _, _ = await _edge_claim(conn, 7)
    message = await _run(tools, "check_similar_claims", claim_number=claim_number)
    assert message.artifact.exact_duplicates == 1
    assert MatchReason.EXACT_DUPLICATE in message.artifact.matches[0].match_reasons


async def test_customer_profile_shows_the_r08_pending_kyc(conn: AsyncConnection, tools: dict[str, object]) -> None:
    _, policy_number, _ = await _edge_claim(conn, 8)
    message = await _run(tools, "get_customer_profile", policy_number=policy_number)
    assert message.artifact.kyc_status == "pending"
    assert "full_name" not in message.content and "email" not in message.content


async def test_unknown_policy_is_reported_back_to_the_model(tools: dict[str, object]) -> None:
    message = await _run(tools, "get_policy_status", policy_number="MOT-1999-000000")
    assert message.status == "error" and "No policy MOT-1999-000000 exists" in message.content


async def test_the_database_wrapper_refuses_a_writable_login(owner: AsyncEngine) -> None:
    with pytest.raises(ReadOnlyViolationError):
        async with ReadOnlyDatabase(owner).connect():
            pass

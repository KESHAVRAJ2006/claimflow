"""Integration tests proving the database itself enforces key invariants.

These need a migrated Postgres, so they run inside docker compose and are skipped elsewhere.
Every test runs inside a transaction that is rolled back, so they never leave data behind.
"""

import asyncio
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import pool, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from app.core.config import get_settings

pytestmark = pytest.mark.integration


@pytest.fixture
async def conn() -> AsyncIterator[AsyncConnection]:
    engine = create_async_engine(get_settings().database_url.get_secret_value(), poolclass=pool.NullPool)
    try:
        connection = await asyncio.wait_for(engine.connect(), timeout=3)
    except Exception:  # noqa: BLE001 — any connection failure means "no database here"
        await engine.dispose()
        pytest.skip("Postgres is not reachable (run inside docker compose)")
    if await connection.scalar(text("SELECT to_regclass('public.audit_log')")) is None:
        await connection.close()
        await engine.dispose()
        pytest.skip("Database is not migrated (run `alembic upgrade head`)")

    # The query above auto-began a transaction; every statement in the test joins it, and the rollback
    # below discards all of it, including the transaction left aborted by an expected database error.
    try:
        yield connection
    finally:
        await connection.rollback()
        await connection.close()
        await engine.dispose()


async def _insert_audit_row(conn: AsyncConnection) -> int:
    return await conn.scalar(
        text("INSERT INTO audit_log (actor, action) VALUES (:actor, :action) RETURNING id"),
        {"actor": "test", "action": "test.inserted"},
    )


async def _insert_customer(conn: AsyncConnection) -> object:
    return await conn.scalar(
        text(
            "INSERT INTO customers (customer_ref, full_name, email, phone, date_of_birth, city, kyc_status) "
            "VALUES ('CUS-TEST1', 'Test User', 'constraint.test@example.com', '+91 900', '1990-01-01', "
            "'Pune', 'verified') RETURNING id"
        )
    )


async def test_audit_log_rejects_update(conn: AsyncConnection) -> None:
    row_id = await _insert_audit_row(conn)
    with pytest.raises(DBAPIError, match="append-only"):
        await conn.execute(text("UPDATE audit_log SET actor = :actor WHERE id = :id"), {"actor": "x", "id": row_id})


async def test_audit_log_rejects_delete(conn: AsyncConnection) -> None:
    row_id = await _insert_audit_row(conn)
    with pytest.raises(DBAPIError, match="append-only"):
        await conn.execute(text("DELETE FROM audit_log WHERE id = :id"), {"id": row_id})


async def test_lapsed_policy_requires_lapse_date(conn: AsyncConnection) -> None:
    customer_id = await _insert_customer(conn)
    with pytest.raises(IntegrityError, match="lapsed_on_matches_status"):
        await conn.execute(
            text(
                "INSERT INTO policies (policy_number, customer_id, product_type, status, start_date, end_date, "
                "sum_insured, deductible, annual_premium, policy_document) VALUES ('MOT-TEST-1', :cid, 'motor', "
                "'lapsed', '2026-01-01', '2026-12-31', 500000, 5000, 12500, 'Motor_Policy.pdf')"
            ),
            {"cid": customer_id},
        )


async def test_unknown_enum_value_is_rejected(conn: AsyncConnection) -> None:
    with pytest.raises(IntegrityError, match="kyc_status"):
        await conn.execute(
            text(
                "INSERT INTO customers (customer_ref, full_name, email, phone, date_of_birth, city, kyc_status) "
                "VALUES ('CUS-TEST2', 'Test', 'enum.test@example.com', '+91 900', '1990-01-01', 'Pune', 'maybe')"
            )
        )


async def test_timestamps_round_trip_with_timezone(conn: AsyncConnection) -> None:
    row_id = await _insert_audit_row(conn)
    created = await conn.scalar(text("SELECT created_at FROM audit_log WHERE id = :id"), {"id": row_id})
    assert created.tzinfo is not None

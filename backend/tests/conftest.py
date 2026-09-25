"""Shared pytest configuration."""

import asyncio
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

# Set before any `app` import so Settings validates without needing a .env file.
# setdefault keeps real values when the tests run inside the docker-compose backend container.
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
os.environ.setdefault("ENVIRONMENT", "test")
# API tests don't need retrieval; tests that do build their own retriever (see test_retrieval_eval.py).
os.environ["LOAD_EMBEDDING_MODEL"] = "false"
# Tests must never call a real LLM (cost, rate limits, non-determinism): blank keys override .env and compose.
os.environ["GROQ_API_KEY"] = ""
os.environ["GOOGLE_API_KEY"] = ""
# Nor post events to a real n8n: with notifications on in .env, the container has both set.
os.environ["N8N_WEBHOOK_URL"] = ""
os.environ["WEBHOOK_SECRET"] = ""

from alembic import command  # noqa: E402 — after the env setup above
from alembic.config import Config  # noqa: E402
from sqlalchemy import pool, text  # noqa: E402
from sqlalchemy.engine import make_url  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from scripts.provision_readonly_role import provision  # noqa: E402
from scripts.seed import build_dataset, write_dataset  # noqa: E402

ALEMBIC_INI = Path(__file__).resolve().parent.parent / "alembic.ini"


@pytest.fixture(autouse=True)
def _fresh_settings() -> Iterator[None]:
    """Clear the settings cache around each test so env changes made with monkeypatch take effect."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@dataclass(frozen=True)
class ScratchDatabase:
    """Connection URLs of the throwaway test database."""

    owner_url: str
    tools_url: str


async def _can_connect(url: str) -> bool:
    """Whether the owner login works here (it does inside docker compose, not with conftest's placeholder URL)."""
    engine = create_async_engine(url, poolclass=pool.NullPool)
    try:
        async with asyncio.timeout(3), engine.connect():
            return True
    except Exception:  # noqa: BLE001 — unreachable host and wrong credentials both mean "no database here"
        return False
    finally:
        await engine.dispose()


async def _recreate(admin_url: str, name: str) -> None:
    """Drop and create the test database. CREATE DATABASE cannot run inside a transaction, hence AUTOCOMMIT."""
    engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT", poolclass=pool.NullPool)
    try:
        async with engine.connect() as connection:
            for template in ("DROP DATABASE IF EXISTS %I WITH (FORCE)", "CREATE DATABASE %I"):
                # DDL takes no bind parameters; Postgres quotes the (validated) name itself via format(%I).
                # The cast is required: format()'s arguments are variadic "any", so Postgres can't infer the type.
                statement = await connection.scalar(
                    text("SELECT format(:t, CAST(:n AS text))"), {"t": template, "n": name}
                )
                await connection.exec_driver_sql(statement)
    finally:
        await engine.dispose()


async def _seed(url: str) -> None:
    engine = create_async_engine(url, poolclass=pool.NullPool)
    try:
        async with AsyncSession(engine) as session, session.begin():
            await write_dataset(session, build_dataset(now=datetime.now(UTC), seed=42))
    finally:
        await engine.dispose()


@pytest.fixture(scope="session")
def test_database() -> Iterator[ScratchDatabase]:
    """A fresh, migrated, provisioned and seeded ``<db>_test`` database, rebuilt once per test session.

    API tests write claims and append-only audit rows. Doing that in the development database would mark its seed
    data as "used" forever (scripts.seed never refreshes used data), so they get their own database instead.
    """
    settings = get_settings()
    if settings.tools_database_url is None:
        pytest.skip("TOOLS_DATABASE_URL is not set")
    owner, tools = (
        make_url(settings.database_url.get_secret_value()),
        make_url(settings.tools_database_url.get_secret_value()),
    )
    name = f"{owner.database}_test"
    if not re.fullmatch(r"[a-z_][a-z0-9_]{0,50}", name):
        pytest.skip(f"unexpected database name {name!r}")
    admin_url = owner.render_as_string(hide_password=False)
    if not asyncio.run(_can_connect(admin_url)):
        pytest.skip("owner login unavailable; run inside docker compose")
    # Anything that fails from here on is a real bug and must fail the run, not skip it.
    asyncio.run(_recreate(admin_url, name))
    owner_url = owner.set(database=name).render_as_string(hide_password=False)
    tools_url = tools.set(database=name).render_as_string(hide_password=False)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("DATABASE_URL", owner_url)  # migrations/env.py reads the URL from settings
        get_settings.cache_clear()
        command.upgrade(Config(str(ALEMBIC_INI)), "head")
    get_settings.cache_clear()
    asyncio.run(provision(owner_url, tools_url))  # grants are per database, so the test DB needs its own
    asyncio.run(_seed(owner_url))
    yield ScratchDatabase(owner_url, tools_url)

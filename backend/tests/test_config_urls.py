"""Deployment-facing configuration: provider database URLs, the derived tools URL, and the CREATEROLE failure."""

import asyncio

import pytest
from sqlalchemy import pool, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import Settings, to_asyncpg_url
from scripts.provision_readonly_role import ProvisioningError, check_configuration, provision
from tests.conftest import ScratchDatabase

OWNER = "postgresql+asyncpg://claimflow:owner-pass@db.internal:5432/claimflow"


def settings(**values: object) -> Settings:
    """Settings from explicit values only, ignoring .env files and the process environment."""
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("postgres://u:p@h:5432/d", "postgresql+asyncpg://u:p@h:5432/d"),
        ("postgresql://u:p@h/d", "postgresql+asyncpg://u:p@h/d"),
        ("postgresql://u:p@h/d?sslmode=require", "postgresql+asyncpg://u:p@h/d?ssl=require"),
        ("postgresql+asyncpg://u:p@h/d", "postgresql+asyncpg://u:p@h/d"),
        ("postgresql+psycopg2://u:p@h/d", "postgresql+psycopg2://u:p@h/d"),
    ],
)
def test_to_asyncpg_url(given: str, expected: str) -> None:
    assert to_asyncpg_url(given) == expected


def test_render_style_url_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOOLS_DATABASE_URL", raising=False)
    loaded = settings(database_url="postgres://claimflow:pw@dpg-abc-a:5432/claimflow")
    assert loaded.database_url.get_secret_value() == "postgresql+asyncpg://claimflow:pw@dpg-abc-a:5432/claimflow"


def test_a_sync_driver_is_still_rejected() -> None:
    with pytest.raises(ValueError, match="asyncpg"):
        settings(database_url="postgresql+psycopg2://u:p@h/d")


def test_tools_url_is_derived_from_the_owner_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOOLS_DATABASE_URL", raising=False)
    # A base64 password with the characters Render's generateValue produces.
    password = "a+b/c=d0123456789XYZ="
    loaded = settings(database_url=OWNER, agent_db_user="claimflow_agent", agent_db_password=password)
    assert loaded.tools_database_url is not None
    tools = make_url(loaded.tools_database_url.get_secret_value())
    assert (tools.username, tools.password) == ("claimflow_agent", password)
    assert (tools.host, tools.port, tools.database) == ("db.internal", 5432, "claimflow")
    assert tools.drivername == "postgresql+asyncpg"
    # The provisioning script accepts the derived URL, password characters included.
    assert check_configuration(OWNER, loaded.tools_database_url.get_secret_value()) == ("claimflow_agent", password)


def test_an_explicit_tools_url_wins() -> None:
    explicit = "postgresql+asyncpg://other_agent:pw-0123456789@h/claimflow"
    loaded = settings(
        database_url=OWNER, tools_database_url=explicit, agent_db_user="claimflow_agent", agent_db_password="x" * 20
    )
    assert loaded.tools_database_url is not None
    assert loaded.tools_database_url.get_secret_value() == explicit


def test_no_tools_url_without_both_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOOLS_DATABASE_URL", raising=False)
    loaded = settings(database_url=OWNER, agent_db_user="claimflow_agent", agent_db_password="")
    assert loaded.tools_database_url is None


# ---- a database owner without CREATEROLE (the situation on some managed Postgres plans) --------------------------

LIMITED_ROLE = "claimflow_test_limited_owner"
LIMITED_PASSWORD = "limited-owner-pass-0123"


async def _run(url: str, *statements: str) -> None:
    engine = create_async_engine(url, poolclass=pool.NullPool)
    try:
        async with engine.begin() as connection:
            for statement in statements:
                await connection.execute(text(statement))
    finally:
        await engine.dispose()


@pytest.mark.integration
def test_provisioning_explains_a_missing_createrole(test_database: ScratchDatabase) -> None:
    # Constant names, so the DDL needs no quoting. NOCREATEROLE is the default; spelled out for the reader.
    create = f"CREATE ROLE {LIMITED_ROLE} LOGIN NOCREATEROLE PASSWORD '{LIMITED_PASSWORD}'"
    drop = f"DROP ROLE IF EXISTS {LIMITED_ROLE}"
    asyncio.run(_run(test_database.owner_url, drop, create))
    try:
        limited_url = (
            make_url(test_database.owner_url)
            .set(username=LIMITED_ROLE, password=LIMITED_PASSWORD)
            .render_as_string(hide_password=False)
        )
        with pytest.raises(ProvisioningError, match="may not create roles or grant access"):
            asyncio.run(provision(limited_url, test_database.tools_url))
    finally:
        asyncio.run(_run(test_database.owner_url, drop))

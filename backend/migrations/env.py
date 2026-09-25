"""Alembic environment: runs migrations over the same async driver the app uses."""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

import app.db.models  # noqa: F401 — importing registers every table on Base.metadata for autogenerate
from app.core.config import get_settings
from app.db.base import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def _database_url() -> str:
    """Read the database URL from application settings.

    Returns:
        The asyncpg connection URL.
    """
    return get_settings().database_url.get_secret_value()


def run_migrations_offline() -> None:
    """Emit migration SQL to stdout without connecting (``alembic upgrade head --sql``)."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _run_sync_migrations(connection: Connection) -> None:
    """Run migrations on a connection that Alembic's synchronous API can drive.

    Args:
        connection: A sync-facade connection provided by ``AsyncConnection.run_sync``.
    """
    # compare_type makes autogenerate notice column type changes (e.g. String(50) -> String(80)).
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Connect with asyncpg and apply migrations."""
    # NullPool: a migration is a one-shot process, so there is no reason to keep pooled connections alive.
    engine = create_async_engine(_database_url(), poolclass=pool.NullPool)
    async with engine.connect() as connection:
        await connection.run_sync(_run_sync_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())

"""Async SQLAlchemy engine and session factories."""

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings


def create_db_engine(settings: Settings) -> AsyncEngine:
    """Create the application's async connection pool.

    Creating the engine does not connect; the first connection opens on first use.

    Args:
        settings: Application settings containing ``database_url``.

    Returns:
        A configured AsyncEngine. Call ``await engine.dispose()`` on shutdown.
    """
    return create_async_engine(
        settings.database_url.get_secret_value(),
        # Tests each pooled connection before use, so a Postgres restart doesn't surface as request errors.
        pool_pre_ping=True,
        # 5 + 10 overflow suits a single Uvicorn worker; Render's small Postgres plans cap near 25 connections.
        pool_size=5,
        max_overflow=10,
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create a factory for short-lived database sessions.

    Args:
        engine: The engine sessions will borrow connections from.

    Returns:
        A sessionmaker producing AsyncSession objects.
    """
    # expire_on_commit=False: after commit, reading an attribute would otherwise trigger a lazy refresh,
    # which async SQLAlchemy cannot do implicitly and would raise MissingGreenlet.
    return async_sessionmaker(engine, expire_on_commit=False)

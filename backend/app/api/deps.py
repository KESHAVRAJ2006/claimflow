"""FastAPI dependencies that hand shared clients to route handlers."""

from collections.abc import AsyncIterator

from fastapi import Request
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.retrieval.retriever import PolicyRetriever
from app.services.events import EventBroker
from app.services.triage import TriageRunner
from app.services.webhooks import WebhookSender


def get_db_engine(request: Request) -> AsyncEngine:
    """Return the database engine created during application startup.

    Args:
        request: The incoming request, used to reach ``app.state``.

    Returns:
        The shared async SQLAlchemy engine.
    """
    return request.app.state.db_engine


async def get_db_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Yield a database session scoped to one request.

    Routes commit explicitly; anything uncommitted is rolled back when the session closes.

    Args:
        request: The incoming request, used to reach ``app.state``.

    Yields:
        An AsyncSession that is closed after the response is sent.
    """
    async with request.app.state.session_factory() as session:
        yield session


def get_qdrant(request: Request) -> AsyncQdrantClient:
    """Return the Qdrant client created during application startup.

    Args:
        request: The incoming request, used to reach ``app.state``.

    Returns:
        The shared async Qdrant client.
    """
    return request.app.state.qdrant


def get_session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    """Return the session factory, for handlers that must control a session's lifetime (SSE).

    Args:
        request: The incoming request.

    Returns:
        The shared session factory.
    """
    return request.app.state.session_factory


def get_event_broker(request: Request) -> EventBroker:
    """Return the run event broker.

    Args:
        request: The incoming request.

    Returns:
        The broker created at startup.
    """
    return request.app.state.event_broker


def get_triage_runner(request: Request) -> TriageRunner | None:
    """Return the triage runner, or None when agents are not configured.

    Args:
        request: The incoming request.

    Returns:
        The runner, or None.
    """
    return request.app.state.triage_runner


def get_triage_disabled_reason(request: Request) -> str:
    """Explain why triage is off, for the 503 answer.

    Args:
        request: The incoming request.

    Returns:
        The reason recorded at startup, or a generic pointer to the startup log.
    """
    reason = getattr(request.app.state, "triage_disabled_reason", None)
    return reason or "Triage is not configured in this API process; see the startup log."


def get_webhooks(request: Request) -> WebhookSender:
    """Return the webhook sender.

    Args:
        request: The incoming request.

    Returns:
        The sender created at startup.
    """
    return request.app.state.webhooks


def get_policy_retriever(request: Request) -> PolicyRetriever | None:
    """Return the policy retriever, or None when the embedding model is not loaded.

    Args:
        request: The incoming request.

    Returns:
        The retriever, or None.
    """
    return request.app.state.policy_retriever

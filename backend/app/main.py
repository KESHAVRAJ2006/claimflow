"""FastAPI application factory and entrypoint (``uvicorn app.main:app``)."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.body_limit import MULTIPART_OVERHEAD_BYTES, BodySizeLimitMiddleware
from app.api.middleware import RequestContextMiddleware
from app.api.problems import install_problem_handlers
from app.api.routes import claims, health, metrics, policies
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.readonly import ReadOnlyDatabase, create_readonly_engine
from app.db.session import create_db_engine, create_session_factory
from app.db.vector import create_qdrant_client
from app.retrieval.embeddings import get_embedder
from app.retrieval.retriever import PolicyRetriever
from app.services.claims import fail_interrupted_runs
from app.services.events import EventBroker
from app.services.triage import TriageDisabledError, build_triage_runner
from app.services.webhooks import HttpWebhookSender

logger = logging.getLogger("claimflow")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create shared clients once at startup and release them at shutdown.

    Expensive resources (connection pools, the embedding model, the compiled graph) live here, never per request.
    The exit stack releases them in reverse order, and even if startup fails halfway.

    Args:
        app: The FastAPI application whose ``state`` holds the clients.

    Yields:
        Control back to FastAPI while the app serves requests.
    """
    settings = get_settings()
    async with AsyncExitStack() as stack:
        app.state.db_engine = create_db_engine(settings)
        stack.push_async_callback(app.state.db_engine.dispose)
        app.state.session_factory = create_session_factory(app.state.db_engine)
        app.state.qdrant = create_qdrant_client(settings)
        stack.push_async_callback(app.state.qdrant.close)
        # The agent tools' own read-only pool. Creating it does not connect; privileges are checked on first use.
        app.state.tools_database = (
            ReadOnlyDatabase(create_readonly_engine(settings)) if settings.tools_database_url is not None else None
        )
        if app.state.tools_database is not None:
            stack.push_async_callback(app.state.tools_database.dispose)
        app.state.policy_retriever = None
        if settings.load_embedding_model:
            # Loaded in a worker thread: it takes seconds and would otherwise block the event loop during startup.
            embedder = await asyncio.to_thread(get_embedder, settings.embedding_model_name)
            app.state.policy_retriever = PolicyRetriever(app.state.qdrant, embedder, settings.qdrant_collection)

        app.state.event_broker = EventBroker()
        secret = settings.webhook_secret.get_secret_value() if settings.webhook_secret else None
        app.state.webhooks = HttpWebhookSender(settings.n8n_webhook_url, secret)
        stack.push_async_callback(app.state.webhooks.aclose)
        app.state.triage_runner, app.state.triage_disabled_reason = None, None
        try:
            app.state.triage_runner = await build_triage_runner(
                settings, stack, session_factory=app.state.session_factory, tools_database=app.state.tools_database,
                retriever=app.state.policy_retriever, broker=app.state.event_broker, webhooks=app.state.webhooks,
            )  # fmt: skip
        except TriageDisabledError as error:
            # The API still serves claims and decisions; "Run triage" answers 503 with this reason.
            app.state.triage_disabled_reason = str(error)
            logger.warning("triage disabled", extra={"reason": str(error)})
        if app.state.triage_runner is not None:
            # Registered last, so it runs first on shutdown: runs are cancelled while the database is still open.
            stack.push_async_callback(app.state.triage_runner.shutdown)
        await _recover_interrupted_runs(app)
        logger.info(
            "startup complete",
            extra={
                "environment": settings.environment,
                "version": settings.app_version,
                "embedding_model_loaded": settings.load_embedding_model,
                "triage_enabled": app.state.triage_runner is not None,
            },
        )
        yield
    logger.info("shutdown complete")


async def _recover_interrupted_runs(app: FastAPI) -> None:
    """Mark claims left PROCESSING by a previous process as FAILED; never blocks startup if Postgres is down.

    Args:
        app: The application.
    """
    try:
        async with app.state.session_factory() as session:
            count = await fail_interrupted_runs(session)
    except Exception as error:  # noqa: BLE001 — the health endpoint reports a down database; startup continues
        logger.warning("could not check for interrupted runs", extra={"error": repr(error)})
        return
    if count:
        logger.warning("marked interrupted triage runs as failed", extra={"count": count})


def create_app() -> FastAPI:
    """Build the FastAPI application.

    Returns:
        The configured application with middleware, error handlers and routers attached.
    """
    settings = get_settings()
    # Configured here (after Uvicorn has set up its own logging) so our JSON handler replaces Uvicorn's.
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
        description="Decisions are AI-assisted recommendations that require human review, not automated "
        "determinations.",
        # Everything lives under /api so the Next.js app can own every other path on the same domain.
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
    )
    # Starlette runs the LAST-added middleware first. Outermost first: request id + logging (so every response,
    # even a 413, carries a request id), then CORS, then the body size limit closest to the routes.
    app.add_middleware(BodySizeLimitMiddleware, max_body_bytes=settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type", "Last-Event-ID", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )
    app.add_middleware(RequestContextMiddleware)
    install_problem_handlers(app)
    for router in (health.router, claims.router, policies.router, metrics.router):
        app.include_router(router, prefix="/api")
    return app


app = create_app()

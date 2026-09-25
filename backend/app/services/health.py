"""Connectivity probes for Postgres and Qdrant."""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from qdrant_client import AsyncQdrantClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.config import Settings
from app.schemas.health import ComponentHealth, HealthReport

logger = logging.getLogger(__name__)


def _elapsed_ms(start: float) -> float:
    """Milliseconds elapsed since a ``time.perf_counter()`` reading.

    Args:
        start: The earlier perf_counter value.

    Returns:
        Elapsed time rounded to 0.01 ms.
    """
    return round((time.perf_counter() - start) * 1000, 2)


async def timed_check(
    name: str,
    probe: Callable[[], Awaitable[object]],
    timeout_s: float,
    expose_errors: bool,
) -> ComponentHealth:
    """Run one probe under a timeout and convert the outcome into a ComponentHealth.

    Never raises: a health endpoint that crashes when a dependency is down reports nothing useful.

    Args:
        name: Dependency name, used in log lines.
        probe: Zero-argument coroutine function that raises if the dependency is unreachable.
        timeout_s: Seconds to wait before declaring the dependency down.
        expose_errors: Include the exception message in the result. Disable in production, where messages
            can reveal internal hostnames and ports to anyone who calls the public endpoint.

    Returns:
        ``status="up"`` if the probe finished in time, otherwise ``status="down"`` with a reason.
    """
    start = time.perf_counter()
    try:
        await asyncio.wait_for(probe(), timeout=timeout_s)
    except TimeoutError:
        error = f"timed out after {timeout_s}s"
    except Exception as exc:  # noqa: BLE001 — any failure means "down"; the full error is logged below
        error = f"{type(exc).__name__}: {exc}" if expose_errors else type(exc).__name__
        logger.warning("health probe failed", extra={"component": name, "error": repr(exc)})
    else:
        return ComponentHealth(status="up", latency_ms=_elapsed_ms(start))
    return ComponentHealth(status="down", latency_ms=_elapsed_ms(start), error=error[:300])


async def probe_postgres(engine: AsyncEngine) -> None:
    """Open a pooled connection and run ``SELECT 1``.

    Args:
        engine: The async SQLAlchemy engine.

    Raises:
        Exception: Any driver or network error when Postgres is unreachable.
    """
    # A real round-trip query, not just a TCP connect, proves auth and the database itself work.
    async with engine.connect() as connection:
        await connection.execute(text("SELECT 1"))


async def probe_qdrant(client: AsyncQdrantClient) -> None:
    """List collections, which requires a working HTTP round-trip to Qdrant.

    Args:
        client: The async Qdrant client.

    Raises:
        Exception: Any HTTP or network error when Qdrant is unreachable.
    """
    await client.get_collections()


async def run_health_checks(engine: AsyncEngine, qdrant: AsyncQdrantClient, settings: Settings) -> HealthReport:
    """Probe every dependency concurrently and build the health report.

    Args:
        engine: The async SQLAlchemy engine.
        qdrant: The async Qdrant client.
        settings: Application settings (timeout, environment, version).

    Returns:
        A HealthReport whose status is ``"ok"`` only if every component is up.
    """
    expose_errors = not settings.is_production
    timeout = settings.health_check_timeout_s
    # gather runs both probes at once, so worst-case latency is one timeout rather than the sum.
    postgres, qdrant_health = await asyncio.gather(
        timed_check("postgres", lambda: probe_postgres(engine), timeout, expose_errors),
        timed_check("qdrant", lambda: probe_qdrant(qdrant), timeout, expose_errors),
    )
    checks = {"postgres": postgres, "qdrant": qdrant_health}
    overall = "ok" if all(check.status == "up" for check in checks.values()) else "degraded"
    return HealthReport(
        status=overall,
        version=settings.app_version,
        environment=settings.environment,
        timestamp=datetime.now(UTC),
        checks=checks,
    )

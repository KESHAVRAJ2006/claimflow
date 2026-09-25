"""Tests for the health probes, using fake probes so no real services are required."""

import asyncio

import pytest

from app.core.config import Settings
from app.services import health as health_service


async def _ok() -> None:
    return None


async def _refused() -> None:
    raise ConnectionRefusedError("connect to 10.0.0.5:5432 refused")


async def _hangs() -> None:
    await asyncio.sleep(10)


async def test_timed_check_up() -> None:
    result = await health_service.timed_check("x", _ok, timeout_s=1, expose_errors=True)
    assert result.status == "up"
    assert result.error is None
    assert result.latency_ms >= 0


async def test_timed_check_down_includes_message_outside_production() -> None:
    result = await health_service.timed_check("x", _refused, timeout_s=1, expose_errors=True)
    assert result.status == "down"
    assert result.error is not None and "10.0.0.5" in result.error


async def test_timed_check_hides_internal_details_in_production() -> None:
    result = await health_service.timed_check("x", _refused, timeout_s=1, expose_errors=False)
    assert result.error == "ConnectionRefusedError"


async def test_timed_check_times_out_instead_of_hanging() -> None:
    result = await health_service.timed_check("x", _hangs, timeout_s=0.05, expose_errors=True)
    assert result.status == "down"
    assert result.error is not None and "timed out" in result.error
    assert result.latency_ms < 1000


async def test_report_is_degraded_when_any_component_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_postgres(engine: object) -> None:
        return None

    async def fake_qdrant(client: object) -> None:
        raise ConnectionError("qdrant down")

    monkeypatch.setattr(health_service, "probe_postgres", fake_postgres)
    monkeypatch.setattr(health_service, "probe_qdrant", fake_qdrant)

    report = await health_service.run_health_checks(engine=None, qdrant=None, settings=Settings())  # type: ignore[arg-type]

    assert report.status == "degraded"
    assert report.checks["postgres"].status == "up"
    assert report.checks["qdrant"].status == "down"
    assert report.timestamp.tzinfo is not None  # timestamps must be timezone-aware

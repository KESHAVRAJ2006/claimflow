"""Tests for GET /api/health, with the probes replaced so results are deterministic."""

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from app.api.routes import health as health_route
from app.main import create_app
from app.schemas.health import ComponentHealth, HealthReport


def _report(qdrant_status: str) -> HealthReport:
    return HealthReport(
        status="ok" if qdrant_status == "up" else "degraded",
        version="0.1.0",
        environment="test",
        timestamp=datetime.now(UTC),
        checks={
            "postgres": ComponentHealth(status="up", latency_ms=1.0),
            "qdrant": ComponentHealth(status=qdrant_status, latency_ms=1.0),  # type: ignore[arg-type]
        },
    )


@pytest.fixture
def client() -> TestClient:
    # `with` runs the lifespan, exercising real engine/client creation and shutdown.
    with TestClient(create_app()) as test_client:
        yield test_client


def test_health_returns_200_when_all_up(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake(**_: object) -> HealthReport:
        return _report("up")

    monkeypatch.setattr(health_route, "run_health_checks", fake)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_returns_503_when_degraded(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake(**_: object) -> HealthReport:
        return _report("down")

    monkeypatch.setattr(health_route, "run_health_checks", fake)
    response = client.get("/api/health")
    assert response.status_code == 503
    assert response.json()["checks"]["qdrant"]["status"] == "down"


def test_request_id_is_echoed_or_generated(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake(**_: object) -> HealthReport:
        return _report("up")

    monkeypatch.setattr(health_route, "run_health_checks", fake)
    assert client.get("/api/health", headers={"X-Request-ID": "abc-123"}).headers["x-request-id"] == "abc-123"
    # An unsafe ID (newline = log injection attempt) is replaced with a generated one.
    generated = client.get("/api/health", headers={"X-Request-ID": "bad\nid"}).headers["x-request-id"]
    assert generated != "bad\nid" and len(generated) == 32

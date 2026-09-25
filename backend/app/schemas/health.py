"""Response models for the health endpoint."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ComponentHealth(BaseModel):
    """Result of probing one external dependency."""

    status: Literal["up", "down"]
    latency_ms: float = Field(ge=0, description="Wall-clock time the probe took")
    error: str | None = Field(default=None, description="Failure reason; only the exception type in production")


class HealthReport(BaseModel):
    """Overall service health returned by GET /api/health."""

    status: Literal["ok", "degraded"]
    version: str
    environment: str
    timestamp: datetime
    checks: dict[str, ComponentHealth]

"""Response model for GET /api/metrics."""

from datetime import date, datetime

from pydantic import BaseModel, Field


class VolumePoint(BaseModel):
    """Claims per day, for the dashboard chart."""

    date: date
    submitted: int
    auto_decided: int = Field(description="Triaged with an approve or reject recommendation")
    escalated: int


class Metrics(BaseModel):
    """Operational metrics over a trailing window. Rates are fractions 0-1, or null when nothing was triaged."""

    window_days: int
    generated_at: datetime
    claims_today: int
    claims_in_window: int
    triaged_in_window: int
    pending_review: int
    auto_decision_rate: float | None
    escalation_rate: float | None
    override_rate: float | None = Field(description="Human decisions that differed from the AI recommendation")
    avg_tool_calls_per_claim: float | None
    latency_ms_p50: float | None = Field(description="End-to-end triage time per claim (sum of node latencies)")
    latency_ms_p95: float | None
    volume: list[VolumePoint]

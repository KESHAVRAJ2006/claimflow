"""Operational metrics for the dashboard, computed in SQL over a trailing window.

Aggregation happens in Postgres (counts, JSONB array lengths, percentile_cont), so the API returns a few numbers
instead of pulling every claim_runs row into Python.
"""

from datetime import UTC, date, datetime, time, timedelta

from sqlalchemy import Float, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AuditLog, Claim, ClaimRun
from app.domain.enums import ClaimStatus, RecommendedOutcome
from app.schemas.metrics import Metrics, VolumePoint

AUTO = (RecommendedOutcome.APPROVE, RecommendedOutcome.REJECT)


def _rate(part: int, whole: int) -> float | None:
    """A fraction rounded to 4 places, or None when there is nothing to divide by (not a misleading 0)."""
    return round(part / whole, 4) if whole else None


async def compute_metrics(session: AsyncSession, window_days: int, now: datetime | None = None) -> Metrics:
    """Compute dashboard metrics.

    Args:
        session: Database session.
        window_days: Trailing window length in days (the volume chart has one point per day).
        now: Current time; injected by tests.

    Returns:
        The metrics.
    """
    now = now or datetime.now(UTC)
    today = now.date()
    start_day = today - timedelta(days=window_days - 1)
    window_start = datetime.combine(start_day, time.min, UTC)
    in_window = Claim.created_at >= window_start

    counts = (
        await session.execute(
            select(
                func.count().filter(Claim.created_at >= datetime.combine(today, time.min, UTC)),
                func.count().filter(in_window),
                func.count().filter(in_window, Claim.recommended_outcome.is_not(None)),
                func.count().filter(in_window, Claim.recommended_outcome.in_(AUTO)),
                func.count().filter(in_window, Claim.recommended_outcome == RecommendedOutcome.ESCALATE),
                func.count().filter(Claim.status.in_((ClaimStatus.AWAITING_REVIEW, ClaimStatus.ESCALATED))),
            )
        )
    ).one()
    claims_today, claims_in_window, triaged, auto, escalated, pending = counts

    decided, overridden = (
        await session.execute(
            select(
                func.count().filter(AuditLog.action.in_(("claim.decided", "claim.overridden"))),
                func.count().filter(AuditLog.action == "claim.overridden"),
            ).where(AuditLog.created_at >= window_start)
        )
    ).one()

    # One row per (claim, run): its tool calls and its end-to-end latency (sum of node latencies).
    per_run = (
        select(
            func.coalesce(
                func.sum(func.jsonb_array_length(ClaimRun.output["tool_calls"])).filter(
                    ClaimRun.agent_name == "investigator"
                ),
                0,
            ).label("tool_calls"),
            func.sum(ClaimRun.latency_ms).label("latency_ms"),
        )
        .where(ClaimRun.created_at >= window_start)
        .group_by(ClaimRun.claim_id, ClaimRun.input["run_id"].astext)
        .subquery()
    )
    avg_tools, p50, p95 = (
        await session.execute(
            select(
                func.avg(cast(per_run.c.tool_calls, Float)),
                func.percentile_cont(0.5).within_group(per_run.c.latency_ms),
                func.percentile_cont(0.95).within_group(per_run.c.latency_ms),
            )
        )
    ).one()

    day = func.date(func.timezone("UTC", Claim.created_at)).label("day")
    daily = await session.execute(
        select(
            day,
            func.count(),
            func.count().filter(Claim.recommended_outcome.in_(AUTO)),
            func.count().filter(Claim.recommended_outcome == RecommendedOutcome.ESCALATE),
        )
        .where(in_window)
        .group_by(day)
    )
    by_day: dict[date, tuple[int, int, int]] = {row[0]: (row[1], row[2], row[3]) for row in daily}
    volume = [
        VolumePoint(date=d, submitted=by_day.get(d, (0, 0, 0))[0], auto_decided=by_day.get(d, (0, 0, 0))[1],
                    escalated=by_day.get(d, (0, 0, 0))[2])
        for d in (start_day + timedelta(days=offset) for offset in range(window_days))
    ]  # fmt: skip

    return Metrics(
        window_days=window_days,
        generated_at=now,
        claims_today=claims_today,
        claims_in_window=claims_in_window,
        triaged_in_window=triaged,
        pending_review=pending,
        auto_decision_rate=_rate(auto, triaged),
        escalation_rate=_rate(escalated, triaged),
        override_rate=_rate(overridden, decided),
        avg_tool_calls_per_claim=round(float(avg_tools), 2) if avg_tools is not None else None,
        latency_ms_p50=round(float(p50), 1) if p50 is not None else None,
        latency_ms_p95=round(float(p95), 1) if p95 is not None else None,
        volume=volume,
    )

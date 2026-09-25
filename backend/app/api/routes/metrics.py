"""GET /api/metrics: dashboard numbers."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db_session
from app.api.security import require_api_key
from app.schemas.metrics import Metrics
from app.services.metrics import compute_metrics

router = APIRouter(tags=["metrics"], dependencies=[Depends(require_api_key)])


@router.get("/metrics", response_model=Metrics, summary="Volume, auto-decision and escalation rates, tool use, latency")
async def metrics(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    days: Annotated[int, Query(ge=1, le=365, description="Trailing window in days")] = 30,
) -> Metrics:
    """Operational metrics over the last ``days`` days.

    Args:
        session: Database session.
        days: Window length.

    Returns:
        The metrics.
    """
    return await compute_metrics(session, days)

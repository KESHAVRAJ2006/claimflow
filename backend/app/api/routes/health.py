"""GET /api/health — dependency connectivity report."""

from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncEngine

from app.api.deps import get_db_engine, get_qdrant
from app.core.config import Settings, get_settings
from app.schemas.health import HealthReport
from app.services.health import run_health_checks

router = APIRouter(tags=["system"])


@router.get(
    "/health",
    response_model=HealthReport,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": HealthReport, "description": "A dependency is down"}},
)
async def health(
    response: Response,
    engine: Annotated[AsyncEngine, Depends(get_db_engine)],
    qdrant: Annotated[AsyncQdrantClient, Depends(get_qdrant)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthReport:
    """Probe Postgres and Qdrant and report their status.

    Args:
        response: Outgoing response, used to set the status code.
        engine: Shared database engine.
        qdrant: Shared Qdrant client.
        settings: Application settings.

    Returns:
        The health report; HTTP 200 when every dependency is up, otherwise 503.
    """
    report = await run_health_checks(engine=engine, qdrant=qdrant, settings=settings)
    if report.status != "ok":
        # 503 (not 200 with a "degraded" body) so load balancers and Docker actually stop routing to us.
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return report

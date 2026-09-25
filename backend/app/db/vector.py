"""Qdrant client factory."""

import math

from qdrant_client import AsyncQdrantClient

from app.core.config import Settings


def create_qdrant_client(settings: Settings) -> AsyncQdrantClient:
    """Create the shared async Qdrant client.

    Args:
        settings: Application settings containing the Qdrant URL and optional API key.

    Returns:
        An AsyncQdrantClient. Call ``await client.close()`` on shutdown.
    """
    api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    return AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=api_key,
        # Client-side timeout just above the health timeout, so asyncio.wait_for reports the failure first.
        timeout=math.ceil(settings.health_check_timeout_s) + 1,
    )

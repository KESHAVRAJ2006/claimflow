"""MCP resources against the seeded Postgres, read through the SELECT-only login."""

import asyncio
import json
from collections.abc import AsyncIterator

import pytest
from fastmcp import Client

from app.core.config import get_settings
from app.db.readonly import ReadOnlyDatabase, create_readonly_engine
from app.mcp.resources import PENDING_STATUSES, fetch_pending_claims, fetch_policy_list
from app.mcp.server import PENDING_CLAIMS_URI, POLICIES_URI, build_mcp_server

pytestmark = pytest.mark.integration

# Customer columns and free text that must never appear in a resource.
PRIVATE_KEYS = {"full_name", "email", "phone", "date_of_birth", "city", "description", "decision_rationale"}


@pytest.fixture
async def database() -> AsyncIterator[ReadOnlyDatabase]:
    settings = get_settings()
    if settings.tools_database_url is None:
        pytest.skip("TOOLS_DATABASE_URL is not set")
    db = ReadOnlyDatabase(create_readonly_engine(settings))
    try:
        async with asyncio.timeout(3):
            await fetch_policy_list(db, limit=1)
    except Exception as error:  # noqa: BLE001 — unreachable, unprovisioned or unseeded all mean "skip"
        await db.dispose()
        pytest.skip(f"seeded database unavailable ({type(error).__name__})")
    yield db
    await db.dispose()


async def test_resources_serve_json_without_private_fields(database: ReadOnlyDatabase) -> None:
    async with Client(build_mcp_server([], database)) as client:
        policies = json.loads((await client.read_resource(POLICIES_URI))[0].text)
        pending = json.loads((await client.read_resource(PENDING_CLAIMS_URI))[0].text)
    assert policies["total"] >= len(policies["policies"]) > 0
    for item in [*policies["policies"], *pending["claims"]]:
        assert not PRIVATE_KEYS & item.keys()
    # Money stays exact: a JSON string, never a float.
    assert isinstance(policies["policies"][0]["sum_insured"], str)
    assert "not a decision" in pending["note"]


async def test_pending_claims_are_undecided_and_oldest_first(database: ReadOnlyDatabase) -> None:
    snapshot = await fetch_pending_claims(database)
    assert snapshot.total >= len(snapshot.claims)
    assert {claim.status for claim in snapshot.claims} <= set(PENDING_STATUSES)
    times = [claim.submitted_at for claim in snapshot.claims]
    assert times == sorted(times)


async def test_limit_marks_the_snapshot_truncated(database: ReadOnlyDatabase) -> None:
    snapshot = await fetch_policy_list(database, limit=1)
    assert len(snapshot.policies) == 1
    assert snapshot.truncated is (snapshot.total > 1)

"""Graph input loaders and the acceptance script's claim picker, against the seeded Postgres (read-only login)."""

import asyncio
from collections.abc import AsyncIterator

import pytest
from qdrant_client import AsyncQdrantClient

from app.agents.investigator import INVESTIGATOR_TOOL_NAMES
from app.agents.loaders import ClaimNotFoundError, build_investigator_tools, load_claim_input, load_rule_context
from app.core.config import get_settings
from app.db.readonly import ReadOnlyDatabase, create_readonly_engine
from app.domain.enums import ProductType
from app.retrieval.retriever import PolicyRetriever
from app.rules import evaluate_all
from scripts.compare_investigations import find_edge_claim, pick_simple_claim
from tests.fakes import FakeEmbedder

pytestmark = pytest.mark.integration


@pytest.fixture
async def database() -> AsyncIterator[ReadOnlyDatabase]:
    settings = get_settings()
    if settings.tools_database_url is None:
        pytest.skip("TOOLS_DATABASE_URL is not set")
    db = ReadOnlyDatabase(create_readonly_engine(settings))
    try:
        async with asyncio.timeout(3):
            await find_edge_claim(db, 2)
    except Exception as error:  # noqa: BLE001 — unreachable, unprovisioned or unseeded all mean "skip"
        await db.dispose()
        pytest.skip(f"seeded database unavailable ({type(error).__name__})")
    yield db
    await db.dispose()


async def test_claim_input_and_rule_context_for_the_r02_edge_case(database: ReadOnlyDatabase) -> None:
    number = await find_edge_claim(database, 2)
    claim = await load_claim_input(database, number)
    assert claim.claim_number == number and claim.product_type is ProductType.HEALTH
    report = evaluate_all(await load_rule_context(database, claim.claim_id))
    assert report.hard_blocks == ["R02"]


async def test_simple_claim_picker_finds_a_clean_claim(database: ReadOnlyDatabase) -> None:
    number = await pick_simple_claim(database)
    claim = await load_claim_input(database, number)
    assert evaluate_all(await load_rule_context(database, claim.claim_id)).triggered_rule_ids == []


async def test_unknown_claim_raises(database: ReadOnlyDatabase) -> None:
    with pytest.raises(ClaimNotFoundError):
        await load_claim_input(database, "CLM-1999-000000")


async def test_all_nine_investigator_tools_are_wired(database: ReadOnlyDatabase) -> None:
    client = AsyncQdrantClient(location=":memory:")
    try:
        tools = build_investigator_tools(PolicyRetriever(client, FakeEmbedder(), "wiring_test"), database)
        assert {tool.name for tool in tools} == INVESTIGATOR_TOOL_NAMES
    finally:
        await client.close()

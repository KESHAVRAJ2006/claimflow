"""Tests for ingestion and filtered search, using in-memory Qdrant and a fake embedder (no network, no model)."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient, models

from app.domain.enums import ProductType
from app.retrieval.chunking import SectionKind
from app.retrieval.retriever import (
    CONFIDENCE_SCORE_CEILING,
    CONFIDENCE_SCORE_FLOOR,
    PolicyRetriever,
    retrieval_confidence,
)
from scripts.generate_policy_pdfs import OUTPUT_DIR, parse_source, render_source
from tests.fakes import FakeEmbedder

COLLECTION = "test_policy_chunks"


@pytest.fixture
async def qdrant() -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(location=":memory:")
    yield client
    await client.close()


@pytest.fixture
async def retriever(qdrant: AsyncQdrantClient) -> PolicyRetriever:
    policy_retriever = PolicyRetriever(qdrant, FakeEmbedder(), COLLECTION)
    await policy_retriever.ingest_policy_documents(OUTPUT_DIR)
    return policy_retriever


async def _point_count(client: AsyncQdrantClient) -> int:
    return (await client.count(COLLECTION, exact=True)).count


async def test_ingestion_is_idempotent(qdrant: AsyncQdrantClient, retriever: PolicyRetriever) -> None:
    count_after_first = await _point_count(qdrant)
    assert count_after_first > 30

    second = await retriever.ingest_policy_documents(OUTPUT_DIR)
    assert [result.status for result in second] == ["unchanged"] * 3
    assert await _point_count(qdrant) == count_after_first
    assert sum(result.chunk_count for result in second) == count_after_first


async def test_changed_document_replaces_its_old_chunks(qdrant: AsyncQdrantClient, tmp_path: Path) -> None:
    policy_retriever = PolicyRetriever(qdrant, FakeEmbedder(), COLLECTION)
    await policy_retriever.ensure_collection()
    path = tmp_path / "Sample.pdf"
    long_page = "SECTION 3 - WHAT IS COVERED\n" + "\n\n".join(["Covered event clause text. " * 12] * 8)
    version_1 = f"TITLE: V1\nDOCUMENT: Sample.pdf\n@@PAGE\n{long_page}\n@@PAGE\n{long_page}"
    version_2 = "TITLE: V2\nDOCUMENT: Sample.pdf\n@@PAGE\nSECTION 3 - WHAT IS COVERED\nShort."

    path.write_bytes(render_source(parse_source(version_1)))
    first = await policy_retriever.ingest_pdf(path, ProductType.HOME)
    path.write_bytes(render_source(parse_source(version_2)))
    second = await policy_retriever.ingest_pdf(path, ProductType.HOME)

    assert first.status == second.status == "ingested"
    assert second.chunk_count < first.chunk_count
    assert await _point_count(qdrant) == second.chunk_count  # no stale chunks from version 1 remain


async def test_search_applies_product_and_section_filters(retriever: PolicyRetriever) -> None:
    result = await retriever.search(
        "is this excluded", product_type=ProductType.HEALTH, section_kinds=(SectionKind.EXCLUSIONS,)
    )
    assert result.chunks
    assert {chunk.product_type for chunk in result.chunks} == {ProductType.HEALTH}
    assert {chunk.section_kind for chunk in result.chunks} == {SectionKind.EXCLUSIONS}
    assert {chunk.citation.document for chunk in result.chunks} == {"Health_Policy.pdf"}


async def test_hits_carry_citations_and_are_ranked(retriever: PolicyRetriever) -> None:
    result = await retriever.search_policy("hydrostatic lock engine water flood", ProductType.MOTOR)
    top = result.chunks[0]
    assert top.citation.label == f"Motor_Policy.pdf p.{top.citation.page}"
    assert top.citation.section is not None
    assert [chunk.score for chunk in result.chunks] == sorted((chunk.score for chunk in result.chunks), reverse=True)
    assert "passages; best Motor_Policy.pdf" in result.summary()


async def test_specialised_searches_only_look_in_their_sections(retriever: PolicyRetriever) -> None:
    exclusions = await retriever.check_exclusions(ProductType.HOME, "burglary while the house was empty")
    waiting = await retriever.get_waiting_period(ProductType.HEALTH, "cataract")
    coverage = await retriever.get_coverage_section(ProductType.MOTOR, "theft")
    assert {c.section_kind for c in exclusions.chunks} <= {SectionKind.EXCLUSIONS, SectionKind.CONDITIONS}
    assert {c.section_kind for c in waiting.chunks} == {SectionKind.WAITING_PERIODS}
    assert {c.section_kind for c in coverage.chunks} <= {SectionKind.COVERAGE, SectionKind.LIMITS}


async def test_collection_with_a_different_vector_size_is_rejected(qdrant: AsyncQdrantClient) -> None:
    await qdrant.create_collection(
        "wrong_size", vectors_config=models.VectorParams(size=8, distance=models.Distance.COSINE)
    )
    with pytest.raises(ValueError, match="vector size"):
        await PolicyRetriever(qdrant, FakeEmbedder(dimension=64), "wrong_size").ensure_collection()


def test_retrieval_confidence_calibration() -> None:
    assert retrieval_confidence(None) == 0.0
    assert retrieval_confidence(CONFIDENCE_SCORE_FLOOR - 0.1) == 0.0
    assert retrieval_confidence(CONFIDENCE_SCORE_CEILING + 0.1) == 1.0
    midpoint = (CONFIDENCE_SCORE_FLOOR + CONFIDENCE_SCORE_CEILING) / 2
    assert retrieval_confidence(midpoint) == pytest.approx(0.5, abs=0.001)

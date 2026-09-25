"""Tests for chunking and section classification."""

import pytest

from app.domain.enums import ProductType
from app.retrieval.chunking import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    SUBSECTION_HEADING,
    Chunk,
    SectionKind,
    chunk_document,
    classify_section,
)
from app.retrieval.pdf_tools import ExtractedDocument, PageText, extract_pdf_file
from scripts.generate_policy_pdfs import OUTPUT_DIR


@pytest.fixture(scope="module")
def motor_document() -> ExtractedDocument:
    return extract_pdf_file(OUTPUT_DIR / "Motor_Policy.pdf")


@pytest.fixture(scope="module")
def motor_chunks(motor_document: ExtractedDocument) -> list[Chunk]:
    return chunk_document(motor_document, ProductType.MOTOR)


@pytest.fixture(scope="module")
def health_chunks() -> list[Chunk]:
    return chunk_document(extract_pdf_file(OUTPUT_DIR / "Health_Policy.pdf"), ProductType.HEALTH)


def _chunk_containing(chunks: list[Chunk], phrase: str) -> Chunk:
    # PDF text wraps at line ends, so compare with all whitespace runs collapsed to single spaces.
    matches = [chunk for chunk in chunks if phrase in " ".join(chunk.text.split())]
    assert matches, f"no chunk contains {phrase!r}"
    return matches[0]


def test_chunks_respect_size_and_stay_on_one_page(motor_document: ExtractedDocument, motor_chunks: list[Chunk]) -> None:
    pages = {page.page_number: page.text for page in motor_document.pages}
    assert all(len(chunk.text) <= CHUNK_SIZE for chunk in motor_chunks)
    for chunk in motor_chunks:
        # Offsets point at the exact text on the chunk's own page, which is what the UI highlights.
        assert pages[chunk.page_number][chunk.char_start : chunk.char_end] == chunk.text


def test_consecutive_chunks_overlap_by_at_most_the_configured_amount(motor_chunks: list[Chunk]) -> None:
    for previous, current in zip(motor_chunks, motor_chunks[1:], strict=False):
        if previous.page_number == current.page_number:
            assert current.char_start > previous.char_start
            assert previous.char_end - current.char_start <= CHUNK_OVERLAP


def test_chunk_ids_are_unique_and_deterministic(motor_document: ExtractedDocument, motor_chunks: list[Chunk]) -> None:
    assert len({chunk.chunk_id for chunk in motor_chunks}) == len(motor_chunks)
    again = chunk_document(motor_document, ProductType.MOTOR)
    assert [chunk.chunk_id for chunk in again] == [chunk.chunk_id for chunk in motor_chunks]
    assert [chunk.chunk_index for chunk in motor_chunks] == list(range(len(motor_chunks)))


def test_exclusion_passage_is_tagged_with_its_section(motor_chunks: list[Chunk]) -> None:
    chunk = _chunk_containing(motor_chunks, "hydrostatic lock), including damage")
    assert chunk.section_kind is SectionKind.EXCLUSIONS
    assert chunk.page_number == 3
    assert chunk.subsection_title == "4.3 Flood and engine damage"


def test_coverage_and_limits_and_waiting_sections_are_distinguished(
    motor_chunks: list[Chunk], health_chunks: list[Chunk]
) -> None:
    assert _chunk_containing(motor_chunks, "Rubber, nylon and plastic parts").section_kind is SectionKind.COVERAGE
    assert _chunk_containing(health_chunks, "INR 40,000 for each eye").section_kind is SectionKind.LIMITS
    waiting = _chunk_containing(health_chunks, "36 months of continuous coverage")
    assert waiting.section_kind is SectionKind.WAITING_PERIODS
    assert _chunk_containing(health_chunks, "Cosmetic or plastic surgery").section_kind is SectionKind.EXCLUSIONS


@pytest.mark.parametrize(
    ("title", "kind"),
    [
        ("SECTION 4 - WHAT IS NOT COVERED (EXCLUSIONS)", SectionKind.EXCLUSIONS),
        ("SECTION 3 - WHAT IS COVERED", SectionKind.COVERAGE),
        ("SECTION 5 - WAITING PERIODS", SectionKind.WAITING_PERIODS),
        ("SECTION 4 - SUB-LIMITS, DEDUCTIBLE AND CO-PAYMENT", SectionKind.LIMITS),
        ("SECTION 6 - CLAIMS PROCEDURE", SectionKind.CLAIMS_PROCEDURE),
        ("SECTION 2 - DEFINITIONS", SectionKind.DEFINITIONS),
        ("SECTION 7 - GENERAL CONDITIONS", SectionKind.CONDITIONS),
        ("SECTION 1 - ABOUT THIS POLICY", SectionKind.GENERAL),
        (None, SectionKind.GENERAL),
    ],
)
def test_classify_section(title: str | None, kind: SectionKind) -> None:
    assert classify_section(title) is kind


def test_body_sentences_are_not_mistaken_for_subheadings() -> None:
    assert SUBSECTION_HEADING.match("4.3 Flood and engine damage")
    assert not SUBSECTION_HEADING.match("3.5 percent of the claim amount is deducted.")
    assert not SUBSECTION_HEADING.match("4.1 The deductible applies to each claim.")


def test_section_carries_over_to_a_page_without_headings() -> None:
    document = ExtractedDocument(
        document="Carry.pdf",
        title=None,
        sha256="0" * 64,
        page_count=2,
        pages=(
            PageText(document="Carry.pdf", page_number=1, text="SECTION 6 - WHAT IS NOT COVERED\nFirst page text."),
            PageText(document="Carry.pdf", page_number=2, text="Continuation of the exclusions list on page two."),
        ),
    )
    chunks = chunk_document(document, ProductType.HOME)
    assert chunks[-1].page_number == 2
    assert chunks[-1].section_kind is SectionKind.EXCLUSIONS

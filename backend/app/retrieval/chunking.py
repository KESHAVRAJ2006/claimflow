"""Split extracted pages into overlapping chunks tagged with section metadata."""

import re
import uuid
from enum import StrEnum

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import ProductType
from app.retrieval.pdf_tools import ExtractedDocument

# 900 characters is roughly 150-200 words: one clause with its conditions, small enough that a citation
# points at a specific passage. 130 characters of overlap (~15%) keeps a sentence that straddles a boundary
# readable in at least one chunk.
CHUNK_SIZE = 900
CHUNK_OVERLAP = 130

# Headings as they appear in the policy wordings, e.g. "SECTION 4 - WHAT IS NOT COVERED (EXCLUSIONS)"
# and "4.3 Flood and engine damage". Subsections must start with a capital and have no full stop, so a body
# sentence such as "3.5 percent of..." is never mistaken for a heading.
SECTION_HEADING = re.compile(r"^SECTION (\d+) - (.+)$", re.MULTILINE)
SUBSECTION_HEADING = re.compile(r"^(\d+)\.(\d+) ([A-Z][^.\n]{0,80})$", re.MULTILINE)

# Fixed namespace so the same document, page and position always produce the same point ID.
CHUNK_ID_NAMESPACE = uuid.UUID("6f1c9a52-3f0e-4d8e-9b7a-2c4d5e6f7a8b")


class SectionKind(StrEnum):
    """What a policy section is about. Lets tools search only the relevant part of a wording."""

    COVERAGE = "coverage"
    LIMITS = "limits"
    EXCLUSIONS = "exclusions"
    WAITING_PERIODS = "waiting_periods"
    CLAIMS_PROCEDURE = "claims_procedure"
    DEFINITIONS = "definitions"
    CONDITIONS = "conditions"
    GENERAL = "general"


# Checked in order: "WHAT IS NOT COVERED" must match exclusions before the looser "COVERED" matches coverage.
_KIND_KEYWORDS: tuple[tuple[SectionKind, tuple[str, ...]], ...] = (
    (SectionKind.EXCLUSIONS, ("NOT COVERED", "EXCLUSION")),
    (SectionKind.WAITING_PERIODS, ("WAITING PERIOD",)),
    (SectionKind.LIMITS, ("LIMIT", "DEDUCTIBLE", "CO-PAYMENT")),
    (SectionKind.COVERAGE, ("WHAT IS COVERED",)),
    (SectionKind.CLAIMS_PROCEDURE, ("CLAIM",)),
    (SectionKind.DEFINITIONS, ("DEFINITION",)),
    (SectionKind.CONDITIONS, ("CONDITION",)),
)


class Chunk(BaseModel):
    """A retrievable passage with everything needed to cite and highlight it."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document: str
    document_sha256: str
    product_type: ProductType
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0, description="Position within the document, across all pages")
    section_title: str | None
    subsection_title: str | None
    section_kind: SectionKind
    text: str
    char_start: int = Field(ge=0, description="Offset of the chunk in its page text, for highlighting")
    char_end: int


def classify_section(section_title: str | None) -> SectionKind:
    """Map a top-level section heading to a SectionKind.

    Args:
        section_title: Heading text such as ``"SECTION 4 - WHAT IS NOT COVERED (EXCLUSIONS)"``.

    Returns:
        The matching kind, or GENERAL when no keyword matches.
    """
    if not section_title:
        return SectionKind.GENERAL
    upper = section_title.upper()
    for kind, keywords in _KIND_KEYWORDS:
        if any(keyword in upper for keyword in keywords):
            return kind
    return SectionKind.GENERAL


def _headings(text: str) -> list[tuple[int, str, str]]:
    """Find headings on a page.

    Args:
        text: Page text.

    Returns:
        (offset, level, heading) tuples sorted by offset; level is "section" or "subsection".
    """
    found = [(m.start(), "section", m.group(0).strip()) for m in SECTION_HEADING.finditer(text)]
    found += [(m.start(), "subsection", m.group(0).strip()) for m in SUBSECTION_HEADING.finditer(text)]
    return sorted(found)


def _segments(text: str, headings: list[tuple[int, str, str]]) -> list[tuple[int, int]]:
    """Split page text at headings, so each segment belongs to exactly one (sub)section.

    A heading with no body text of its own (e.g. "SECTION 3 - ..." immediately followed by "3.1 ...") stays in
    the same segment as the next heading, so no chunk consists of a bare heading.

    Args:
        text: Page text.
        headings: Headings found on the page, sorted by offset.

    Returns:
        (start, end) offsets covering the whole page, in order.
    """
    boundaries = [0]
    for index, (offset, _, _heading) in enumerate(headings):
        if index > 0:
            previous_offset, _, previous_heading = headings[index - 1]
            body_between = text[previous_offset + len(previous_heading) : offset]
            if not body_between.strip():
                continue  # previous heading has no body; keep it with this one
        if offset > boundaries[-1]:
            boundaries.append(offset)
    return list(zip(boundaries, [*boundaries[1:], len(text)], strict=True))


def chunk_document(
    document: ExtractedDocument,
    product_type: ProductType,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split a document into overlapping chunks that never cross a page or a heading.

    Not crossing a page means every citation has exactly one page number. Not crossing a heading means every
    chunk's section label is correct by construction: a chunk that ran from the end of "4.2 Wear and tear" into
    "4.3 Flood and engine damage" would otherwise be cited under the wrong clause. Section headings carry over
    to text at the top of the next page.

    Args:
        document: Extracted PDF.
        product_type: Product the wording belongs to.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters shared by consecutive chunks within one (sub)section.

    Returns:
        Chunks in reading order.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Prefer breaking between paragraphs, then lines, then sentences, before splitting mid-sentence.
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks: list[Chunk] = []
    section: str | None = None
    subsection: str | None = None

    for page in document.pages:
        headings = _headings(page.text)
        page_chunk_index = 0
        for segment_start, segment_end in _segments(page.text, headings):
            # Headings only ever sit at the start of a segment, so updating state here covers the whole segment.
            for offset, level, heading in headings:
                if segment_start <= offset < segment_end:
                    if level == "section":
                        section, subsection = heading, None
                    else:
                        subsection = heading
            for piece in splitter.create_documents([page.text[segment_start:segment_end]]):
                start = segment_start + piece.metadata["start_index"]
                chunks.append(
                    Chunk(
                        chunk_id=str(
                            uuid.uuid5(CHUNK_ID_NAMESPACE, f"{document.document}|{page.page_number}|{page_chunk_index}")
                        ),
                        document=document.document,
                        document_sha256=document.sha256,
                        product_type=product_type,
                        page_number=page.page_number,
                        chunk_index=len(chunks),
                        section_title=section,
                        subsection_title=subsection,
                        section_kind=classify_section(section),
                        text=piece.page_content,
                        char_start=start,
                        char_end=start + len(piece.page_content),
                    )
                )
                page_chunk_index += 1
    return chunks

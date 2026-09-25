"""Per-page text extraction from PDFs, with the metadata citations need."""

import hashlib
import io
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from pypdf import PdfReader

# Every PDF starts with these bytes. Checking content rather than the file extension stops a renamed
# executable or HTML file from being processed as a PDF.
PDF_MAGIC_BYTES = b"%PDF-"
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


class PdfExtractionError(ValueError):
    """Raised when a file is not a readable, unencrypted PDF."""


class PageText(BaseModel):
    """Text of one PDF page."""

    model_config = ConfigDict(frozen=True)

    document: str
    page_number: int = Field(ge=1, description="1-based, matching the number a human sees in a PDF viewer")
    text: str


class ExtractedDocument(BaseModel):
    """All pages of one PDF plus identifying metadata."""

    model_config = ConfigDict(frozen=True)

    document: str = Field(description="File name used in citations, e.g. Motor_Policy.pdf")
    title: str | None
    sha256: str = Field(description="Hash of the raw file bytes; changes whenever the document changes")
    page_count: int
    pages: tuple[PageText, ...]


def is_pdf(data: bytes) -> bool:
    """Check whether bytes look like a PDF by their magic number.

    Args:
        data: The file contents, or at least its first few bytes.

    Returns:
        True if the data starts with ``%PDF-``.
    """
    return data.startswith(PDF_MAGIC_BYTES)


def _normalise(text: str) -> str:
    """Tidy extracted text without changing its words.

    Args:
        text: Raw text from pypdf.

    Returns:
        Text with trailing spaces removed and runs of blank lines collapsed.
    """
    lines = [line.rstrip() for line in text.replace("\x00", "").splitlines()]
    return _EXCESS_BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def extract_pdf(data: bytes, document_name: str) -> ExtractedDocument:
    """Extract the text of every page of a PDF.

    Args:
        data: Raw PDF bytes.
        document_name: Name to record for citations (normally the file name).

    Returns:
        The extracted document with one PageText per page, in order.

    Raises:
        PdfExtractionError: If the bytes are not a PDF, are encrypted, or cannot be parsed.
    """
    if not is_pdf(data):
        raise PdfExtractionError(f"{document_name} is not a PDF (missing %PDF- header)")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise PdfExtractionError(f"{document_name} is encrypted")
        pages = tuple(
            PageText(document=document_name, page_number=index + 1, text=_normalise(page.extract_text() or ""))
            for index, page in enumerate(reader.pages)
        )
        title = reader.metadata.title if reader.metadata else None
    except PdfExtractionError:
        raise
    except Exception as exc:  # noqa: BLE001 — pypdf raises many exception types (PdfReadError, KeyError, ...) on malformed files
        raise PdfExtractionError(f"{document_name} could not be parsed: {type(exc).__name__}: {exc}") from exc
    return ExtractedDocument(
        document=document_name,
        title=title,
        sha256=hashlib.sha256(data).hexdigest(),
        page_count=len(pages),
        pages=pages,
    )


def extract_pdf_file(path: Path) -> ExtractedDocument:
    """Extract a PDF from disk, naming it after the file.

    Args:
        path: Path to the PDF.

    Returns:
        The extracted document.

    Raises:
        FileNotFoundError: If the file does not exist.
        PdfExtractionError: If it is not a readable PDF.
    """
    return extract_pdf(path.read_bytes(), path.name)

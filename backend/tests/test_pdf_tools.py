"""Tests for PDF extraction and the policy PDF generator."""

from pathlib import Path

import pytest

from app.retrieval.pdf_tools import PdfExtractionError, extract_pdf, extract_pdf_file, is_pdf
from scripts.generate_policy_pdfs import OUTPUT_DIR, PageOverflowError, generate, parse_source, render_source

SAMPLE_SOURCE = """TITLE: Sample Wording
DOCUMENT: Sample.pdf
@@PAGE
SECTION 1 - ABOUT THIS POLICY
Hello (world) on page one.
@@PAGE
1.1 More details
Text that lives on page two.
"""


@pytest.fixture(scope="module")
def sample_pdf() -> bytes:
    return render_source(parse_source(SAMPLE_SOURCE))


def test_extracts_each_page_with_one_based_numbers(sample_pdf: bytes) -> None:
    document = extract_pdf(sample_pdf, "Sample.pdf")
    assert document.page_count == 2
    assert [page.page_number for page in document.pages] == [1, 2]
    assert "Hello (world) on page one." in document.pages[0].text
    assert "Text that lives on page two." in document.pages[1].text
    assert "page two" not in document.pages[0].text
    assert document.title == "Sample Wording"


def test_hash_identifies_content_and_is_stable(sample_pdf: bytes) -> None:
    first = extract_pdf(sample_pdf, "Sample.pdf")
    assert first.sha256 == extract_pdf(sample_pdf, "Renamed.pdf").sha256
    assert len(first.sha256) == 64


def test_magic_bytes_decide_what_is_a_pdf(sample_pdf: bytes) -> None:
    assert is_pdf(sample_pdf)
    assert not is_pdf(b"<html><script>alert(1)</script>")
    assert not is_pdf(b"MZ\x90\x00")  # a Windows executable header


def test_rejects_non_pdf_even_when_named_pdf() -> None:
    with pytest.raises(PdfExtractionError, match="not a PDF"):
        extract_pdf(b"<html>not really a pdf</html>", "invoice.pdf")


def test_rejects_corrupt_pdf() -> None:
    with pytest.raises(PdfExtractionError, match="could not be parsed"):
        extract_pdf(b"%PDF-1.4\nthis is not a real pdf body", "broken.pdf")


def test_generator_refuses_a_page_that_does_not_fit() -> None:
    overflowing = SAMPLE_SOURCE.replace("Hello (world) on page one.", "\n".join(["A long paragraph line."] * 80))
    with pytest.raises(PageOverflowError, match="Sample.pdf page 1"):
        render_source(parse_source(overflowing))


def test_generator_requires_ascii() -> None:
    with pytest.raises(ValueError, match="ASCII"):
        parse_source(SAMPLE_SOURCE.replace("Hello", "Price ₹500"))


def test_generator_output_is_deterministic() -> None:
    assert render_source(parse_source(SAMPLE_SOURCE)) == render_source(parse_source(SAMPLE_SOURCE))


def test_committed_policy_pdfs_match_their_sources() -> None:
    assert generate(check_only=True) == [], "run: python -m scripts.generate_policy_pdfs"


@pytest.mark.parametrize(("name", "pages"), [("Motor_Policy.pdf", 5), ("Health_Policy.pdf", 6), ("Home_Policy.pdf", 5)])
def test_policy_pdfs_extract_cleanly(name: str, pages: int) -> None:
    document = extract_pdf_file(Path(OUTPUT_DIR) / name)
    assert document.page_count == pages
    assert all(len(page.text) > 500 for page in document.pages)
    assert "SECTION 1 - ABOUT THIS POLICY" in document.pages[0].text

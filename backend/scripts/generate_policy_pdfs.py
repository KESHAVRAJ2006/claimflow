"""Generate the sample policy wording PDFs from the plain-text sources in data/policy_sources.

The PDFs are committed, so this only needs re-running after editing a source. Output is byte-for-byte
deterministic (no timestamps), so an unchanged source produces an identical file and ingestion skips it.

Source format:
    TITLE: <document title>
    DOCUMENT: <output file name>
    @@PAGE                      <- starts each page; page numbers are therefore fixed by the author
    SECTION 4 - HEADING         <- bold section heading
    4.1 Subheading              <- bold subsection heading
    - bullet text
    paragraph text (one line per paragraph; wrapped automatically)

Usage:
    python -m scripts.generate_policy_pdfs            # write PDFs
    python -m scripts.generate_policy_pdfs --check    # exit 1 if any PDF is out of date
"""

import argparse
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

from app.retrieval.chunking import SECTION_HEADING, SUBSECTION_HEADING

BACKEND_DIR = Path(__file__).resolve().parent.parent
SOURCES_DIR = BACKEND_DIR / "data" / "policy_sources"
OUTPUT_DIR = BACKEND_DIR / "data" / "policies"
PAGE_BREAK = "@@PAGE"

# A4 in PostScript points (1/72 inch).
PAGE_WIDTH, PAGE_HEIGHT = 595, 842
LEFT_MARGIN = 56
TOP_Y = 786
BOTTOM_Y = 50
BODY_SIZE, BODY_LEADING = 10, 13.5
PARAGRAPH_GAP = 5
# Helvetica at 10pt averages about 5pt per character, and the text column is 483pt wide.
WRAP_CHARACTERS = 96


class PageOverflowError(ValueError):
    """A source page holds more text than fits on one PDF page."""


@dataclass(frozen=True)
class PolicySource:
    """A parsed source file."""

    title: str
    document: str
    pages: list[list[str]]


def parse_source(text: str) -> PolicySource:
    """Parse a policy source file.

    Args:
        text: File contents.

    Returns:
        The title, output file name and the lines of each page.

    Raises:
        ValueError: If the header is incomplete, there are no pages, or the text is not ASCII.
    """
    if not text.isascii():
        # The built-in Helvetica font only covers a basic character set; non-ASCII would extract as garbage.
        raise ValueError("policy sources must be ASCII (write INR, not the rupee sign; straight quotes only)")
    header, *pages = text.split(PAGE_BREAK)
    fields = dict(line.split(": ", 1) for line in header.strip().splitlines())
    if "TITLE" not in fields or "DOCUMENT" not in fields or not pages:
        raise ValueError("source needs TITLE:, DOCUMENT: and at least one @@PAGE")
    return PolicySource(fields["TITLE"], fields["DOCUMENT"], [page.strip("\n").splitlines() for page in pages])


def _escape(text: str) -> str:
    """Escape characters that are special inside a PDF string literal."""
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def layout_page(lines: list[str], *, title: str | None, label: str) -> bytes:
    """Lay out one page as a PDF content stream.

    Args:
        lines: Source lines for the page.
        title: Document title to print at the top (first page only).
        label: Name used in error messages, e.g. "Motor_Policy.pdf page 3".

    Returns:
        The content stream bytes.

    Raises:
        PageOverflowError: If the text runs past the bottom margin.
    """
    operations: list[str] = []
    y = float(TOP_Y)

    def draw(text: str, font: str, size: float, x: float = LEFT_MARGIN) -> None:
        # One BT...ET block per line at an absolute position: simple, and pypdf extracts each as its own line.
        operations.append(f"BT /{font} {size} Tf {x:.1f} {y:.1f} Td ({_escape(text)}) Tj ET")

    if title:
        draw(title, "F2", 15)
        y -= 26
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            y -= PARAGRAPH_GAP
        elif SECTION_HEADING.match(line):
            y -= 6
            draw(line, "F2", 12)
            y -= 18
        elif SUBSECTION_HEADING.match(line):
            draw(line, "F2", 10.5)
            y -= 15
        elif line.startswith("- "):
            for index, part in enumerate(textwrap.wrap(line[2:], WRAP_CHARACTERS - 4)):
                draw(("- " if index == 0 else "  ") + part, "F1", BODY_SIZE, LEFT_MARGIN + 8)
                y -= BODY_LEADING
        else:
            for part in textwrap.wrap(line, WRAP_CHARACTERS):
                draw(part, "F1", BODY_SIZE)
                y -= BODY_LEADING
        if y < BOTTOM_Y:
            raise PageOverflowError(f"{label} is too long to fit on one page; move text to the next @@PAGE")
    return "\n".join(operations).encode("ascii")


def build_pdf(page_streams: list[bytes], title: str) -> bytes:
    """Assemble a minimal, valid PDF 1.4 file.

    Args:
        page_streams: One content stream per page.
        title: Document title for the metadata dictionary.

    Returns:
        The PDF bytes.
    """
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        4: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
        5: f"<< /Title ({_escape(title)}) /Producer (ClaimFlow policy generator) >>".encode("ascii"),
    }
    page_ids = []
    for index, stream in enumerate(page_streams):
        page_id, content_id = 6 + 2 * index, 7 + 2 * index
        page_ids.append(page_id)
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode("ascii")
        objects[content_id] = f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"\nendstream"
    objects[2] = f"<< /Type /Pages /Kids [{' '.join(f'{i} 0 R' for i in page_ids)}] /Count {len(page_ids)} >>".encode()

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")  # binary comment line marks the file as binary
    offsets: dict[int, int] = {}
    for object_id in sorted(objects):
        offsets[object_id] = len(output)
        output += f"{object_id} 0 obj\n".encode("ascii") + objects[object_id] + b"\nendobj\n"
    xref_offset = len(output)
    size = max(objects) + 1
    # Each cross-reference entry must be exactly 20 bytes, including the trailing space and newline.
    output += f"xref\n0 {size}\n0000000000 65535 f \n".encode("ascii")
    output += "".join(f"{offsets[i]:010d} 00000 n \n" for i in range(1, size)).encode("ascii")
    output += f"trailer\n<< /Size {size} /Root 1 0 R /Info 5 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
    return bytes(output)


def render_source(source: PolicySource) -> bytes:
    """Render a parsed source into PDF bytes.

    Args:
        source: Parsed source.

    Returns:
        The PDF bytes.
    """
    streams = [
        layout_page(lines, title=source.title if number == 1 else None, label=f"{source.document} page {number}")
        for number, lines in enumerate(source.pages, start=1)
    ]
    return build_pdf(streams, source.title)


def generate(check_only: bool = False) -> list[str]:
    """Render every source and write (or verify) the PDFs.

    Args:
        check_only: Compare with existing files instead of writing.

    Returns:
        Names of documents that were written, or in check mode that are missing or out of date.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    changed = []
    for source_path in sorted(SOURCES_DIR.glob("*.txt")):
        source = parse_source(source_path.read_text(encoding="utf-8"))
        pdf = render_source(source)
        target = OUTPUT_DIR / source.document
        if target.is_file() and target.read_bytes() == pdf:
            continue
        changed.append(source.document)
        if not check_only:
            target.write_bytes(pdf)
    return changed


def main() -> int:
    """CLI entry point.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Generate policy wording PDFs from text sources.")
    parser.add_argument("--check", action="store_true", help="fail if any PDF is missing or out of date")
    args = parser.parse_args()
    changed = generate(check_only=args.check)
    if args.check:
        if changed:
            print(f"Out of date: {', '.join(changed)}. Run: python -m scripts.generate_policy_pdfs")
            return 1
        print("All policy PDFs are up to date.")
        return 0
    print(f"Wrote: {', '.join(changed)}" if changed else "All policy PDFs already up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

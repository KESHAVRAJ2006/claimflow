"""Receive a claim document: size-limit, type-check by magic bytes, extract text, and always delete the file.

The uploaded PDF is never kept. Its text goes to the intake agent and the file is deleted in a ``finally`` block,
on success and on every failure path, so no customer document outlives the request that carried it.
"""

import asyncio
import logging
import re
import tempfile
from pathlib import Path

from fastapi import UploadFile, status

from app.agents.state import ClaimDocument, DocumentPage
from app.api.problems import ProblemError
from app.retrieval.pdf_tools import PDF_MAGIC_BYTES, PdfExtractionError, extract_pdf_file

logger = logging.getLogger(__name__)

CHUNK_BYTES = 1024 * 1024
_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._ -]")


def safe_filename(name: str | None) -> str:
    """Reduce a client-supplied file name to something safe to store and display.

    Args:
        name: The name the browser sent (may contain paths, control characters, anything).

    Returns:
        The base name with unusual characters replaced, at most 120 characters; "document.pdf" if nothing is left.
    """
    # Both separators: a Windows browser can send "C:\\Users\\me\\claim.pdf".
    base = re.split(r"[\\/]", name or "")[-1]
    cleaned = _UNSAFE_FILENAME.sub("_", base).strip(" .")[:120]
    return cleaned or "document.pdf"


async def save_upload(upload: UploadFile, directory: Path, max_bytes: int) -> Path:
    """Stream an upload to a new temporary file, enforcing the size limit.

    Args:
        upload: The uploaded file.
        directory: Where to write it.
        max_bytes: Largest allowed size.

    Returns:
        Path of the written file. The caller must delete it.

    Raises:
        ProblemError: 413 if the file is larger than ``max_bytes``.
    """
    await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)
    # delete=False: we delete it ourselves in the caller's finally, after pypdf has read it by path.
    handle = tempfile.NamedTemporaryFile(dir=directory, prefix="claim-", suffix=".pdf", delete=False)  # noqa: SIM115
    path = Path(handle.name)
    try:
        size = 0
        while chunk := await upload.read(CHUNK_BYTES):
            size += len(chunk)
            if size > max_bytes:
                raise ProblemError(
                    413, "file-too-large", "File too large",
                    f"The document exceeds the {max_bytes // (1024 * 1024)} MB limit.",
                )  # fmt: skip
            handle.write(chunk)
    except BaseException:
        handle.close()
        # Synchronous on purpose: an await here could itself be cancelled and skip the deletion.
        path.unlink(missing_ok=True)  # noqa: ASYNC240 — the caller never receives the path, so clean up here
        raise
    handle.close()
    return path


def _read_document(path: Path, filename: str, max_pages: int) -> ClaimDocument:
    """Validate and extract a saved PDF. Blocking; run in a worker thread.

    Args:
        path: The saved file.
        filename: Display name for the document.
        max_pages: Largest page count accepted.

    Returns:
        The document text by page.

    Raises:
        ProblemError: 415 if the bytes are not a PDF; 422 if it is unreadable, encrypted, empty or too long.
    """
    with path.open("rb") as file:
        header = file.read(len(PDF_MAGIC_BYTES))
    # Magic bytes, not the extension or Content-Type: both are chosen by the client and trivially faked.
    if header != PDF_MAGIC_BYTES:
        raise ProblemError(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "not-a-pdf", "Unsupported file type",
            "The document must be a PDF (the file does not start with %PDF-).",
        )  # fmt: skip
    try:
        extracted = extract_pdf_file(path)
    except PdfExtractionError as error:
        raise ProblemError(422, "unreadable-pdf", "Unreadable PDF", str(error).replace(path.name, filename)) from error
    if extracted.page_count > max_pages:
        raise ProblemError(
            422, "too-many-pages", "PDF has too many pages",
            f"The document has {extracted.page_count} pages; the limit is {max_pages}.",
        )  # fmt: skip
    pages = tuple(DocumentPage(page=p.page_number, text=p.text) for p in extracted.pages if p.text.strip())
    if not pages:
        # A scanned image without a text layer: intake would have nothing to read. OCR is a listed limitation.
        raise ProblemError(
            422, "no-text-in-pdf", "PDF contains no text",
            "No text could be extracted. Upload a PDF with selectable text (scanned images are not supported).",
        )  # fmt: skip
    return ClaimDocument(name=filename, pages=pages)


async def receive_claim_document(upload: UploadFile, directory: Path, max_bytes: int, max_pages: int) -> ClaimDocument:
    """Save, validate and extract an uploaded claim document, deleting the file whatever happens.

    Args:
        upload: The uploaded file.
        directory: Temporary upload directory.
        max_bytes: Size limit.
        max_pages: Page limit.

    Returns:
        The extracted document.

    Raises:
        ProblemError: 413, 415 or 422 describing what was wrong with the file.
    """
    filename = safe_filename(upload.filename)
    path: Path | None = None
    try:
        path = await save_upload(upload, directory, max_bytes)
        # pypdf is CPU-bound and synchronous; a worker thread keeps the event loop serving other requests.
        return await asyncio.to_thread(_read_document, path, filename, max_pages)
    finally:
        # Always runs: success, validation failure, parser crash or client disconnect (CancelledError).
        if path is not None:
            # Synchronous on purpose (a local unlink takes microseconds): an awaited delete inside ``finally`` could be
            # interrupted by a second cancellation and leave the customer's document on disk.
            path.unlink(missing_ok=True)  # noqa: ASYNC240
        await upload.close()

"""Upload validation, text extraction and chunking.

Pipeline for one upload:

1. ``validate_upload``  - extension, size, emptiness and magic-byte checks
   (cheap checks that run before any parsing).
2. ``extract_pages``     - PDF page-by-page extraction with pypdf, or TXT
   decoding with explicit encoding detection.
3. ``chunk_pages``       - recursive character splitting over the whole text so
   sentences that cross a page break stay together, then mapping every chunk
   back to the one-based page(s) it came from via character offsets.
"""

from __future__ import annotations

import hashlib
import io
import re
import unicodedata
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import PurePath

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.exceptions import (
    CorruptedPDFError,
    EmptyFileError,
    EncryptedPDFError,
    FileTooLargeError,
    InvalidFileContentError,
    NoExtractableTextError,
    TooManyPagesError,
    UnsupportedFileTypeError,
)
from app.core.models import ChunkRecord, FileType, ProcessedDocument
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS: dict[str, FileType] = {".pdf": "pdf", ".txt": "txt"}
PAGE_SEPARATOR = "\n\n"
# Order matters: prefer paragraph breaks, then lines, then sentences, then words.
SPLIT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
_CONTROL_CHARS_ALLOWED = {"\n", "\r", "\t", "\f"}


@dataclass(frozen=True)
class PageText:
    """Text of one extracted page. ``page_number`` is one-based; None for TXT files."""

    page_number: int | None
    text: str


def compute_document_id(content: bytes) -> str:
    """Stable, content-addressed ID: identical bytes always produce the same ID."""
    return hashlib.sha256(content).hexdigest()[:16]


def sanitize_filename(raw_name: str | None, file_type: FileType | None = None) -> str:
    """Keep only the base name, drop control characters and cap the length.

    Clients may send names such as ``..\\..\\secret.txt`` or ``/etc/passwd``;
    only the final component is kept so no path is ever stored or echoed.
    """
    name = (raw_name or "").replace("\\", "/").split("/")[-1]
    name = "".join(ch for ch in name if unicodedata.category(ch)[0] != "C").strip()
    if len(name) > 200:
        stem, dot, ext = name.rpartition(".")
        name = (stem[: 200 - len(ext) - 1] + dot + ext) if dot else name[:200]
    if not name or name in {".", ".."}:
        name = f"document.{file_type or 'txt'}"
    return name


def normalize_text(text: str) -> str:
    """Normalise extracted text without changing its meaning.

    NFKC folds PDF ligatures (e.g. "ﬁ" -> "fi") and full-width characters so the
    same words embed identically; line endings and excess blank lines are tidied.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class DocumentProcessor:
    def __init__(
        self,
        *,
        chunk_size: int,
        chunk_overlap: int,
        max_upload_bytes: int,
        max_pdf_pages: int,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_upload_bytes = max_upload_bytes
        self.max_pdf_pages = max_pdf_pages
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=SPLIT_SEPARATORS,
            keep_separator="end",  # keep sentence punctuation with its sentence
            strip_whitespace=True,
        )

    # ------------------------------------------------------------------ validation
    def detect_file_type(self, filename: str | None) -> FileType:
        suffix = PurePath((filename or "").replace("\\", "/")).suffix.lower()
        file_type = SUPPORTED_EXTENSIONS.get(suffix)
        if file_type is None:
            shown = suffix or "(no extension)"
            raise UnsupportedFileTypeError(
                f"Unsupported file type '{shown}'. Only .pdf and .txt files are accepted."
            )
        return file_type

    def validate_upload(self, filename: str | None, content: bytes) -> tuple[str, FileType]:
        """Run all cheap checks before any parsing or embedding happens."""
        file_type = self.detect_file_type(filename)
        if len(content) == 0:
            raise EmptyFileError()
        if len(content) > self.max_upload_bytes:
            limit_mb = self.max_upload_bytes / (1024 * 1024)
            raise FileTooLargeError(f"The uploaded file exceeds the {limit_mb:g} MB limit.")
        if file_type == "pdf":
            # The PDF header must appear within the first 1024 bytes (ISO 32000).
            if b"%PDF-" not in content[:1024]:
                raise InvalidFileContentError("The file has a .pdf extension but is not a PDF document.")
        elif content.lstrip()[:5] == b"%PDF-":
            raise InvalidFileContentError("The file has a .txt extension but contains a PDF document.")
        return sanitize_filename(filename, file_type), file_type

    # ------------------------------------------------------------------ extraction
    def extract_pages(self, content: bytes, file_type: FileType) -> tuple[list[PageText], str | None, int | None, list[int]]:
        """Return (pages, encoding, page_count, skipped_pages)."""
        if file_type == "pdf":
            pages, page_count, skipped = self._extract_pdf(content)
            return pages, None, page_count, skipped
        text, encoding = self._decode_text(content)
        return [PageText(page_number=None, text=normalize_text(text))], encoding, None, []

    def _extract_pdf(self, content: bytes) -> tuple[list[PageText], int, list[int]]:
        try:
            reader = PdfReader(io.BytesIO(content), strict=False)
            if reader.is_encrypted:
                try:
                    decrypted = reader.decrypt("")  # many PDFs use an empty user password
                except Exception as exc:  # missing crypto backend, unsupported algorithm
                    raise EncryptedPDFError() from exc
                if not decrypted:
                    raise EncryptedPDFError()
            page_count = len(reader.pages)
        except EncryptedPDFError:
            raise
        except (PdfReadError, ValueError, TypeError, KeyError, IndexError, AttributeError, OSError, RecursionError) as exc:
            logger.warning("PDF parsing failed: %s", type(exc).__name__)
            raise CorruptedPDFError() from exc
        except Exception as exc:  # pypdf can raise many internal errors on damaged files
            logger.warning("Unexpected PDF parsing failure: %s", type(exc).__name__)
            raise CorruptedPDFError() from exc

        if page_count == 0:
            raise CorruptedPDFError("The PDF contains no pages.")
        if page_count > self.max_pdf_pages:
            raise TooManyPagesError(
                f"The PDF has {page_count} pages; the maximum is {self.max_pdf_pages}."
            )

        pages: list[PageText] = []
        skipped: list[int] = []
        for page_number in range(1, page_count + 1):  # one-based page numbers
            try:
                raw = reader.pages[page_number - 1].extract_text() or ""
            except Exception as exc:  # one damaged page should not lose the others
                logger.warning("Text extraction failed on page %d: %s", page_number, type(exc).__name__)
                skipped.append(page_number)
                continue
            pages.append(PageText(page_number=page_number, text=normalize_text(raw)))
        if skipped and len(skipped) == page_count:
            raise CorruptedPDFError("Text could not be extracted from any page of the PDF.")
        return pages, page_count, skipped

    def _decode_text(self, content: bytes) -> tuple[str, str]:
        """Decode TXT bytes deliberately instead of guessing silently.

        Order: BOM-marked UTF-8/UTF-16 -> strict UTF-8 -> strict Windows-1252
        (common for files saved by Windows Notepad in older versions) -> Latin-1.
        Binary files are rejected before decoding.
        """
        if content.startswith(b"\xef\xbb\xbf"):
            candidates = ["utf-8-sig"]
        elif content.startswith((b"\xff\xfe", b"\xfe\xff")):
            candidates = ["utf-16"]
        else:
            if b"\x00" in content:
                raise InvalidFileContentError("The .txt file appears to be binary, not plain text.")
            candidates = ["utf-8", "cp1252", "latin-1"]

        for encoding in candidates:
            try:
                text = content.decode(encoding)
            except UnicodeDecodeError:
                continue
            control = sum(
                1 for ch in text if unicodedata.category(ch) == "Cc" and ch not in _CONTROL_CHARS_ALLOWED
            )
            if text and control / len(text) > 0.01:
                raise InvalidFileContentError("The .txt file appears to be binary, not plain text.")
            return text, encoding
        raise InvalidFileContentError("The .txt file could not be decoded as text.")

    # ------------------------------------------------------------------ chunking
    def chunk_pages(self, pages: list[PageText], document_id: str) -> list[ChunkRecord]:
        """Split the full text and map each chunk back to its source page(s).

        Pages are joined with a blank line so the splitter prefers page breaks
        as boundaries, but a paragraph that continues onto the next page can
        still stay in one chunk. Each chunk's character span is located in the
        joined text and intersected with the page spans: a chunk inside one page
        gets page_start == page_end; a chunk crossing a break gets the real
        range; a chunk that cannot be located gets no page at all (never a guess).
        """
        parts: list[str] = []
        page_spans: list[tuple[int, int, int | None]] = []  # (start, end, page_number)
        cursor = 0
        for page in pages:
            if not page.text:
                continue
            if parts:
                parts.append(PAGE_SEPARATOR)
                cursor += len(PAGE_SEPARATOR)
            parts.append(page.text)
            page_spans.append((cursor, cursor + len(page.text), page.page_number))
            cursor += len(page.text)
        full_text = "".join(parts)
        if not full_text.strip() or not any(ch.isalnum() for ch in full_text):
            raise NoExtractableTextError()

        span_starts = [start for start, _, _ in page_spans]
        chunks: list[ChunkRecord] = []
        search_from = 0
        for index, text in enumerate(self._splitter.split_text(full_text)):
            start = full_text.find(text, search_from)
            page_start = page_end = None
            char_start = char_end = None
            if start >= 0:
                char_start, char_end = start, start + len(text)
                # Each new chunk starts after the previous chunk's start.
                search_from = start + 1
                page_numbers = self._pages_for_span(char_start, char_end, page_spans, span_starts)
                if page_numbers:
                    page_start, page_end = min(page_numbers), max(page_numbers)
            else:
                logger.warning("Could not locate chunk %d in source text; page left unknown", index)
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{document_id}:{index:05d}",
                    document_id=document_id,
                    chunk_index=index,
                    text=text,
                    page_start=page_start,
                    page_end=page_end,
                    char_start=char_start,
                    char_end=char_end,
                )
            )
        return chunks

    @staticmethod
    def _pages_for_span(
        start: int,
        end: int,
        page_spans: list[tuple[int, int, int | None]],
        span_starts: list[int],
    ) -> list[int]:
        found: list[int] = []
        first = max(bisect_right(span_starts, start) - 1, 0)
        for span_start, span_end, page_number in page_spans[first:]:
            if span_start >= end:
                break
            if span_end > start and page_number is not None:
                found.append(page_number)
        return found

    # ------------------------------------------------------------------ full pipeline
    def process(self, filename: str | None, content: bytes) -> ProcessedDocument:
        safe_name, file_type = self.validate_upload(filename, content)
        document_id = compute_document_id(content)
        pages, encoding, page_count, skipped = self.extract_pages(content, file_type)
        chunks = self.chunk_pages(pages, document_id)
        char_count = sum(len(page.text) for page in pages)
        logger.info(
            "Processed document %s (%s): pages=%s chunks=%d chars=%d encoding=%s skipped_pages=%s",
            document_id,
            file_type,
            page_count,
            len(chunks),
            char_count,
            encoding,
            skipped,
        )
        return ProcessedDocument(
            document_id=document_id,
            filename=safe_name,
            file_type=file_type,
            page_count=page_count,
            char_count=char_count,
            encoding=encoding,
            chunks=chunks,
            skipped_pages=skipped,
        )

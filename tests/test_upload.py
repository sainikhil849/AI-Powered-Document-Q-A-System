"""Requirements 2-7: uploads, validation errors, metadata and page numbers."""

from __future__ import annotations

import pytest

from app.core.exceptions import (
    CorruptedPDFError,
    EmptyFileError,
    FileTooLargeError,
    InvalidFileContentError,
    NoExtractableTextError,
    UnsupportedFileTypeError,
)
from app.services.document_processor import DocumentProcessor, PageText, sanitize_filename
from tests.conftest import upload
from tests.helpers.pdf_factory import build_pdf


def processor(**overrides) -> DocumentProcessor:
    values = dict(chunk_size=200, chunk_overlap=30, max_upload_bytes=1024 * 1024, max_pdf_pages=50)
    values.update(overrides)
    return DocumentProcessor(**values)


# --- 2. TXT upload ------------------------------------------------------------------
def test_txt_upload_is_indexed(client, handbook_bytes):
    response = upload(client, "employee_handbook.txt", handbook_bytes, "text/plain")
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "indexed"
    assert body["file_type"] == "txt"
    assert body["page_count"] is None
    assert body["chunk_count"] > 1
    assert len(body["document_id"]) == 16
    listed = client.get("/documents").json()
    assert listed["latest_document_id"] == body["document_id"]
    assert listed["documents"][0]["chunk_count"] == body["chunk_count"]


def test_duplicate_upload_is_idempotent(client, container, handbook_bytes):
    first = upload(client, "handbook.txt", handbook_bytes).json()
    chunks_after_first = container.store.count()
    second = upload(client, "copy-of-handbook.txt", handbook_bytes)
    assert second.status_code == 200
    assert second.json()["status"] == "duplicate"
    assert second.json()["document_id"] == first["document_id"]
    assert container.store.count() == chunks_after_first  # nothing re-embedded or duplicated


# --- 3. PDF upload -------------------------------------------------------------------
def test_pdf_upload_is_indexed_with_page_count(client, handbook_pdf_bytes):
    response = upload(client, "employee_handbook.pdf", handbook_pdf_bytes, "application/pdf")
    assert response.status_code == 201
    body = response.json()
    assert body["file_type"] == "pdf"
    assert body["page_count"] == 3
    assert body["chunk_count"] >= 3


# --- 4. invalid file type -------------------------------------------------------------
@pytest.mark.parametrize("name", ["report.docx", "malware.exe", "notes", "image.png", "archive.pdf.zip"])
def test_unsupported_file_types_are_rejected(client, name):
    response = upload(client, name, b"some content")
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_FILE_TYPE"


def test_missing_file_field_is_a_validation_error(client):
    response = client.post("/upload")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


# --- 5. empty / textless documents ------------------------------------------------------
def test_empty_file_is_rejected(client):
    response = upload(client, "empty.txt", b"")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "EMPTY_FILE"


def test_whitespace_only_txt_has_no_extractable_text(client):
    response = upload(client, "blank.txt", b"   \n\n\t  \r\n")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "NO_EXTRACTABLE_TEXT"


def test_pdf_without_text_is_rejected(client):
    response = upload(client, "scanned.pdf", build_pdf([[], []]))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "NO_EXTRACTABLE_TEXT"


# --- 6. corrupted / mismatched / oversized files ----------------------------------------
def test_corrupted_pdf_is_rejected(client):
    corrupted = b"%PDF-1.4\n" + bytes(range(256)) * 20
    response = upload(client, "broken.pdf", corrupted)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "CORRUPTED_PDF"


def test_truncated_pdf_is_rejected(client, handbook_pdf_bytes):
    response = upload(client, "truncated.pdf", handbook_pdf_bytes[:300])
    assert response.status_code == 422
    assert response.json()["error"]["code"] in {"CORRUPTED_PDF", "NO_EXTRACTABLE_TEXT"}


def test_pdf_extension_with_non_pdf_content_is_rejected(client):
    response = upload(client, "fake.pdf", b"just some text pretending to be a PDF")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_FILE_CONTENT"


def test_binary_content_in_txt_is_rejected(client):
    response = upload(client, "binary.txt", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" * 10)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_FILE_CONTENT"


def test_oversized_file_is_rejected_before_processing(client, container, fake_embeddings):
    too_big = b"a " * (container.settings.max_upload_bytes // 2 + 10)
    response = upload(client, "big.txt", too_big)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "FILE_TOO_LARGE"
    assert fake_embeddings.document_calls == []


def test_oversized_body_is_rejected_by_content_length(client, container):
    huge = b"x" * (container.settings.max_upload_bytes + 200 * 1024)
    response = client.post("/upload", content=huge, headers={"content-type": "application/octet-stream"})
    assert response.status_code == 413


def test_error_responses_do_not_echo_paths(client):
    response = upload(client, "C:\\Users\\someone\\secret\\empty.txt", b"")
    assert response.status_code == 400
    assert "Users" not in response.text and "secret" not in response.text


# --- 7. metadata and page-number preservation (processor unit tests) --------------------
def test_pdf_pages_are_one_based_and_preserved():
    pages = [[f"Page one sentence {i} about alpha topics." for i in range(3)],
             [f"Page two sentence {i} about beta topics." for i in range(3)],
             [f"Page three sentence {i} about gamma topics." for i in range(3)]]
    doc = processor(chunk_size=120, chunk_overlap=0).process("three.pdf", build_pdf(pages))
    assert doc.page_count == 3
    for chunk in doc.chunks:
        assert chunk.page_start is not None and chunk.page_start >= 1
        if "alpha" in chunk.text and "beta" not in chunk.text:
            assert chunk.pages == [1]
        if "gamma" in chunk.text and "beta" not in chunk.text:
            assert chunk.pages == [3]
    assert {p for c in doc.chunks for p in c.pages} == {1, 2, 3}


def test_chunk_spanning_a_page_break_reports_both_pages():
    pages = [["The overtime rule continues on the next"], ["page and requires written manager approval."]]
    doc = processor(chunk_size=500, chunk_overlap=0).process("span.pdf", build_pdf(pages))
    assert len(doc.chunks) == 1
    chunk = doc.chunks[0]
    assert (chunk.page_start, chunk.page_end) == (1, 2)
    assert chunk.pages == [1, 2]


def test_chunk_metadata_is_complete_and_offsets_are_exact(handbook_bytes):
    proc = processor(chunk_size=300, chunk_overlap=50)
    doc = proc.process("handbook.txt", handbook_bytes)
    full_text = proc.extract_pages(handbook_bytes, "txt")[0][0].text
    assert [c.chunk_index for c in doc.chunks] == list(range(len(doc.chunks)))
    for chunk in doc.chunks:
        assert chunk.document_id == doc.document_id
        assert chunk.chunk_id == f"{doc.document_id}:{chunk.chunk_index:05d}"
        assert chunk.page_start is None and chunk.page_end is None  # TXT has no pages: never invented
        assert full_text[chunk.char_start : chunk.char_end] == chunk.text
        assert len(chunk.text) <= 300


def test_overlap_repeats_text_between_neighbouring_chunks(handbook_bytes):
    doc = processor(chunk_size=300, chunk_overlap=80).process("handbook.txt", handbook_bytes)
    overlapping = sum(1 for a, b in zip(doc.chunks, doc.chunks[1:], strict=False) if b.char_start < a.char_end)
    assert overlapping >= 1


def test_unlocatable_chunk_gets_no_page():
    proc = processor()
    proc._splitter.split_text = lambda text: ["text that is not in the source"]  # type: ignore[method-assign]
    chunks = proc.chunk_pages([PageText(page_number=1, text="Real page text.")], "abc")
    assert chunks[0].page_start is None and chunks[0].pages == []


@pytest.mark.parametrize(
    ("raw", "expected_encoding"),
    [
        ("Leave policy: 20 days. Café allowance ₹1,500.".encode("utf-8"), "utf-8"),
        ("Leave policy: 20 days. Café".encode("utf-8-sig"), "utf-8-sig"),
        ("Leave policy: 20 days. Café".encode("utf-16"), "utf-16"),
        ("Leave policy: 20 days. Café – “quoted”".encode("cp1252"), "cp1252"),
    ],
)
def test_txt_encodings_are_detected(raw, expected_encoding):
    pages, encoding, page_count, _ = processor().extract_pages(raw, "txt")
    assert encoding == expected_encoding
    assert page_count is None
    assert "Café" in pages[0].text


def test_sanitize_filename_strips_paths_and_control_characters():
    assert sanitize_filename("../../etc/passwd.txt") == "passwd.txt"
    assert sanitize_filename("C:\\Users\\me\\doc.pdf") == "doc.pdf"
    assert sanitize_filename("bad\x00name\n.txt") == "badname.txt"
    assert sanitize_filename("", "pdf") == "document.pdf"


@pytest.mark.parametrize(
    ("name", "content", "error"),
    [
        ("a.docx", b"x", UnsupportedFileTypeError),
        ("a.txt", b"", EmptyFileError),
        ("a.pdf", b"not a pdf", InvalidFileContentError),
        ("a.pdf", b"%PDF-1.7\n garbage garbage", CorruptedPDFError),
        ("a.txt", b"\n\n   \n", NoExtractableTextError),
        ("a.txt", b"x" * (1024 * 1024 + 1), FileTooLargeError),
    ],
)
def test_processor_raises_typed_errors(name, content, error):
    with pytest.raises(error):
        processor().process(name, content)

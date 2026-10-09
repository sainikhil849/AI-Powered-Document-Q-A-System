"""Upload pipeline: validate -> extract -> chunk -> embed -> store.

Duplicate policy: the document ID is a hash of the file bytes. Uploading
identical content again does not re-embed anything; the existing document is
marked as the most recent upload (so it becomes the default for /ask) and the
response status is ``duplicate``. Different files are stored side by side.

Atomicity: all embeddings are computed before anything is written, so an
embedding failure leaves the store untouched; a failed write is cleaned up.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel

from app.core.models import FileType
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import ChromaVectorStore
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class IngestionResult(BaseModel):
    document_id: str
    filename: str
    file_type: FileType
    page_count: int | None
    chunk_count: int
    status: Literal["indexed", "duplicate"]
    message: str
    uploaded_at: datetime


class IngestionService:
    def __init__(self, processor: DocumentProcessor, embeddings: EmbeddingService, store: ChromaVectorStore) -> None:
        self.processor = processor
        self.embeddings = embeddings
        self.store = store
        self._lock = threading.Lock()  # serialises the check-then-write for duplicates

    def ingest(self, filename: str | None, content: bytes) -> IngestionResult:
        document = self.processor.process(filename, content)
        now = datetime.now(UTC)
        with self._lock:
            existing = self.store.get_document(document.document_id)
            if existing is not None:
                self.store.mark_uploaded(document.document_id, now)
                logger.info("Duplicate upload of %s; marked as latest", document.document_id)
                return IngestionResult(
                    document_id=existing.document_id,
                    filename=existing.filename,
                    file_type=existing.file_type,
                    page_count=existing.page_count,
                    chunk_count=existing.chunk_count,
                    status="duplicate",
                    message="Identical content is already indexed; it is now the latest document.",
                    uploaded_at=now,
                )
            vectors = self.embeddings.embed_documents([chunk.text for chunk in document.chunks])
            self.store.add_document(document, vectors, now)

        logger.info("Indexed document %s with %d chunks", document.document_id, len(document.chunks))
        message = "Document indexed successfully."
        if document.skipped_pages:
            message += f" Text could not be extracted from page(s) {document.skipped_pages}."
        return IngestionResult(
            document_id=document.document_id,
            filename=document.filename,
            file_type=document.file_type,
            page_count=document.page_count,
            chunk_count=len(document.chunks),
            status="indexed",
            message=message,
            uploaded_at=now,
        )

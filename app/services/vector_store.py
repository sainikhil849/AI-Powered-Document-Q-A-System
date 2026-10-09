"""Persistent Chroma vector store.

Design decisions:

* ``chromadb.PersistentClient`` writes to ``VECTOR_STORE_PATH`` on local disk,
  so indexed documents survive application restarts.
* The collection uses **cosine** distance; ``similarity = 1 - distance``.
* Embeddings are always computed by ``EmbeddingService`` and passed in
  explicitly, so Chroma's own default embedding function is never used and the
  same model is guaranteed for indexing and querying.
* Chroma itself is the single source of truth for which documents exist: the
  document list is rebuilt from chunk metadata, never from an in-memory list.
* Queries can be limited to one or more documents with a ``where`` filter
  (``document_id`` equals / ``$in``); without a filter they search everything.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.exceptions import EmbeddingDimensionMismatchError, VectorStoreError
from app.core.models import DocumentInfo, ProcessedDocument, RetrievedChunk
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


def _is_dimension_error(exc: Exception) -> bool:
    return "dimension" in str(exc).lower()


class ChromaVectorStore:
    def __init__(self, path: Path, collection_name: str, embedding_model: str) -> None:
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        try:
            path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(path),
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                configuration={"hnsw": {"space": "cosine"}},
                metadata={"embedding_model": embedding_model},
                embedding_function=None,
            )
        except Exception as exc:
            logger.exception("Could not open the Chroma vector store")
            raise VectorStoreError() from exc
        logger.info("Vector store ready: collection=%s chunks=%d", collection_name, self.count())

    # ------------------------------------------------------------------ writes
    def add_document(self, document: ProcessedDocument, embeddings: list[list[float]], uploaded_at: datetime) -> None:
        if len(embeddings) != len(document.chunks):
            raise VectorStoreError("Embedding count does not match chunk count.")
        ids, texts, metadatas = [], [], []
        for chunk in document.chunks:
            metadata: dict[str, Any] = {
                "document_id": document.document_id,
                "filename": document.filename,
                "file_type": document.file_type,
                "chunk_index": chunk.chunk_index,
                "total_chunks": len(document.chunks),
                "uploaded_at": uploaded_at.timestamp(),
                "embedding_model": self.embedding_model,
            }
            # Chroma metadata cannot hold None, so unknown values are omitted, never faked.
            if document.page_count is not None:
                metadata["page_count"] = document.page_count
            if chunk.page_start is not None and chunk.page_end is not None:
                metadata["page_start"] = chunk.page_start
                metadata["page_end"] = chunk.page_end
            if chunk.char_start is not None and chunk.char_end is not None:
                metadata["char_start"] = chunk.char_start
                metadata["char_end"] = chunk.char_end
            ids.append(chunk.chunk_id)
            texts.append(chunk.text)
            metadatas.append(metadata)
        try:
            # upsert keeps re-indexing of the same content idempotent.
            self._collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        except Exception as exc:
            logger.error("Chroma upsert failed for %s: %s", document.document_id, type(exc).__name__)
            self._safe_delete(document.document_id)  # never leave a half-written document
            if _is_dimension_error(exc):
                raise EmbeddingDimensionMismatchError() from exc
            raise VectorStoreError() from exc

    def mark_uploaded(self, document_id: str, uploaded_at: datetime) -> None:
        """Refresh the upload time of an already-indexed document (duplicate upload)."""
        try:
            result = self._collection.get(where={"document_id": document_id}, include=["metadatas"])
            metadatas = [dict(m or {}) for m in result["metadatas"] or []]
            for metadata in metadatas:
                metadata["uploaded_at"] = uploaded_at.timestamp()
            if result["ids"]:
                self._collection.update(ids=result["ids"], metadatas=metadatas)
        except Exception as exc:
            raise VectorStoreError() from exc

    def delete_document(self, document_id: str) -> None:
        try:
            self._collection.delete(where={"document_id": document_id})
        except Exception as exc:
            raise VectorStoreError() from exc

    def _safe_delete(self, document_id: str) -> None:
        try:
            self._collection.delete(where={"document_id": document_id})
        except Exception:
            logger.warning("Cleanup after a failed write also failed for %s", document_id)

    # ------------------------------------------------------------------ reads
    def count(self) -> int:
        try:
            return self._collection.count()
        except Exception as exc:
            raise VectorStoreError() from exc

    def _aggregate(self, metadatas: list[dict[str, Any]]) -> list[DocumentInfo]:
        documents: dict[str, DocumentInfo] = {}
        for metadata in metadatas:
            doc_id = str(metadata.get("document_id", ""))
            if not doc_id:
                continue
            uploaded = datetime.fromtimestamp(float(metadata.get("uploaded_at", 0.0)), tz=UTC)
            info = documents.get(doc_id)
            if info is None:
                page_count = metadata.get("page_count")
                documents[doc_id] = DocumentInfo(
                    document_id=doc_id,
                    filename=str(metadata.get("filename", "unknown")),
                    file_type=metadata.get("file_type", "txt"),
                    page_count=int(page_count) if page_count is not None else None,
                    chunk_count=1,
                    uploaded_at=uploaded,
                    embedding_model=str(metadata.get("embedding_model", self.embedding_model)),
                )
            else:
                info.chunk_count += 1
                if uploaded > info.uploaded_at:
                    info.uploaded_at = uploaded
        return sorted(documents.values(), key=lambda d: d.uploaded_at, reverse=True)

    def list_documents(self) -> list[DocumentInfo]:
        """All indexed documents, newest upload first (rebuilt from persisted metadata)."""
        try:
            result = self._collection.get(include=["metadatas"])
        except Exception as exc:
            raise VectorStoreError() from exc
        return self._aggregate(result["metadatas"] or [])

    def get_document(self, document_id: str) -> DocumentInfo | None:
        try:
            result = self._collection.get(where={"document_id": document_id}, include=["metadatas"])
        except Exception as exc:
            raise VectorStoreError() from exc
        documents = self._aggregate(result["metadatas"] or [])
        return documents[0] if documents else None

    def latest_document(self) -> DocumentInfo | None:
        documents = self.list_documents()
        return documents[0] if documents else None

    @staticmethod
    def _where(document_ids: list[str] | None) -> dict[str, Any] | None:
        if document_ids is None:
            return None  # all documents
        if len(document_ids) == 1:
            return {"document_id": document_ids[0]}
        return {"document_id": {"$in": list(document_ids)}}

    @staticmethod
    def _to_chunk(chunk_id: str, text: str | None, metadata: dict[str, Any] | None, similarity: float | None) -> RetrievedChunk:
        metadata = metadata or {}
        page_start = metadata.get("page_start")
        page_end = metadata.get("page_end")
        return RetrievedChunk(
            chunk_id=chunk_id,
            document_id=str(metadata.get("document_id", "")),
            filename=str(metadata.get("filename", "unknown")),
            chunk_index=int(metadata.get("chunk_index", -1)),
            text=text or "",
            page_start=int(page_start) if page_start is not None else None,
            page_end=int(page_end) if page_end is not None else None,
            similarity=similarity,
        )

    def query(self, embedding: list[float], top_k: int, document_ids: list[str] | None) -> list[RetrievedChunk]:
        """Nearest chunks, most similar first.

        ``document_ids=None`` searches every document; otherwise only the listed
        documents (``where`` filter applied inside Chroma's search).
        """
        if document_ids is not None and not document_ids:
            return []
        try:
            result = self._collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                where=self._where(document_ids),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.error("Chroma query failed: %s", type(exc).__name__)
            if _is_dimension_error(exc):
                raise EmbeddingDimensionMismatchError() from exc
            raise VectorStoreError() from exc

        allowed = set(document_ids) if document_ids is not None else None
        chunks: list[RetrievedChunk] = []
        ids = (result.get("ids") or [[]])[0]
        texts = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for chunk_id, text, metadata, distance in zip(ids, texts, metadatas, distances, strict=False):
            chunk = self._to_chunk(chunk_id, text, metadata, round(1.0 - float(distance), 4))
            if allowed is not None and chunk.document_id not in allowed:  # defence in depth
                continue
            chunks.append(chunk)
        chunks.sort(key=lambda c: c.similarity or 0.0, reverse=True)
        return chunks

    def best_similarity(self, document_id: str, embedding: list[float]) -> float | None:
        """Relevance of a whole document = similarity of its single best-matching chunk."""
        best = self.query(embedding, 1, [document_id])
        return best[0].similarity if best else None

    def get_document_chunks(self, document_id: str) -> list[RetrievedChunk]:
        """All chunks of one document in reading order (no similarity)."""
        try:
            result = self._collection.get(where={"document_id": document_id}, include=["documents", "metadatas"])
        except Exception as exc:
            raise VectorStoreError() from exc
        chunks = [
            self._to_chunk(chunk_id, text, metadata, None)
            for chunk_id, text, metadata in zip(result["ids"], result["documents"] or [], result["metadatas"] or [], strict=False)
        ]
        return sorted(chunks, key=lambda c: c.chunk_index)

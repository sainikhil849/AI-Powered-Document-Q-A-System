"""Document selection and similarity retrieval.

How relevance is calculated
---------------------------
1. The question is embedded with nomic-embed-text (``search_query:`` prefix).
2. **Chunk relevance** = cosine similarity between the question vector and a
   chunk vector, reported as ``similarity = 1 - cosine distance`` (higher =
   closer in meaning). Chroma returns the ``TOP_K`` most similar chunks.
3. **Document relevance** = the similarity of that document's single best
   chunk. It is shown in ``document_selection.candidates`` so you can see why
   a document was (or was not) used.
4. Similarity only ranks. Whether the passages really answer the question is
   decided afterwards by the LLM evidence evaluator; ``MIN_SIMILARITY``
   (default 0.0 = off) is an optional floor.

Which documents are searched (``select_documents``)
----------------------------------------------------
* ``document_id`` given             -> only that document.
* only one document indexed         -> that document.
* ``DEFAULT_DOCUMENT_SCOPE=latest`` -> the most recently uploaded document.
* otherwise (``auto``, the default):
  - fact question  -> if it names a document's filename (all its words, in order), that document;
                      else **all documents**, and the best chunks win wherever they are;
  - document-level question ("what is this pdf about?") -> ONE document:
                      the one named in the question, else the best-scoring one
                      if the question has topic words found in it, else the latest upload.
"""

from __future__ import annotations

from typing import Literal

from app.core.exceptions import (
    AppError,
    DocumentNotFoundError,
    NoDocumentsError,
    RetrievalError,
    VectorStoreError,
)
from app.core.models import DocumentCandidate, DocumentInfo, DocumentSelection, RetrievedChunk
from app.services.embedding_service import EmbeddingService
from app.services.question_analysis import filename_stems, mentions_filename, stem, vocabulary
from app.services.vector_store import ChromaVectorStore
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

QuestionType = Literal["fact", "summary"]


class RetrievalService:
    def __init__(
        self,
        embeddings: EmbeddingService,
        store: ChromaVectorStore,
        *,
        top_k: int,
        min_similarity: float = 0.0,
    ) -> None:
        self.embeddings = embeddings
        self.store = store
        self.top_k = top_k
        self.min_similarity = min_similarity

    # ------------------------------------------------------------------ document selection
    def score_documents(self, question: str, documents: list[DocumentInfo]) -> list[DocumentCandidate]:
        """Relevance of each document = cosine similarity of its best chunk to the question."""
        vector = self.embeddings.embed_query(question)
        candidates = []
        for doc in documents:
            try:
                score = self.store.best_similarity(doc.document_id, vector)
            except VectorStoreError as exc:
                raise RetrievalError() from exc
            candidates.append(DocumentCandidate(document_id=doc.document_id, filename=doc.filename, score=score))
        return sorted(candidates, key=lambda c: c.score if c.score is not None else -2.0, reverse=True)

    def _vocabulary(self, document: DocumentInfo) -> set[str]:
        """Word stems of a document's text plus its filename (a question may refer to either)."""
        texts = [c.text for c in self.store.get_document_chunks(document.document_id)]
        return vocabulary(texts) | filename_stems(document.filename)

    def select_documents(
        self,
        question: str,
        requested_document_id: str | None,
        *,
        summary_cue: bool,
        topic_words: list[str],
        scope: Literal["auto", "latest"] = "auto",
    ) -> tuple[DocumentSelection, QuestionType]:
        """Decide which documents to search and whether this is a document-level question."""
        if requested_document_id:
            document = self.store.get_document(requested_document_id)
            if document is None:
                raise DocumentNotFoundError(f"No document exists with document_id '{requested_document_id}'.")
            pool = [document]
            mode, reason = "explicit", "document_id was given in the request"
        else:
            pool = self.store.list_documents()  # newest first
            if not pool:
                raise NoDocumentsError()
            if len(pool) == 1:
                mode, reason = "only_document", "only one document is indexed"
            elif scope == "latest":
                pool = [pool[0]]
                mode, reason = "latest", "DEFAULT_DOCUMENT_SCOPE=latest, so the most recently uploaded document is used"
            else:
                mode, reason = "relevance", ""

        candidates = self.score_documents(question, pool)
        by_id = {doc.document_id: doc for doc in pool}
        question_stems = {stem(w) for w in topic_words}
        named = [c for c in candidates if mentions_filename(question, by_id[c.document_id].filename)] if len(pool) > 1 else []

        # --- document-level question: describe exactly one document -------------------
        if summary_cue:
            if len(pool) == 1:
                target = pool[0]
            elif named:
                target = by_id[named[0].document_id]
                mode, reason = "filename_match", f"the question mentions the file '{target.filename}'"
            else:
                target = None
                if question_stems:
                    for candidate in candidates:  # best score first
                        if self._vocabulary(by_id[candidate.document_id]) & question_stems:
                            target = by_id[candidate.document_id]
                            mode = "relevance"
                            reason = f"'{target.filename}' has the highest relevance score ({candidate.score}) among documents containing the question's topic words"
                            break
                if target is None:
                    target = pool[0]
                    mode = "latest"
                    reason = "the question names no topic or file, so the most recently uploaded document is described"

            vocab = self._vocabulary(target)
            focused = [w for w in topic_words if stem(w) in vocab and stem(w) not in filename_stems(target.filename)]
            if not focused:
                return DocumentSelection(mode=mode, reason=reason, document_ids=[target.document_id], candidates=candidates), "summary"
            logger.info("Summary wording with focused topic words %s: answering as a fact question", focused)
            if mode in ("relevance", "filename_match", "latest") and len(pool) > 1:
                mode, reason = "relevance", ""  # re-decided below for a fact question

        # --- fact question -------------------------------------------------------------
        if mode in ("explicit", "only_document") or (mode == "latest" and len(pool) == 1):
            return DocumentSelection(mode=mode, reason=reason, document_ids=[d.document_id for d in pool], candidates=candidates), "fact"
        if named:
            ids = [c.document_id for c in named]
            names = ", ".join(by_id[i].filename for i in ids)
            return DocumentSelection(mode="filename_match", reason=f"the question mentions the file(s): {names}", document_ids=ids, candidates=candidates), "fact"
        best = candidates[0]
        reason = (
            f"no document_id given: all {len(pool)} documents are searched and passages are ranked by cosine "
            f"similarity; best match is '{best.filename}' (score {best.score})"
        )
        return DocumentSelection(mode="relevance", reason=reason, document_ids=None, candidates=candidates), "fact"

    # ------------------------------------------------------------------ retrieval
    def retrieve(self, query: str, document_ids: list[str] | None) -> tuple[list[RetrievedChunk], int]:
        """Top-k chunks for ``query`` (``document_ids=None`` = all documents).

        Returns (chunks that pass the similarity floor, number dropped by the floor).
        Embedding errors keep their own type; vector store errors become RetrievalError.
        """
        vector = self.embeddings.embed_query(query)
        try:
            candidates = self.store.query(vector, self.top_k, document_ids)
        except VectorStoreError as exc:
            raise RetrievalError() from exc
        except AppError:
            raise
        except Exception as exc:  # unexpected failure inside the store layer
            raise RetrievalError() from exc

        kept = [
            c.model_copy(update={"retrieved_by_query": query})
            for c in candidates
            if c.similarity is not None and c.similarity >= self.min_similarity
        ]
        dropped = len(candidates) - len(kept)
        logger.info(
            "Retrieved %d/%d chunks from %s (top similarity=%s, dropped by floor=%d)",
            len(kept),
            len(candidates),
            "all documents" if document_ids is None else f"{len(document_ids)} document(s)",
            candidates[0].similarity if candidates else None,
            dropped,
        )
        return kept, dropped

    def overview_chunks(self, document_id: str, max_chunks: int) -> tuple[list[RetrievedChunk], int]:
        """Passages for a document-level question: evenly spread, in reading order.

        Returns (selected chunks, total chunks in the document). Short documents
        are used completely; long ones are sampled from start to end so the
        description covers the whole document rather than one part of it.
        """
        try:
            chunks = self.store.get_document_chunks(document_id)
        except VectorStoreError as exc:
            raise RetrievalError() from exc
        total = len(chunks)
        if total <= max_chunks:
            return chunks, total
        positions = sorted({round(i * (total - 1) / (max_chunks - 1)) for i in range(max_chunks)})
        return [chunks[p] for p in positions], total

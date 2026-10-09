"""Embedding generation through LangChain's ``OllamaEmbeddings``.

The service adds three things on top of the raw LangChain embedder:

* the task prefixes that nomic-embed-text expects (``search_document:`` for
  chunks, ``search_query:`` for questions) - both sides use the same model;
* validation of the returned vectors (count, dimension, finite numbers);
* translation of client errors into stable application errors.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Sequence

from langchain_core.embeddings import Embeddings

from app.core.config import Settings
from app.core.exceptions import EmbeddingDimensionMismatchError, EmbeddingError
from app.services.ollama_utils import translate_ollama_error
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


def build_ollama_embeddings(settings: Settings) -> Embeddings:
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
        client_kwargs={"timeout": settings.embedding_timeout_seconds},
    )


class EmbeddingService:
    def __init__(
        self,
        embedder: Embeddings,
        *,
        model_name: str,
        document_prefix: str = "",
        query_prefix: str = "",
        batch_size: int = 32,
    ) -> None:
        self._embedder = embedder
        self.model_name = model_name
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self.dimension: int | None = None
        # Small cache: the same question is embedded for document scoring and for retrieval.
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._query_cache_size = 64

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [self.document_prefix + text for text in texts[start : start + self.batch_size]]
            try:
                result = self._embedder.embed_documents(batch)
            except Exception as exc:
                logger.error("Document embedding failed (%s): %s", self.model_name, type(exc).__name__)
                raise translate_ollama_error(exc, model=self.model_name, operation="embedding") from exc
            vectors.extend(self._validate(result, expected=len(batch)))
        logger.info("Embedded %d chunks with %s (dimension=%s)", len(vectors), self.model_name, self.dimension)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        if text in self._query_cache:
            self._query_cache.move_to_end(text)
            return self._query_cache[text]
        try:
            vector = self._embedder.embed_query(self.query_prefix + text)
        except Exception as exc:
            logger.error("Query embedding failed (%s): %s", self.model_name, type(exc).__name__)
            raise translate_ollama_error(exc, model=self.model_name, operation="embedding") from exc
        vector = self._validate([vector], expected=1)[0]
        self._query_cache[text] = vector
        if len(self._query_cache) > self._query_cache_size:
            self._query_cache.popitem(last=False)
        return vector

    def _validate(self, vectors: object, *, expected: int) -> list[list[float]]:
        if not isinstance(vectors, list) or len(vectors) != expected:
            raise EmbeddingError("The embedding model returned an unexpected number of vectors.")
        cleaned: list[list[float]] = []
        for vector in vectors:
            if not isinstance(vector, (list, tuple)) or not vector:
                raise EmbeddingError("The embedding model returned an empty vector.")
            values = [float(v) for v in vector]
            if not all(math.isfinite(v) for v in values):
                raise EmbeddingError("The embedding model returned non-numeric values.")
            if self.dimension is None:
                self.dimension = len(values)
            elif len(values) != self.dimension:
                raise EmbeddingDimensionMismatchError(
                    "The embedding model returned vectors of inconsistent dimension."
                )
            cleaned.append(values)
        return cleaned

"""Domain models shared by the services and the LangGraph workflow.

These are internal data structures. The public API request/response models
live in ``app/api/schemas.py``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

FileType = Literal["pdf", "txt"]


class ChunkRecord(BaseModel):
    """One chunk of a processed document, ready to be embedded and stored."""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    # One-based page range for PDFs. None for TXT files or when the source page
    # could not be established. A chunk crossing a page break has start < end.
    page_start: int | None = None
    page_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None

    @property
    def pages(self) -> list[int]:
        if self.page_start is None or self.page_end is None:
            return []
        return list(range(self.page_start, self.page_end + 1))


class ProcessedDocument(BaseModel):
    document_id: str
    filename: str
    file_type: FileType
    page_count: int | None
    char_count: int
    encoding: str | None = None
    chunks: list[ChunkRecord]
    skipped_pages: list[int] = Field(default_factory=list)


class DocumentInfo(BaseModel):
    """A document as discovered from the persistent vector store."""

    document_id: str
    filename: str
    file_type: FileType
    page_count: int | None
    chunk_count: int
    uploaded_at: datetime
    embedding_model: str


class RetrievedChunk(BaseModel):
    """A chunk returned by similarity search, with its ranking signal."""

    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    text: str
    page_start: int | None = None
    page_end: int | None = None
    # Cosine similarity (1 - cosine distance). A ranking signal, not proof of an answer.
    # None for passages selected by position (document overview), not by similarity.
    similarity: float | None = None
    retrieved_by_query: str = ""

    @property
    def pages(self) -> list[int]:
        if self.page_start is None or self.page_end is None:
            return []
        return list(range(self.page_start, self.page_end + 1))

    @property
    def single_page(self) -> int | None:
        """The page number only when the chunk lies entirely on one known page."""
        if self.page_start is not None and self.page_start == self.page_end:
            return self.page_start
        return None


class EvidenceEvaluation(BaseModel):
    """Result of the evaluate_evidence step (relevance + sufficiency)."""

    relevant_chunk_ids: list[str] = Field(default_factory=list)
    sufficient: bool = False
    missing_information: str = ""
    reasoning: str = ""
    # "llm" = structured LLM judgement, "similarity_floor" = short-circuited
    # because no chunk passed MIN_SIMILARITY, "no_chunks", or "malformed_output"
    # when the evaluator's JSON was invalid (treated conservatively).
    # "overview" = document-level question: passages chosen by position, not judged by the LLM.
    method: Literal["llm", "similarity_floor", "no_chunks", "malformed_output", "overview"] = "llm"
    top_similarity: float | None = None


class DocumentCandidate(BaseModel):
    """One document and how relevant it is to the question.

    score = the highest cosine similarity between the question embedding and any
    chunk of that document (None when the document was chosen without scoring).
    """

    document_id: str
    filename: str
    score: float | None = None


class DocumentSelection(BaseModel):
    """Which document(s) a question is answered from, and why."""

    # explicit       - document_id was given in the request
    # only_document  - only one document is indexed
    # latest         - DEFAULT_DOCUMENT_SCOPE=latest, or a generic summary question
    # filename_match - the question names a document's filename
    # relevance      - chosen by similarity score (summary) or searched across all documents (fact)
    mode: Literal["explicit", "only_document", "latest", "filename_match", "relevance"]
    reason: str
    # None = search across ALL documents; otherwise only these documents.
    document_ids: list[str] | None
    candidates: list[DocumentCandidate] = Field(default_factory=list)


class GeneratedAnswer(BaseModel):
    found: bool
    answer: str = ""
    cited_chunk_ids: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    method: Literal["llm", "malformed_output"] = "llm"


class GroundingResult(BaseModel):
    grounded: bool
    issues: list[str] = Field(default_factory=list)
    method: Literal["deterministic", "llm", "malformed_output"] = "deterministic"

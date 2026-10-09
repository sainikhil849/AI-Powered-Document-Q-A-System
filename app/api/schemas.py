"""Public request and response models (these define the Swagger/OpenAPI schema)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

DOCUMENT_ID_PATTERN = r"^[a-f0-9]{16}$"


# --- Health ---------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class OllamaDependency(BaseModel):
    reachable: bool
    base_url: str
    llm_model: str
    llm_model_available: bool
    embedding_model: str
    embedding_model_available: bool
    installed_models: list[str]
    hints: list[str]


class VectorStoreDependency(BaseModel):
    ok: bool
    collection: str
    document_count: int | None
    chunk_count: int | None


class DependenciesResponse(BaseModel):
    status: Literal["ok", "degraded"]
    ollama: OllamaDependency
    vector_store: VectorStoreDependency


# --- Documents --------------------------------------------------------------------
class UploadResponse(BaseModel):
    document_id: str = Field(..., description="Stable content-based ID (first 16 hex chars of SHA-256)")
    filename: str
    file_type: Literal["pdf", "txt"]
    page_count: int | None = Field(None, description="Number of PDF pages; null for TXT files")
    chunk_count: int
    status: Literal["indexed", "duplicate"]
    message: str
    uploaded_at: datetime

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": "3f9a1c2b7d4e8f60",
                "filename": "employee_handbook.pdf",
                "file_type": "pdf",
                "page_count": 3,
                "chunk_count": 9,
                "status": "indexed",
                "message": "Document indexed successfully.",
                "uploaded_at": "2026-10-09T09:30:00Z",
            }
        }
    )


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    file_type: Literal["pdf", "txt"]
    page_count: int | None
    chunk_count: int
    uploaded_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]
    latest_document_id: str | None = Field(None, description="Default document used by /ask when document_id is omitted")


# --- Questions ----------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="Question about the uploaded document(s)")
    document_id: str | None = Field(
        None,
        pattern=DOCUMENT_ID_PATTERN,
        description="Optional. Leave it out and the system chooses the relevant document(s) itself "
        "(see document_selection in the response). Give it to search only that document.",
    )
    include_trace: bool = Field(False, description="Include the LangGraph execution trace in the response")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {"question": "How many days of annual leave do employees get?"}
        },
    )

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("question must contain non-whitespace text")
        return value


class SourceItem(BaseModel):
    document_id: str
    filename: str
    page: int | None = Field(
        None,
        description="One-based page number when the passage lies on a single PDF page; "
        "null for TXT files or when the passage spans pages (see 'pages').",
    )
    pages: list[int] = Field(default_factory=list, description="All one-based pages the passage spans; empty for TXT")
    chunk_index: int
    content: str = Field(..., description="Exact stored text of the cited passage")
    relevance_score: float | None = Field(
        None,
        description="Cosine similarity between the question and this passage (1 - cosine distance; higher = closer). "
        "A ranking signal, not a probability. null for passages chosen by position for a document overview.",
    )


class WorkflowTrace(BaseModel):
    path: list[str]
    queries: list[str]
    retries_used: int
    generation_attempts: int
    evidence_method: str | None
    evidence_reasoning: str | None
    grounding_method: str | None
    abstain_reason: str | None


class DocumentCandidateItem(BaseModel):
    document_id: str
    filename: str
    relevance_score: float | None = Field(
        None, description="Highest cosine similarity between the question and any passage of this document"
    )


class DocumentSelectionInfo(BaseModel):
    mode: Literal["explicit", "only_document", "latest", "filename_match", "relevance"] = Field(
        ...,
        description="explicit = document_id given; only_document = one document indexed; latest = most recent upload; "
        "filename_match = the question names the file; relevance = chosen/searched by similarity",
    )
    reason: str
    searched_document_ids: list[str] | None = Field(
        None, description="Documents that were searched; null means all documents"
    )
    candidates: list[DocumentCandidateItem] = Field(
        default_factory=list, description="Every candidate document with its relevance score, best first"
    )


class AskResponse(BaseModel):
    answer: str
    found: bool
    question_type: Literal["fact", "summary"] = Field(
        "fact", description="summary = a question about the whole document (e.g. 'what is this pdf about?')"
    )
    sources: list[SourceItem]
    document_id: str | None = Field(
        None,
        description="The document the answer comes from (null if several were searched and none had the answer)",
    )
    document_selection: DocumentSelectionInfo
    trace: WorkflowTrace | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answer": "Full-time employees receive 20 days of paid annual leave per calendar year.",
                "found": True,
                "question_type": "fact",
                "sources": [
                    {
                        "document_id": "3f9a1c2b7d4e8f60",
                        "filename": "employee_handbook.pdf",
                        "page": 1,
                        "pages": [1],
                        "chunk_index": 2,
                        "content": "3. ANNUAL LEAVE\n3.1 Full-time employees receive 20 days of paid annual leave per calendar year...",
                        "relevance_score": 0.7421,
                    }
                ],
                "document_id": "3f9a1c2b7d4e8f60",
                "document_selection": {
                    "mode": "only_document",
                    "reason": "only one document is indexed",
                    "searched_document_ids": ["3f9a1c2b7d4e8f60"],
                    "candidates": [
                        {"document_id": "3f9a1c2b7d4e8f60", "filename": "employee_handbook.pdf", "relevance_score": 0.7421}
                    ],
                },
                "trace": None,
            }
        }
    )


# --- Errors -------------------------------------------------------------------------
class ErrorBody(BaseModel):
    code: str = Field(..., description="Stable machine-readable error code")
    message: str = Field(..., description="Human-readable, client-safe message")
    details: list[dict[str, Any]] | None = None
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class GraphResponse(BaseModel):
    mermaid: str

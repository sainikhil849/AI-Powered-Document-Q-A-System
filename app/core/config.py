"""Application configuration.

All runtime settings are read from environment variables (or a local ``.env``
file) through pydantic-settings, so nothing is hard-coded in the services.
Every field below maps to an upper-case environment variable of the same name,
for example ``llm_model`` <- ``LLM_MODEL``.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Task prefixes recommended by Nomic for nomic-embed-text. The model was trained
# with these instructions, so documents and queries must be embedded with them.
NOMIC_DOCUMENT_PREFIX = "search_document: "
NOMIC_QUERY_PREFIX = "search_query: "

NOT_FOUND_ANSWER = "The document does not contain sufficient information to answer this question."


def normalize_model_name(name: str) -> str:
    """Return an Ollama model reference with an explicit tag (``llama3`` -> ``llama3:latest``)."""
    name = name.strip()
    return name if ":" in name else f"{name}:latest"


def model_slug(name: str) -> str:
    """Turn a model reference into a string that is safe inside a Chroma collection name."""
    return re.sub(r"[^a-z0-9]+", "-", normalize_model_name(name).lower()).strip("-")


class Settings(BaseSettings):
    """Typed configuration with validation of every tunable value."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- General -----------------------------------------------------------
    app_name: str = "Document Q&A System"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # --- Ollama / models ---------------------------------------------------
    llm_provider: Literal["ollama"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3:latest"
    embedding_model: str = "nomic-embed-text:latest"
    llm_temperature: float = Field(0.0, ge=0.0, le=2.0)
    llm_num_ctx: int = Field(4096, ge=512, le=131072)
    llm_num_predict: int = Field(512, ge=32, le=8192)
    llm_timeout_seconds: float = Field(180.0, gt=0)
    # "json_schema" asks Ollama (>= 0.5) for schema-constrained decoding,
    # "json" only asks for syntactically valid JSON (older Ollama versions).
    llm_structured_output: Literal["json_schema", "json"] = "json_schema"
    llm_json_retries: int = Field(1, ge=0, le=3)
    embedding_timeout_seconds: float = Field(60.0, gt=0)
    embedding_batch_size: int = Field(32, ge=1, le=512)
    # None = choose automatically from the model name; "" = no prefix.
    embedding_document_prefix: str | None = None
    embedding_query_prefix: str | None = None

    # --- Storage -------------------------------------------------------------
    vector_store_path: Path = Path("./data/chroma")
    collection_prefix: str = "documents"

    # --- Ingestion -----------------------------------------------------------
    max_upload_size_mb: float = Field(10.0, gt=0, le=200)
    max_pdf_pages: int = Field(500, ge=1)
    chunk_size: int = Field(800, ge=100, le=8000)
    chunk_overlap: int = Field(120, ge=0)

    # --- Retrieval and answering ---------------------------------------------
    top_k: int = Field(4, ge=1, le=20)
    # Optional similarity floor. 0.0 disables it: similarity is then used only
    # for ranking, and relevance is decided by the LLM evaluator.
    min_similarity: float = Field(0.0, ge=-1.0, le=1.0)
    max_retries: int = Field(2, ge=0, le=5)
    max_grounding_retries: int = Field(1, ge=0, le=3)
    enable_llm_grounding_check: bool = False
    max_question_length: int = Field(1000, ge=10, le=1000)  # the API schema caps questions at 1000 chars

    # --- Document selection when /ask has no document_id ----------------------
    # "auto"   = score every document against the question and search by relevance
    #            (fact questions search across all documents; summaries pick one document).
    # "latest" = only search the most recently uploaded document.
    default_document_scope: Literal["auto", "latest"] = "auto"

    # --- Document-level questions ("what is this PDF about?") ----------------
    # Number of passages, spread evenly through the document in reading order,
    # that are used to describe a document. Larger = better coverage, longer prompts.
    summary_max_chunks: int = Field(10, ge=2, le=40)

    @field_validator("ollama_base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("OLLAMA_BASE_URL must start with http:// or https://")
        return value

    @field_validator("llm_model", "embedding_model")
    @classmethod
    def _non_empty_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model names must not be empty")
        return value.strip()

    @model_validator(mode="after")
    def _check_overlap(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    # --- Derived values ------------------------------------------------------
    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_size_mb * 1024 * 1024)

    @property
    def collection_name(self) -> str:
        """One Chroma collection per embedding model.

        Vectors from different embedding models live in different vector spaces
        (and often have different dimensions), so they must never be mixed.
        Binding the collection name to the model guarantees that indexing and
        querying always use the same model.
        """
        return f"{self.collection_prefix}-{model_slug(self.embedding_model)}"[:120]

    @property
    def resolved_document_prefix(self) -> str:
        if self.embedding_document_prefix is not None:
            return self.embedding_document_prefix
        return NOMIC_DOCUMENT_PREFIX if "nomic-embed" in self.embedding_model.lower() else ""

    @property
    def resolved_query_prefix(self) -> str:
        if self.embedding_query_prefix is not None:
            return self.embedding_query_prefix
        return NOMIC_QUERY_PREFIX if "nomic-embed" in self.embedding_model.lower() else ""


def enforce_local_only() -> None:
    """Make sure no LangChain/LangSmith tracing sends prompts or documents off the machine."""
    for key in ("LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING", "LANGSMITH_TRACING"):
        os.environ[key] = "false"
    # Chroma's anonymous product telemetry is also disabled in the vector store client.
    os.environ["ANONYMIZED_TELEMETRY"] = "False"


@lru_cache
def get_settings() -> Settings:
    return Settings()

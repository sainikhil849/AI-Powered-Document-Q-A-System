"""Dependency wiring.

All services are created once at startup and shared through
``app.state.container``. Tests build the same container with a fake LLM client
and fake embeddings, so the code under test is the real code.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.embeddings import Embeddings

from app.core.config import Settings
from app.graph.nodes import WorkflowNodes
from app.graph.workflow import QAWorkflow
from app.services.answer_service import AnswerService
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService, build_ollama_embeddings
from app.services.ingestion_service import IngestionService
from app.services.llm_service import LLMClient, OllamaChatClient
from app.services.qa_service import QuestionAnsweringService
from app.services.retrieval_service import RetrievalService
from app.services.vector_store import ChromaVectorStore


@dataclass
class ServiceContainer:
    settings: Settings
    processor: DocumentProcessor
    embeddings: EmbeddingService
    store: ChromaVectorStore
    retrieval: RetrievalService
    answers: AnswerService
    workflow: QAWorkflow
    ingestion: IngestionService
    qa: QuestionAnsweringService


def build_container(
    settings: Settings,
    *,
    llm_client: LLMClient | None = None,
    embedder: Embeddings | None = None,
) -> ServiceContainer:
    processor = DocumentProcessor(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_upload_bytes=settings.max_upload_bytes,
        max_pdf_pages=settings.max_pdf_pages,
    )
    embeddings = EmbeddingService(
        embedder or build_ollama_embeddings(settings),
        model_name=settings.embedding_model,
        document_prefix=settings.resolved_document_prefix,
        query_prefix=settings.resolved_query_prefix,
        batch_size=settings.embedding_batch_size,
    )
    store = ChromaVectorStore(settings.vector_store_path, settings.collection_name, settings.embedding_model)
    retrieval = RetrievalService(embeddings, store, top_k=settings.top_k, min_similarity=settings.min_similarity)
    answers = AnswerService(
        llm_client or OllamaChatClient(settings),
        json_retries=settings.llm_json_retries,
        enable_llm_grounding_check=settings.enable_llm_grounding_check,
    )
    nodes = WorkflowNodes(
        retrieval,
        answers,
        max_question_length=settings.max_question_length,
        max_evidence_chunks=2 * settings.top_k,
        default_document_scope=settings.default_document_scope,
        summary_max_chunks=settings.summary_max_chunks,
    )
    workflow = QAWorkflow(nodes, max_retries=settings.max_retries, max_grounding_retries=settings.max_grounding_retries)
    return ServiceContainer(
        settings=settings,
        processor=processor,
        embeddings=embeddings,
        store=store,
        retrieval=retrieval,
        answers=answers,
        workflow=workflow,
        ingestion=IngestionService(processor, embeddings, store),
        qa=QuestionAnsweringService(workflow),
    )

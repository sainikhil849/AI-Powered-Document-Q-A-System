"""LangGraph node functions.

Every node is a focused function: it reads what it needs from the state,
performs ONE step, and returns only the keys it changes plus its name in
``trace``. Nodes never raise: any failure is recorded in ``state["error"]``
and the conditional edges route the run to ``handle_error``, which ends it.
Routing decisions live in ``workflow.py``, not in the nodes.
"""

from __future__ import annotations

from typing import Any

from app.core.config import NOT_FOUND_ANSWER
from app.core.exceptions import AppError, InvalidQuestionError, WorkflowError
from app.core.models import EvidenceEvaluation, RetrievedChunk
from app.graph.state import QAState
from app.services.answer_service import AnswerService
from app.services.question_analysis import has_summary_cue, topic_words
from app.services.retrieval_service import RetrievalService
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


def _failure(node: str, exc: Exception) -> dict[str, Any]:
    if isinstance(exc, AppError):
        logger.warning("Node %s failed with %s", node, exc.code)
        error = {"code": exc.code, "message": exc.message, "node": node}
    else:
        logger.exception("Unexpected error in node %s", node)
        error = {"code": WorkflowError.code, "message": WorkflowError.message, "node": node}
    return {"error": error, "trace": [node]}


def source_payload(chunk: RetrievedChunk) -> dict[str, Any]:
    """Public source attribution built only from stored chunk text and metadata."""
    return {
        "document_id": chunk.document_id,
        "filename": chunk.filename,
        "page": chunk.single_page,
        "pages": chunk.pages,
        "chunk_index": chunk.chunk_index,
        "content": chunk.text,
        "relevance_score": chunk.similarity,
    }


def _relevant_chunks(state: QAState) -> list[RetrievedChunk]:
    evaluation = state.get("relevance_result")
    relevant_ids = set(evaluation.relevant_chunk_ids) if evaluation else set()
    return [c for c in state.get("retrieved_documents", []) if c.chunk_id in relevant_ids]


class WorkflowNodes:
    def __init__(
        self,
        retrieval: RetrievalService,
        answers: AnswerService,
        *,
        max_question_length: int = 1000,
        max_evidence_chunks: int = 8,
        default_document_scope: str = "auto",
        summary_max_chunks: int = 10,
    ) -> None:
        self.retrieval = retrieval
        self.answers = answers
        self.max_question_length = max_question_length
        self.max_evidence_chunks = max_evidence_chunks
        self.default_document_scope = default_document_scope
        self.summary_max_chunks = summary_max_chunks

    # ---------------------------------------------------------------------
    def process_question(self, state: QAState) -> dict[str, Any]:
        """Normalise whitespace and validate the question; initialise the query."""
        question = " ".join(str(state.get("original_question", "")).split())
        if not question:
            return _failure("process_question", InvalidQuestionError("The question must not be empty."))
        if len(question) > self.max_question_length:
            return _failure(
                "process_question",
                InvalidQuestionError(f"The question exceeds {self.max_question_length} characters."),
            )
        return {
            "original_question": question,
            "current_question": question,
            "summary_cue": has_summary_cue(question),
            "topic_words": topic_words(question),
            "question_type": "fact",  # final type is decided in select_documents
            "retry_count": 0,
            "trace": ["process_question"],
        }

    # ---------------------------------------------------------------------
    def select_documents(self, state: QAState) -> dict[str, Any]:
        """Choose which document(s) to search and whether the question is about a whole document."""
        try:
            selection, question_type = self.retrieval.select_documents(
                state["original_question"],
                state.get("document_id"),
                summary_cue=state.get("summary_cue", False),
                topic_words=state.get("topic_words", []),
                scope=self.default_document_scope,  # type: ignore[arg-type]
            )
        except Exception as exc:
            return _failure("select_documents", exc)
        logger.info(
            "Document selection: mode=%s type=%s scope=%s",
            selection.mode,
            question_type,
            "all" if selection.document_ids is None else len(selection.document_ids),
        )
        return {
            "document_selection": selection,
            "document_ids": selection.document_ids,
            "question_type": question_type,
            "trace": ["select_documents"],
        }

    # ---------------------------------------------------------------------
    def gather_overview(self, state: QAState) -> dict[str, Any]:
        """Document-level question: take passages spread evenly through the document.

        Relevance is by construction (all passages come from the selected document)
        and there is no single fact to find, so the LLM evidence evaluator is not
        used; the overview is still fact-checked in validate_grounding.
        """
        document_id = (state.get("document_ids") or [None])[0]
        try:
            if document_id is None:
                raise WorkflowError("No document was selected for the overview.")
            chunks, total = self.retrieval.overview_chunks(document_id, self.summary_max_chunks)
        except Exception as exc:
            return _failure("gather_overview", exc)
        evaluation = EvidenceEvaluation(
            relevant_chunk_ids=[c.chunk_id for c in chunks],
            sufficient=bool(chunks),
            method="overview",
            reasoning=f"Document-level question: {len(chunks)} of {total} passages used, spread evenly in reading order.",
            missing_information="" if chunks else "The document has no stored passages.",
        )
        return {
            "retrieved_documents": chunks,
            "relevance_result": evaluation,
            "evidence_sufficient": evaluation.sufficient,
            "trace": ["gather_overview"],
        }

    # ---------------------------------------------------------------------
    def retrieve_documents(self, state: QAState) -> dict[str, Any]:
        """Similarity search (selected documents, or all documents) with the current query."""
        query = state["current_question"]
        try:
            chunks, dropped = self.retrieval.retrieve(query, state.get("document_ids"))
        except Exception as exc:
            return _failure("retrieve_documents", exc)

        # Keep chunks that an earlier attempt already judged relevant, so evidence
        # found by different queries can be combined for a multi-part question.
        new_ids = {c.chunk_id for c in chunks}
        carried = [c for c in _relevant_chunks(state) if c.chunk_id not in new_ids]
        merged = (chunks + carried)[: self.max_evidence_chunks]
        return {
            "retrieved_documents": merged,
            "chunks_below_floor": dropped,
            "query_history": [query],
            "trace": ["retrieve_documents"],
        }

    # ---------------------------------------------------------------------
    def evaluate_evidence(self, state: QAState) -> dict[str, Any]:
        """Judge relevance and sufficiency against the ORIGINAL question."""
        try:
            evaluation = self.answers.evaluate_evidence(
                state["original_question"], state.get("retrieved_documents", [])
            )
        except Exception as exc:
            return _failure("evaluate_evidence", exc)
        if evaluation.method == "no_chunks" and state.get("chunks_below_floor"):
            evaluation = EvidenceEvaluation(
                sufficient=False,
                method="similarity_floor",
                missing_information="No passage reached the minimum similarity.",
            )
        logger.info(
            "Evidence evaluation: sufficient=%s relevant=%d method=%s attempt=%d",
            evaluation.sufficient,
            len(evaluation.relevant_chunk_ids),
            evaluation.method,
            state.get("retry_count", 0) + 1,
        )
        return {
            "relevance_result": evaluation,
            "evidence_sufficient": evaluation.sufficient,
            "trace": ["evaluate_evidence"],
        }

    # ---------------------------------------------------------------------
    def reformulate_question(self, state: QAState) -> dict[str, Any]:
        """Write a new search query with the same intent; increments retry_count."""
        evaluation = state.get("relevance_result")
        try:
            query, method = self.answers.reformulate_question(
                state["original_question"],
                list(state.get("query_history", [])),
                evaluation.missing_information if evaluation else "",
                max_length=self.max_question_length,
            )
        except Exception as exc:
            return _failure("reformulate_question", exc)
        retry_count = state.get("retry_count", 0) + 1
        logger.info("Reformulated query (retry %d, method=%s)", retry_count, method)
        return {
            "current_question": query,
            "retry_count": retry_count,
            "reformulation_method": method,
            "trace": ["reformulate_question"],
        }

    # ---------------------------------------------------------------------
    def generate_answer(self, state: QAState) -> dict[str, Any]:
        """Answer the ORIGINAL question from the relevant chunks only (or describe the document)."""
        previous = state.get("grounding_result")
        feedback = previous.issues if previous is not None and not previous.grounded else None
        try:
            answer = self.answers.generate_answer(
                state["original_question"],
                _relevant_chunks(state),
                feedback,
                mode="summary" if state.get("question_type") == "summary" else "fact",
            )
        except Exception as exc:
            return _failure("generate_answer", exc)
        return {
            "generated_answer": answer,
            "grounding_result": None,
            "generation_attempts": state.get("generation_attempts", 0) + 1,
            "trace": ["generate_answer"],
        }

    # ---------------------------------------------------------------------
    def validate_grounding(self, state: QAState) -> dict[str, Any]:
        """Check that every claim is supported by the chunks the answer cites."""
        try:
            result = self.answers.validate_grounding(
                state["original_question"], state["generated_answer"], _relevant_chunks(state)
            )
        except Exception as exc:
            return _failure("validate_grounding", exc)
        logger.info("Grounding check: grounded=%s method=%s issues=%d", result.grounded, result.method, len(result.issues))
        return {"grounding_result": result, "trace": ["validate_grounding"]}

    # --- terminal node ----------------------------------------------------
    def final_response(self, state: QAState) -> dict[str, Any]:
        """Build the answer with sources taken from stored chunks, never from model text."""
        answer = state["generated_answer"]
        by_id = {c.chunk_id: c for c in state.get("retrieved_documents", [])}
        sources = [source_payload(by_id[cid]) for cid in answer.cited_chunk_ids if cid in by_id]
        return {
            "final_response": {"answer": answer.answer, "found": True, "sources": sources},
            "source_metadata": sources,
            "abstain_reason": None,
            "trace": ["final_response"],
        }

    # --- terminal node ----------------------------------------------------
    def insufficient_response(self, state: QAState) -> dict[str, Any]:
        """Abstain with the standard answer and no sources."""
        grounding = state.get("grounding_result")
        generated = state.get("generated_answer")
        if grounding is not None and not grounding.grounded:
            reason = "grounding_failed"
        elif generated is not None and not generated.found:
            reason = "model_abstained"
        else:
            reason = "insufficient_evidence"
        logger.info("Abstaining: %s", reason)
        return {
            "final_response": {"answer": NOT_FOUND_ANSWER, "found": False, "sources": []},
            "source_metadata": [],
            "abstain_reason": reason,
            "trace": ["insufficient_response"],
        }

    # 7c (terminal) -----------------------------------------------------------
    def handle_error(self, state: QAState) -> dict[str, Any]:
        """End the run; the service layer converts state['error'] into an HTTP error."""
        error = state.get("error") or {}
        logger.warning("Workflow ended with error %s in node %s", error.get("code"), error.get("node"))
        return {"final_response": None, "trace": ["handle_error"]}

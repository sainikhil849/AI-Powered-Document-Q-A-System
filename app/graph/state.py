"""Typed state shared by every node of the question-answering graph.

LangGraph passes this dictionary from node to node. Each node returns only the
keys it changes; LangGraph merges them into the state. Most keys are simply
overwritten. ``query_history`` and ``trace`` use an ``operator.add`` reducer,
so every node *appends* to them, which gives a full audit trail of the run.
"""

import operator
from typing import Annotated, Any, TypedDict

from app.core.models import DocumentSelection, EvidenceEvaluation, GeneratedAnswer, GroundingResult, RetrievedChunk


class ErrorInfo(TypedDict):
    code: str  # stable AppError code, e.g. "OLLAMA_UNAVAILABLE"
    message: str  # client-safe message
    node: str  # node where the error happened


class QAState(TypedDict, total=False):
    # --- question -----------------------------------------------------------
    original_question: str  # never modified after process_question
    current_question: str  # the query used for the next retrieval
    question_type: str  # "fact" (look up information) or "summary" (describe the whole document)
    summary_cue: bool  # wording asks about the document as a whole
    topic_words: list[str]  # content words of the question (stop/generic words removed)

    # --- document scope ----------------------------------------------------------
    document_id: str | None  # document_id from the request; None = choose automatically
    document_ids: list[str] | None  # documents actually searched; None = all documents
    document_selection: DocumentSelection | None  # how and why they were chosen, with scores

    # --- retrieval and evidence ---------------------------------------------
    retrieved_documents: list[RetrievedChunk]
    chunks_below_floor: int  # candidates dropped by MIN_SIMILARITY in the last retrieval
    relevance_result: EvidenceEvaluation | None
    evidence_sufficient: bool
    retry_count: int  # reformulations performed so far
    max_retries: int  # maximum reformulations (default 2)
    reformulation_method: str | None

    # --- answer ---------------------------------------------------------------
    generated_answer: GeneratedAnswer | None
    grounding_result: GroundingResult | None
    generation_attempts: int  # answers generated so far
    max_grounding_retries: int  # extra generations allowed after a failed grounding check
    source_metadata: list[dict[str, Any]]  # sources actually cited by the final answer
    final_response: dict[str, Any] | None
    abstain_reason: str | None  # insufficient_evidence | model_abstained | grounding_failed

    # --- errors and audit trail -------------------------------------------------
    error: ErrorInfo | None
    query_history: Annotated[list[str], operator.add]
    trace: Annotated[list[str], operator.add]


def initial_state(
    question: str,
    document_id: str | None,
    *,
    max_retries: int,
    max_grounding_retries: int,
) -> QAState:
    return QAState(
        original_question=question,
        current_question=question,
        question_type="fact",
        summary_cue=False,
        topic_words=[],
        document_id=document_id,
        document_ids=None,
        document_selection=None,
        retrieved_documents=[],
        chunks_below_floor=0,
        relevance_result=None,
        evidence_sufficient=False,
        retry_count=0,
        max_retries=max_retries,
        reformulation_method=None,
        generated_answer=None,
        grounding_result=None,
        generation_attempts=0,
        max_grounding_retries=max_grounding_retries,
        source_metadata=[],
        final_response=None,
        abstain_reason=None,
        error=None,
        query_history=[],
        trace=[],
    )

"""Entry point for answering a question: run the graph and convert the final
state into a response or a typed application error."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from app.core.exceptions import WorkflowError, error_from_code
from app.graph.workflow import QAWorkflow
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class QAResult(BaseModel):
    answer: str
    found: bool
    question_type: str
    sources: list[dict[str, Any]]
    document_id: str | None
    document_selection: dict[str, Any]
    trace: dict[str, Any]


class QuestionAnsweringService:
    def __init__(self, workflow: QAWorkflow) -> None:
        self.workflow = workflow

    def ask(self, question: str, document_id: str | None) -> QAResult:
        state = self.workflow.run(question, document_id)

        error = state.get("error")
        if error:
            # Re-raise as the same typed error the node recorded (e.g. 404 NO_DOCUMENTS, 503 OLLAMA_UNAVAILABLE)
            raise error_from_code(error["code"], error.get("message"))
        final = state.get("final_response")
        selection = state.get("document_selection")
        if not final or selection is None:
            raise WorkflowError("The workflow finished without a response.")

        # The document the answer is about: the single searched document, or - when
        # several were searched - the document of the first cited passage.
        searched = selection.document_ids
        if searched is not None and len(searched) == 1:
            primary = searched[0]
        elif final["sources"]:
            primary = final["sources"][0]["document_id"]
        else:
            primary = None

        evaluation = state.get("relevance_result")
        grounding = state.get("grounding_result")
        trace = {
            "path": list(state.get("trace", [])),
            "queries": list(state.get("query_history", [])),
            "retries_used": state.get("retry_count", 0),
            "generation_attempts": state.get("generation_attempts", 0),
            "evidence_method": evaluation.method if evaluation else None,
            "evidence_reasoning": evaluation.reasoning if evaluation else None,
            "grounding_method": grounding.method if grounding else None,
            "abstain_reason": state.get("abstain_reason"),
        }
        return QAResult(
            answer=final["answer"],
            found=final["found"],
            question_type=state.get("question_type", "fact"),
            sources=final["sources"],
            document_id=primary,
            document_selection={
                "mode": selection.mode,
                "reason": selection.reason,
                "searched_document_ids": selection.document_ids,
                "candidates": [
                    {"document_id": c.document_id, "filename": c.filename, "relevance_score": c.score}
                    for c in selection.candidates
                ],
            },
            trace=trace,
        )

"""The LangGraph ``StateGraph`` for question answering.

    START
      -> process_question
      -> select_documents                 (which document(s)? fact or document-level question?)
           document-level ("what is this pdf about?") -> gather_overview -> generate_answer
           fact question                              -> retrieve_documents
      -> retrieve_documents
      -> evaluate_evidence
           sufficient                          -> generate_answer
           insufficient, retries remaining     -> reformulate_question -> retrieve_documents
           insufficient, retry limit reached   -> insufficient_response -> END
      generate_answer
           model abstained (found=false)       -> insufficient_response -> END
           otherwise                           -> validate_grounding
      validate_grounding
           grounded                            -> final_response -> END
           not grounded, corrections remaining -> generate_answer (with feedback)
           not grounded, no corrections left   -> insufficient_response -> END
      any node with state["error"]             -> handle_error -> END

Termination: retry_count only grows (one per reformulation) and is capped by
max_retries; generation_attempts only grows and is capped by
1 + max_grounding_retries. Every other edge moves forward, so every run
reaches END. LangGraph's recursion_limit is set from the same bounds as an
additional safety net.
"""

from __future__ import annotations

from typing import Any

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from app.core.exceptions import AppError, WorkflowError
from app.graph.nodes import WorkflowNodes
from app.graph.state import QAState, initial_state
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

# Node names (used by edges, tests and the trace returned to clients)
PROCESS = "process_question"
SELECT = "select_documents"
OVERVIEW = "gather_overview"
RETRIEVE = "retrieve_documents"
EVALUATE = "evaluate_evidence"
REFORMULATE = "reformulate_question"
GENERATE = "generate_answer"
VALIDATE = "validate_grounding"
FINAL = "final_response"
INSUFFICIENT = "insufficient_response"
ERROR = "handle_error"
TERMINAL_NODES = (FINAL, INSUFFICIENT, ERROR)


# ---------------------------------------------------------------------------
# Conditional-edge routers (pure functions of the state)
# ---------------------------------------------------------------------------
def route_after_process(state: QAState) -> str:
    return ERROR if state.get("error") else SELECT


def route_after_selection(state: QAState) -> str:
    if state.get("error"):
        return ERROR
    return OVERVIEW if state.get("question_type") == "summary" else RETRIEVE


def route_after_overview(state: QAState) -> str:
    if state.get("error"):
        return ERROR
    return GENERATE if state.get("evidence_sufficient") else INSUFFICIENT


def route_after_retrieval(state: QAState) -> str:
    return ERROR if state.get("error") else EVALUATE


def route_after_evaluation(state: QAState) -> str:
    if state.get("error"):
        return ERROR
    if state.get("evidence_sufficient"):
        return GENERATE
    if state.get("retry_count", 0) < state.get("max_retries", 0):
        return REFORMULATE
    return INSUFFICIENT


def route_after_reformulation(state: QAState) -> str:
    return ERROR if state.get("error") else RETRIEVE


def route_after_generation(state: QAState) -> str:
    if state.get("error"):
        return ERROR
    answer = state.get("generated_answer")
    if answer is None or not answer.found:
        return INSUFFICIENT
    return VALIDATE


def route_after_grounding(state: QAState) -> str:
    if state.get("error"):
        return ERROR
    result = state.get("grounding_result")
    if result is not None and result.grounded:
        return FINAL
    if state.get("generation_attempts", 0) <= state.get("max_grounding_retries", 0):
        return GENERATE
    return INSUFFICIENT


def build_graph(nodes: WorkflowNodes):
    graph = StateGraph(QAState)
    graph.add_node(PROCESS, nodes.process_question)
    graph.add_node(SELECT, nodes.select_documents)
    graph.add_node(OVERVIEW, nodes.gather_overview)
    graph.add_node(RETRIEVE, nodes.retrieve_documents)
    graph.add_node(EVALUATE, nodes.evaluate_evidence)
    graph.add_node(REFORMULATE, nodes.reformulate_question)
    graph.add_node(GENERATE, nodes.generate_answer)
    graph.add_node(VALIDATE, nodes.validate_grounding)
    graph.add_node(FINAL, nodes.final_response)
    graph.add_node(INSUFFICIENT, nodes.insufficient_response)
    graph.add_node(ERROR, nodes.handle_error)

    graph.add_edge(START, PROCESS)
    graph.add_conditional_edges(PROCESS, route_after_process, {SELECT: SELECT, ERROR: ERROR})
    graph.add_conditional_edges(SELECT, route_after_selection, {OVERVIEW: OVERVIEW, RETRIEVE: RETRIEVE, ERROR: ERROR})
    graph.add_conditional_edges(
        OVERVIEW, route_after_overview, {GENERATE: GENERATE, INSUFFICIENT: INSUFFICIENT, ERROR: ERROR}
    )
    graph.add_conditional_edges(RETRIEVE, route_after_retrieval, {EVALUATE: EVALUATE, ERROR: ERROR})
    graph.add_conditional_edges(
        EVALUATE,
        route_after_evaluation,
        {GENERATE: GENERATE, REFORMULATE: REFORMULATE, INSUFFICIENT: INSUFFICIENT, ERROR: ERROR},
    )
    graph.add_conditional_edges(REFORMULATE, route_after_reformulation, {RETRIEVE: RETRIEVE, ERROR: ERROR})
    graph.add_conditional_edges(
        GENERATE, route_after_generation, {VALIDATE: VALIDATE, INSUFFICIENT: INSUFFICIENT, ERROR: ERROR}
    )
    graph.add_conditional_edges(
        VALIDATE,
        route_after_grounding,
        {FINAL: FINAL, GENERATE: GENERATE, INSUFFICIENT: INSUFFICIENT, ERROR: ERROR},
    )
    for terminal in TERMINAL_NODES:
        graph.add_edge(terminal, END)
    return graph.compile()


def max_steps(max_retries: int, max_grounding_retries: int) -> int:
    """Upper bound on node executions in one run (used for recursion_limit)."""
    retrieval_rounds = 1 + max_retries
    generations = 1 + max_grounding_retries
    # process + select + (retrieve + evaluate) per round + reformulations + (generate + validate) + terminal
    return 2 + 2 * retrieval_rounds + max_retries + 2 * generations + 1


class QAWorkflow:
    def __init__(self, nodes: WorkflowNodes, *, max_retries: int, max_grounding_retries: int) -> None:
        self.max_retries = max_retries
        self.max_grounding_retries = max_grounding_retries
        self.graph = build_graph(nodes)
        self.recursion_limit = max_steps(max_retries, max_grounding_retries) + 5

    def run(self, question: str, document_id: str | None = None) -> QAState:
        state = initial_state(
            question,
            document_id,
            max_retries=self.max_retries,
            max_grounding_retries=self.max_grounding_retries,
        )
        try:
            final: dict[str, Any] = self.graph.invoke(state, config={"recursion_limit": self.recursion_limit})
        except GraphRecursionError as exc:
            logger.error("Workflow exceeded its recursion limit (%d)", self.recursion_limit)
            raise WorkflowError("The workflow did not terminate within its step limit.") from exc
        except AppError:
            raise
        except Exception as exc:
            logger.exception("LangGraph execution failed")
            raise WorkflowError() from exc
        logger.info(
            "Workflow finished: path=%s retries=%d generations=%d",
            " -> ".join(final.get("trace", [])),
            final.get("retry_count", 0),
            final.get("generation_attempts", 0),
        )
        return final  # type: ignore[return-value]

    def mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()

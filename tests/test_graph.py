"""Requirements 13-15 and 20: reformulation, retry limits, grounding failures and
termination of every LangGraph branch."""

from __future__ import annotations

import pytest

from app.core.config import NOT_FOUND_ANSWER
from app.core.exceptions import EmbeddingError, NoDocumentsError, OllamaUnavailableError, RetrievalError
from app.core.models import (
    DocumentCandidate,
    DocumentSelection,
    EvidenceEvaluation,
    GeneratedAnswer,
    GroundingResult,
    RetrievedChunk,
)
from app.graph import workflow as wf
from app.graph.nodes import WorkflowNodes
from app.graph.workflow import QAWorkflow, max_steps
from tests.conftest import upload
from tests.helpers.fakes import FakeEmbeddings, FakeLLM, parse_sources

# ---------------------------------------------------------------------------
# 13. Reformulation after irrelevant retrieval - real Chroma, real retrieval
# ---------------------------------------------------------------------------
CAFETERIA = "Cafeteria hours: staff can get lunch at any time between noon and 2 pm."
LEAVE = "Annual leave: employees receive 25 days of annual leave each year."
REFORMULATED = "annual leave days employees receive each year"


def test_reformulation_retrieves_new_evidence_with_the_reformulated_query(client_factory):
    embeddings = FakeEmbeddings()
    llm = FakeLLM(
        handlers={
            # sufficient only when the retrieved sources actually state the fact
            "evaluate_evidence": lambda p: {
                "relevant_sources": [label for label, text in parse_sources(p).items() if "25 days" in text],
                "reasoning": "",
                "sufficient": any("25 days" in t for t in parse_sources(p).values()),
                "missing_information": "the number of vacation days",
            },
            "reformulate_question": {"query": REFORMULATED},
            "generate_answer": {"found": True, "answer": "Employees receive 25 days of annual leave each year.", "citations": ["S1"]},
            "validate_grounding": {"unsupported_claims": [], "grounded": True},
        }
    )
    client = client_factory(llm=llm, embeddings=embeddings, top_k=1, chunk_size=100, chunk_overlap=0)
    upload(client, "policy.txt", f"{CAFETERIA}\n\n{LEAVE}".encode())
    question = "How much vacation time do staff get?"

    body = client.post("/ask", json={"question": question, "include_trace": True}).json()

    # First retrieval used the original question and found only the cafeteria chunk.
    assert embeddings.query_calls[0] == question
    first_eval = llm.calls_for("evaluate_evidence")[0].user
    assert "Cafeteria" in first_eval and "25 days" not in first_eval
    # Second retrieval used the reformulated query and found the leave chunk.
    assert embeddings.query_calls[1] == REFORMULATED
    second_eval = llm.calls_for("evaluate_evidence")[1].user
    assert "25 days" in second_eval
    # Evaluation and generation always use the ORIGINAL question.
    assert f"QUESTION:\n{question}" in second_eval
    assert f"QUESTION:\n{question}" in llm.calls_for("generate_answer")[0].user
    # The reformulation prompt saw the original question and the failed query.
    assert question in llm.calls_for("reformulate_question")[0].user

    assert body["found"] is True and "25 days" in body["answer"]
    assert body["sources"][0]["content"] == LEAVE
    assert body["trace"]["queries"] == [question, REFORMULATED]
    assert body["trace"]["retries_used"] == 1
    assert body["trace"]["path"] == [
        "process_question", "select_documents", "retrieve_documents", "evaluate_evidence", "reformulate_question",
        "retrieve_documents", "evaluate_evidence", "generate_answer", "validate_grounding", "final_response",
    ]


# ---------------------------------------------------------------------------
# Graph-level tests with stub services that drive each branch deterministically
# ---------------------------------------------------------------------------
CHUNK = RetrievedChunk(chunk_id="d:00000", document_id="d", filename="f.txt", chunk_index=0,
                       text="Employees receive 20 days of annual leave.", similarity=0.8)


class StubRetrieval:
    """Stands in for RetrievalService: one document, one chunk."""

    def __init__(self, error: Exception | None = None, question_type: str = "fact", select_error: Exception | None = None,
                 overview_error: Exception | None = None):
        self.queries: list[str] = []
        self.scopes: list = []
        self.error = error
        self.question_type = question_type
        self.select_error = select_error
        self.overview_error = overview_error

    def select_documents(self, question, requested_document_id, *, summary_cue, topic_words, scope="auto"):
        if self.select_error:
            raise self.select_error
        selection = DocumentSelection(mode="only_document", reason="stub", document_ids=["d"],
                                      candidates=[DocumentCandidate(document_id="d", filename="f.txt", score=0.8)])
        return selection, self.question_type

    def retrieve(self, query, document_ids):
        self.queries.append(query)
        self.scopes.append(document_ids)
        if self.error:
            raise self.error
        return [CHUNK.model_copy(update={"retrieved_by_query": query})], 0

    def overview_chunks(self, document_id, max_chunks):
        if self.overview_error:
            raise self.overview_error
        return [CHUNK.model_copy(update={"similarity": None})], 1


class StubAnswers:
    def __init__(self, sufficient=(True,), found=(True,), grounded=(True,), errors=None):
        self.sufficient = list(sufficient)
        self.found = list(found)
        self.grounded = list(grounded)
        self.errors = errors or {}
        self.feedback: list = []
        self.modes: list = []
        self.reformulations = 0

    @staticmethod
    def _next(values):
        return values.pop(0) if len(values) > 1 else values[0]

    def _maybe_fail(self, step):
        if step in self.errors:
            raise self.errors[step]

    def evaluate_evidence(self, question, chunks):
        self._maybe_fail("evaluate")
        ok = self._next(self.sufficient)
        return EvidenceEvaluation(relevant_chunk_ids=[c.chunk_id for c in chunks] if ok else [], sufficient=ok,
                                  missing_information="" if ok else "missing")

    def reformulate_question(self, original, previous, missing, max_length=1000):
        self._maybe_fail("reformulate")
        self.reformulations += 1
        return f"{original} (rewrite {self.reformulations})", "llm"

    def generate_answer(self, question, chunks, feedback=None, mode="fact"):
        self._maybe_fail("generate")
        self.feedback.append(feedback)
        self.modes.append(mode)
        found = self._next(self.found)
        return GeneratedAnswer(found=found, answer="Employees receive 20 days." if found else "",
                               cited_chunk_ids=[CHUNK.chunk_id] if found else [])

    def validate_grounding(self, question, answer, chunks):
        self._maybe_fail("validate")
        ok = self._next(self.grounded)
        return GroundingResult(grounded=ok, issues=[] if ok else ["unsupported claim"], method="llm")


def run(answers=None, retrieval=None, question="How many days of annual leave?", max_retries=2, max_grounding_retries=1):
    retrieval = retrieval or StubRetrieval()
    answers = answers or StubAnswers()
    workflow = QAWorkflow(WorkflowNodes(retrieval, answers), max_retries=max_retries, max_grounding_retries=max_grounding_retries)
    return workflow.run(question, None), retrieval, answers


def test_success_path():
    state, _, _ = run()
    assert state["trace"] == ["process_question", "select_documents", "retrieve_documents", "evaluate_evidence",
                              "generate_answer", "validate_grounding", "final_response"]
    assert state["question_type"] == "fact"
    assert state["final_response"]["found"] is True
    assert state["final_response"]["sources"][0]["content"] == CHUNK.text
    assert state["source_metadata"] == state["final_response"]["sources"]


# 14. retry-limit enforcement -----------------------------------------------------------
@pytest.mark.parametrize("max_retries", [0, 1, 2, 3])
def test_retry_limit_is_enforced(max_retries):
    state, retrieval, answers = run(StubAnswers(sufficient=(False,)), max_retries=max_retries)
    assert len(retrieval.queries) == 1 + max_retries  # initial retrieval + N reformulations
    assert state["retry_count"] == max_retries
    assert answers.reformulations == max_retries
    assert state["trace"].count("reformulate_question") == max_retries
    assert state["trace"][-1] == "insufficient_response"
    assert state["final_response"] == {"answer": NOT_FOUND_ANSWER, "found": False, "sources": []}
    assert state["abstain_reason"] == "insufficient_evidence"
    assert state["original_question"] == "How many days of annual leave?"  # never overwritten
    assert retrieval.queries[0] == "How many days of annual leave?"
    assert all("(rewrite" in q for q in retrieval.queries[1:])


def test_retry_succeeds_on_second_attempt():
    state, retrieval, _ = run(StubAnswers(sufficient=(False, True)))
    assert state["retry_count"] == 1
    assert retrieval.queries[1].endswith("(rewrite 1)")
    assert state["final_response"]["found"] is True


# 15. grounding validation failure --------------------------------------------------------
def test_grounding_failure_triggers_one_correction_then_abstains():
    state, _, answers = run(StubAnswers(grounded=(False,)))
    assert state["generation_attempts"] == 2  # 1 + max_grounding_retries
    assert state["trace"].count("generate_answer") == 2
    assert answers.feedback == [None, ["unsupported claim"]]  # the correction received the issues
    assert state["trace"][-1] == "insufficient_response"
    assert state["abstain_reason"] == "grounding_failed"
    assert state["final_response"]["found"] is False and state["final_response"]["sources"] == []


def test_grounding_correction_can_succeed():
    state, _, _ = run(StubAnswers(grounded=(False, True)))
    assert state["generation_attempts"] == 2
    assert state["final_response"]["found"] is True


def test_grounding_retries_disabled_abstains_immediately():
    state, _, _ = run(StubAnswers(grounded=(False,)), max_grounding_retries=0)
    assert state["generation_attempts"] == 1
    assert state["abstain_reason"] == "grounding_failed"


def test_model_abstention_is_respected():
    state, _, _ = run(StubAnswers(found=(False,)))
    assert "validate_grounding" not in state["trace"]
    assert state["abstain_reason"] == "model_abstained"
    assert state["final_response"]["found"] is False


def test_summary_path_skips_retrieval_and_evaluation():
    retrieval = StubRetrieval(question_type="summary")
    state, retrieval, answers = run(StubAnswers(), retrieval=retrieval, question="What is this pdf about?")
    assert state["trace"] == ["process_question", "select_documents", "gather_overview",
                              "generate_answer", "validate_grounding", "final_response"]
    assert retrieval.queries == []  # no similarity search for a document-level question
    assert answers.modes == ["summary"]
    assert state["relevance_result"].method == "overview"
    assert state["summary_cue"] is True and state["question_type"] == "summary"
    assert state["final_response"]["sources"][0]["relevance_score"] is None


def test_grounding_failure_end_to_end_with_number_check(client_factory, handbook_bytes):
    """A hallucinated number is caught by the deterministic check, corrected once, then abstained."""
    llm = FakeLLM(
        handlers={
            "evaluate_evidence": lambda p: {"relevant_sources": [l for l, t in parse_sources(p).items() if "20 days" in t],
                                            "reasoning": "", "sufficient": True},
            "generate_answer": {"found": True, "answer": "Employees receive 30 days of annual leave.", "citations": ["S1"]},
            "validate_grounding": {"unsupported_claims": [], "grounded": True},
            "reformulate_question": {"query": "x"},
        }
    )
    client = client_factory(llm=llm)
    upload(client, "employee_handbook.txt", handbook_bytes)
    body = client.post("/ask", json={"question": "How many days of annual leave do employees get?", "include_trace": True}).json()
    assert body["found"] is False and body["sources"] == []
    assert body["trace"]["abstain_reason"] == "grounding_failed"
    assert body["trace"]["generation_attempts"] == 2
    second_prompt = llm.calls_for("generate_answer")[1].user
    assert "30 does not appear" in second_prompt
    assert llm.calls_for("validate_grounding") == []  # deterministic check failed first


def test_llm_grader_rejection_leads_to_abstention(client_factory, handbook_bytes):
    llm = FakeLLM(
        handlers={
            "evaluate_evidence": lambda p: {"relevant_sources": ["S1"], "reasoning": "", "sufficient": True},
            "generate_answer": lambda p: {"found": True, "answer": "Leave rules exist and managers are generous.", "citations": ["S1"]},
            "validate_grounding": {"unsupported_claims": ["managers are generous"], "grounded": False},
            "reformulate_question": {"query": "x"},
        }
    )
    client = client_factory(llm=llm, enable_llm_grounding_check=True)
    upload(client, "employee_handbook.txt", handbook_bytes)
    body = client.post("/ask", json={"question": "Tell me about leave", "include_trace": True}).json()
    assert body["found"] is False
    assert body["trace"]["grounding_method"] == "llm"
    assert len(llm.calls_for("validate_grounding")) == 2


# 20. termination on every branch -------------------------------------------------------------
BRANCHES = {
    "success": dict(answers=StubAnswers()),
    "insufficient_after_retries": dict(answers=StubAnswers(sufficient=(False,))),
    "model_abstains": dict(answers=StubAnswers(found=(False,))),
    "grounding_failed": dict(answers=StubAnswers(grounded=(False,))),
    "empty_question": dict(question="   "),
    "summary_success": dict(retrieval=StubRetrieval(question_type="summary")),
    "summary_grounding_failed": dict(retrieval=StubRetrieval(question_type="summary"), answers=StubAnswers(grounded=(False,))),
    "selection_error": dict(retrieval=StubRetrieval(select_error=NoDocumentsError())),
    "overview_error": dict(retrieval=StubRetrieval(question_type="summary", overview_error=RetrievalError())),
    "retrieval_error": dict(retrieval=StubRetrieval(error=RetrievalError())),
    "embedding_error": dict(retrieval=StubRetrieval(error=EmbeddingError())),
    "evaluate_error": dict(answers=StubAnswers(errors={"evaluate": OllamaUnavailableError()})),
    "reformulate_error": dict(answers=StubAnswers(sufficient=(False,), errors={"reformulate": OllamaUnavailableError()})),
    "generate_error": dict(answers=StubAnswers(errors={"generate": OllamaUnavailableError()})),
    "validate_error": dict(answers=StubAnswers(errors={"validate": RuntimeError("boom")})),
}


@pytest.mark.parametrize("branch", list(BRANCHES))
def test_every_branch_terminates_in_exactly_one_terminal_node(branch):
    kwargs = {k: v for k, v in BRANCHES[branch].items()}
    # fresh stubs per run (lists inside are consumed)
    if "answers" in kwargs:
        src = kwargs["answers"]
        kwargs["answers"] = StubAnswers(src.sufficient, src.found, src.grounded, src.errors)
    state, _, _ = run(**kwargs)
    terminals = [node for node in state["trace"] if node in wf.TERMINAL_NODES]
    assert len(terminals) == 1 and state["trace"][-1] == terminals[0]
    assert len(state["trace"]) <= max_steps(2, 1)
    if terminals[0] == wf.ERROR:
        assert state["error"]["code"] and state["error"]["node"] in state["trace"]
        assert state["final_response"] is None
    else:
        assert state.get("error") is None
        assert state["final_response"]["answer"]


def test_recursion_limit_turns_into_workflow_error():
    from app.core.exceptions import WorkflowError

    workflow = QAWorkflow(WorkflowNodes(StubRetrieval(), StubAnswers(sufficient=(False,))), max_retries=2, max_grounding_retries=1)
    workflow.recursion_limit = 3  # simulate a runaway graph
    with pytest.raises(WorkflowError):
        workflow.run("How many days of annual leave?", None)


def test_error_node_names_the_failing_step():
    state, _, _ = run(StubAnswers(errors={"generate": OllamaUnavailableError()}))
    assert state["error"] == {"code": "OLLAMA_UNAVAILABLE", "message": OllamaUnavailableError.message, "node": "generate_answer"}


def test_routers_are_pure_functions_of_state():
    assert wf.route_after_selection({"question_type": "summary"}) == wf.OVERVIEW
    assert wf.route_after_selection({"question_type": "fact"}) == wf.RETRIEVE
    assert wf.route_after_overview({"evidence_sufficient": False}) == wf.INSUFFICIENT
    assert wf.route_after_evaluation({"evidence_sufficient": True}) == wf.GENERATE
    assert wf.route_after_evaluation({"evidence_sufficient": False, "retry_count": 0, "max_retries": 2}) == wf.REFORMULATE
    assert wf.route_after_evaluation({"evidence_sufficient": False, "retry_count": 2, "max_retries": 2}) == wf.INSUFFICIENT
    assert wf.route_after_evaluation({"error": {"code": "X"}, "evidence_sufficient": True}) == wf.ERROR
    assert wf.route_after_grounding({"grounding_result": GroundingResult(grounded=False), "generation_attempts": 2,
                                     "max_grounding_retries": 1}) == wf.INSUFFICIENT


def test_compiled_graph_has_the_required_nodes_and_conditional_edges():
    workflow = QAWorkflow(WorkflowNodes(StubRetrieval(), StubAnswers()), max_retries=2, max_grounding_retries=1)
    graph = workflow.graph.get_graph()
    nodes = set(graph.nodes)
    assert {"process_question", "select_documents", "gather_overview", "retrieve_documents", "evaluate_evidence",
            "reformulate_question", "generate_answer", "validate_grounding", "final_response",
            "insufficient_response", "handle_error"} <= nodes
    conditional = {(e.source, e.target) for e in graph.edges if e.conditional}
    assert ("select_documents", "gather_overview") in conditional
    assert ("select_documents", "retrieve_documents") in conditional
    assert ("gather_overview", "generate_answer") in conditional
    assert ("evaluate_evidence", "generate_answer") in conditional
    assert ("evaluate_evidence", "reformulate_question") in conditional
    assert ("evaluate_evidence", "insufficient_response") in conditional
    assert ("reformulate_question", "retrieve_documents") in conditional
    assert ("validate_grounding", "generate_answer") in conditional
    assert ("validate_grounding", "final_response") in conditional
    assert workflow.recursion_limit > max_steps(2, 1)

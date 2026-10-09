"""Requirements 8-12 and 16-19 through the HTTP API."""

from __future__ import annotations

import pytest

from app.core.config import NOT_FOUND_ANSWER
from app.core.exceptions import (
    EmbeddingError,
    LLMGenerationError,
    ModelNotAvailableError,
    OllamaTimeoutError,
    OllamaUnavailableError,
    VectorStoreError,
)
from tests.conftest import upload
from tests.helpers.fakes import HANDBOOK_FACTS, Fact, FakeEmbeddings, FakeLLM, oracle_llm


def ask(client, question, **extra):
    return client.post("/ask", json={"question": question, **extra})


# --- 8. supported questions -------------------------------------------------------------
def test_supported_question_returns_grounded_answer_with_real_sources(client, handbook_bytes):
    doc_id = upload(client, "employee_handbook.txt", handbook_bytes).json()["document_id"]
    response = ask(client, "How many days of annual leave do employees get per year?")
    assert response.status_code == 200
    body = response.json()
    assert body["found"] is True
    assert "20 days" in body["answer"]
    assert body["document_id"] == doc_id
    assert len(body["sources"]) >= 1
    handbook_text = handbook_bytes.decode("utf-8")
    for source in body["sources"]:
        assert source["document_id"] == doc_id
        assert source["filename"] == "employee_handbook.txt"
        assert source["page"] is None and source["pages"] == []  # TXT: no fabricated page
        assert "20 days of paid annual leave" in source["content"]
        assert source["content"] in handbook_text  # the snippet is real document text
    assert body["trace"] is None


def test_answer_from_a_different_section_with_pdf_page_number(client, handbook_pdf_bytes):
    upload(client, "employee_handbook.pdf", handbook_pdf_bytes)
    body = ask(client, "When is a medical certificate needed for sick leave?").json()
    assert body["found"] is True
    source = body["sources"][0]
    assert "longer than 3 consecutive days" in source["content"]
    assert source["page"] == 2 and source["pages"] == [2]
    assert source["filename"] == "employee_handbook.pdf"


def test_include_trace_shows_the_graph_path(client, handbook_bytes):
    upload(client, "employee_handbook.txt", handbook_bytes)
    trace = ask(client, "What are the core working hours?", include_trace=True).json()["trace"]
    assert trace["path"] == [
        "process_question", "select_documents", "retrieve_documents", "evaluate_evidence",
        "generate_answer", "validate_grounding", "final_response",
    ]
    assert trace["queries"] == ["What are the core working hours?"]
    assert trace["retries_used"] == 0


# --- 9. unsupported question -----------------------------------------------------------------
def test_unsupported_question_abstains_without_sources(client, fake_llm, handbook_bytes):
    upload(client, "employee_handbook.txt", handbook_bytes)
    response = ask(client, "How many weeks of maternity leave do employees get?", include_trace=True)
    assert response.status_code == 200
    body = response.json()
    assert body["found"] is False
    assert body["answer"] == NOT_FOUND_ANSWER
    assert body["sources"] == []
    assert body["trace"]["abstain_reason"] == "insufficient_evidence"
    assert body["trace"]["retries_used"] == 2  # tried the original + 2 reformulations
    assert fake_llm.calls_for("generate_answer") == []  # no answer was even drafted


# --- 10. invalid questions ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "    \n\t "},
        {"question": "x" * 1001},
        {},
        {"question": 123},
        {"question": "ok?", "document_id": "../../etc/passwd"},
        {"question": "ok?", "unexpected": True},
    ],
)
def test_invalid_questions_are_rejected(client, handbook_bytes, payload):
    upload(client, "employee_handbook.txt", handbook_bytes)
    response = client.post("/ask", json=payload)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "VALIDATION_ERROR"
    assert error["details"]
    assert "passwd" not in response.text  # input values are not echoed


def test_non_json_body_is_rejected(client):
    response = client.post("/ask", content=b"question=hello", headers={"content-type": "text/plain"})
    assert response.status_code == 422


# --- 11 / 12. no documents, unknown document --------------------------------------------------------
def test_question_before_any_upload(client):
    response = ask(client, "How many days of annual leave do employees get?")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NO_DOCUMENTS"


def test_unknown_document_id(client, handbook_bytes):
    upload(client, "employee_handbook.txt", handbook_bytes)
    response = ask(client, "What are the core working hours?", document_id="0123456789abcdef")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"


# --- 16. LLM failures --------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("exc", "status", "code"),
    [
        (OllamaUnavailableError(), 503, "OLLAMA_UNAVAILABLE"),
        (OllamaTimeoutError(), 504, "OLLAMA_TIMEOUT"),
        (ModelNotAvailableError("The configured language model 'x' is not installed."), 503, "MODEL_NOT_AVAILABLE"),
        (LLMGenerationError(), 502, "LLM_FAILED"),
        (RuntimeError("C:\\internal\\path exploded"), 500, "WORKFLOW_ERROR"),
    ],
)
def test_llm_failures_map_to_clear_errors(client_factory, handbook_bytes, exc, status, code):
    llm = oracle_llm()
    llm.handlers["evaluate_evidence"] = exc
    client = client_factory(llm=llm)
    upload(client, "employee_handbook.txt", handbook_bytes)
    response = ask(client, "How many days of annual leave do employees get?")
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert "internal\\path" not in response.text and "Traceback" not in response.text


def test_generation_failure_is_reported_not_hidden_as_not_found(client_factory, handbook_bytes):
    llm = oracle_llm()
    llm.handlers["generate_answer"] = OllamaUnavailableError()
    client = client_factory(llm=llm)
    upload(client, "employee_handbook.txt", handbook_bytes)
    response = ask(client, "How many days of annual leave do employees get?")
    assert response.status_code == 503


def test_malformed_llm_output_leads_to_safe_abstention(client_factory, handbook_bytes):
    llm = oracle_llm()
    llm.handlers["evaluate_evidence"] = "Sure! The evidence looks great."  # not JSON
    client = client_factory(llm=llm)
    upload(client, "employee_handbook.txt", handbook_bytes)
    body = ask(client, "How many days of annual leave do employees get?", include_trace=True).json()
    assert body["found"] is False
    assert body["sources"] == []
    assert body["trace"]["evidence_method"] == "malformed_output"


# --- 17. embedding failures ------------------------------------------------------------------------------
def test_embedding_failure_during_upload_stores_nothing(client_factory, handbook_bytes):
    embeddings = FakeEmbeddings()
    embeddings.fail_on_documents = ConnectionError("connection refused")
    client = client_factory(embeddings=embeddings)
    response = upload(client, "employee_handbook.txt", handbook_bytes)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "OLLAMA_UNAVAILABLE"
    assert client.get("/documents").json()["documents"] == []


def test_invalid_embedding_output_is_rejected(client_factory, handbook_bytes):
    class BrokenEmbeddings(FakeEmbeddings):
        def embed_documents(self, texts):
            return [[float("nan")] * 8 for _ in texts]

    client = client_factory(embeddings=BrokenEmbeddings())
    response = upload(client, "employee_handbook.txt", handbook_bytes)
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "EMBEDDING_FAILED"


def test_query_embedding_failure_during_ask(client_factory, handbook_bytes):
    embeddings = FakeEmbeddings()
    client = client_factory(embeddings=embeddings)
    upload(client, "employee_handbook.txt", handbook_bytes)
    embeddings.fail_on_query = EmbeddingError()
    response = ask(client, "How many days of annual leave do employees get?")
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "EMBEDDING_FAILED"


# --- 18. vector store / retrieval failures ----------------------------------------------------------------
def test_vector_query_failure_returns_retrieval_error(client, container, handbook_bytes, monkeypatch):
    upload(client, "employee_handbook.txt", handbook_bytes)

    def broken_query(*args, **kwargs):
        raise RuntimeError("disk I/O error at /var/lib/secret")

    monkeypatch.setattr(container.store._collection, "query", broken_query)
    response = ask(client, "How many days of annual leave do employees get?")
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "RETRIEVAL_FAILED"
    assert "secret" not in response.text


def test_vector_store_write_failure_leaves_no_partial_document(client, container, handbook_bytes, monkeypatch):
    def broken_upsert(*args, **kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(container.store._collection, "upsert", broken_upsert)
    response = upload(client, "employee_handbook.txt", handbook_bytes)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "VECTOR_STORE_UNAVAILABLE"
    assert container.store.count() == 0


def test_vector_store_read_failure_on_ask(client, container, handbook_bytes, monkeypatch):
    upload(client, "employee_handbook.txt", handbook_bytes)
    monkeypatch.setattr(container.store, "list_documents", lambda: (_ for _ in ()).throw(VectorStoreError()))
    response = ask(client, "How many days of annual leave do employees get?")
    assert response.status_code == 503


def test_unexpected_exception_returns_generic_500(client_factory, handbook_bytes, monkeypatch):
    client = client_factory(raise_server_exceptions=False)
    upload(client, "employee_handbook.txt", handbook_bytes)
    monkeypatch.setattr(client.container.qa, "ask", lambda *a, **k: 1 / 0)
    response = ask(client, "anything?")
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"
    assert "ZeroDivisionError" not in response.text


# --- 19. persistence and document isolation -------------------------------------------------------------------
POLICY_A = b"Leave policy of Alpha Corp. Employees receive 18 days of annual leave per year."
POLICY_B = b"Leave policy of Beta Corp. Employees receive 26 days of annual leave per year."


def isolation_llm() -> FakeLLM:
    return FakeLLM(
        handlers={
            "evaluate_evidence": lambda p: {"relevant_sources": ["S1"], "reasoning": "", "sufficient": True},
            "generate_answer": lambda p: {
                "found": True,
                "answer": "Employees receive " + ("18" if "18 days" in p else "26") + " days of annual leave per year.",
                "citations": ["S1"],
            },
            "validate_grounding": {"unsupported_claims": [], "grounded": True},
            "reformulate_question": {"query": "annual leave days"},
        }
    )


def test_questions_only_search_the_selected_document(client_factory):
    llm = isolation_llm()
    client = client_factory(llm=llm)
    doc_a = upload(client, "alpha.txt", POLICY_A).json()["document_id"]
    doc_b = upload(client, "beta.txt", POLICY_B).json()["document_id"]
    assert doc_a != doc_b

    body_a = ask(client, "How many days of annual leave?", document_id=doc_a).json()
    assert "18" in body_a["answer"]
    assert {s["document_id"] for s in body_a["sources"]} == {doc_a}
    prompt = llm.calls_for("evaluate_evidence")[-1].user
    assert "Alpha Corp" in prompt and "Beta Corp" not in prompt
    assert body_a["document_selection"]["mode"] == "explicit"
    assert body_a["document_selection"]["searched_document_ids"] == [doc_a]


# --- automatic document selection (no document_id) ----------------------------------------------
CAFETERIA_DOC = (
    b"CAFETERIA MENU\n\n1. OPENING HOURS\nThe cafeteria is open for lunch from noon to 2 pm.\n\n"
    b"2. WEEKLY MENU\nMonday: vegetable biryani.\nTuesday: paneer butter masala.\nWednesday: dal and rice.\n"
    b"Thursday: chicken curry.\nFriday: fish curry with rice."
)
CAFETERIA_FACT = Fact("friday", "Friday: fish curry", "On Friday the cafeteria serves fish curry with rice.")


def two_document_client(client_factory, handbook_bytes, **overrides):
    client = client_factory(llm=oracle_llm(HANDBOOK_FACTS + [CAFETERIA_FACT]), **overrides)
    cafe = upload(client, "cafeteria_menu.txt", CAFETERIA_DOC).json()["document_id"]
    handbook = upload(client, "employee_handbook.txt", handbook_bytes).json()["document_id"]  # latest upload
    return client, cafe, handbook


def test_relevant_document_is_found_without_document_id(client_factory, handbook_bytes):
    client, cafe, handbook = two_document_client(client_factory, handbook_bytes)
    body = ask(client, "What is served for lunch on Friday?").json()
    assert body["found"] is True and "fish curry" in body["answer"]
    assert body["document_id"] == cafe  # not the latest upload: chosen by relevance
    assert {s["document_id"] for s in body["sources"]} == {cafe}
    selection = body["document_selection"]
    assert selection["mode"] == "relevance"
    assert selection["searched_document_ids"] is None  # all documents were searched
    scores = {c["document_id"]: c["relevance_score"] for c in selection["candidates"]}
    assert set(scores) == {cafe, handbook}
    assert scores[cafe] > scores[handbook]
    assert [c["relevance_score"] for c in selection["candidates"]] == sorted(scores.values(), reverse=True)


def test_fact_answer_comes_from_the_document_that_contains_it(client_factory, handbook_bytes):
    client, cafe, handbook = two_document_client(client_factory, handbook_bytes)
    body = ask(client, "How many days of annual leave do employees get?").json()
    assert body["found"] is True and "20 days" in body["answer"]
    assert body["document_id"] == handbook
    assert all(s["filename"] == "employee_handbook.txt" for s in body["sources"])


def test_question_naming_a_file_searches_that_file(client_factory, handbook_bytes):
    client, cafe, handbook = two_document_client(client_factory, handbook_bytes)
    body = ask(client, "In the cafeteria menu, what is served on Friday?").json()
    assert body["document_selection"]["mode"] == "filename_match"
    assert body["document_selection"]["searched_document_ids"] == [cafe]


def test_default_scope_latest_only_searches_the_latest_upload(client_factory, handbook_bytes):
    client, cafe, handbook = two_document_client(client_factory, handbook_bytes, default_document_scope="latest")
    body = ask(client, "What is served for lunch on Friday?").json()
    assert body["document_selection"]["mode"] == "latest"
    assert body["document_selection"]["searched_document_ids"] == [handbook]
    assert body["found"] is False  # the menu is in the other document, which is not searched


# --- document-level questions ("what is this pdf about?") ----------------------------------------
@pytest.mark.parametrize(
    "question",
    [
        "Explain me what is this pdf about",
        "explianme whatbis pdf about",  # typos from a real user
        "Summarize the document",
        "Give me an overview",
        "What are the main points?",
    ],
)
def test_document_level_question_gets_an_overview(client_factory, handbook_pdf_bytes, question):
    llm = oracle_llm()
    client = client_factory(llm=llm)
    doc_id = upload(client, "employee_handbook.pdf", handbook_pdf_bytes).json()["document_id"]
    body = ask(client, question, include_trace=True).json()
    assert body["found"] is True, body
    assert body["question_type"] == "summary"
    assert body["document_id"] == doc_id
    for topic in ("Working Hours", "Annual Leave", "Sick Leave", "Remote Work"):
        assert topic in body["answer"]
    assert body["trace"]["path"] == ["process_question", "select_documents", "gather_overview",
                                     "generate_answer", "validate_grounding", "final_response"]
    assert body["trace"]["evidence_method"] == "overview"
    assert len(body["sources"]) == 7  # every passage of the 7-chunk PDF, in reading order
    assert [s["chunk_index"] for s in body["sources"]] == list(range(7))
    assert all(s["relevance_score"] is None for s in body["sources"])  # chosen by position, not similarity
    assert {p for s in body["sources"] for p in s["pages"]} == {1, 2, 3}
    assert llm.calls_for("evaluate_evidence") == []


def test_generic_overview_with_several_documents_uses_latest_upload(client_factory, handbook_bytes):
    client, cafe, handbook = two_document_client(client_factory, handbook_bytes)
    body = ask(client, "What is this pdf about?").json()
    assert body["question_type"] == "summary"
    assert body["document_selection"]["mode"] == "latest"
    assert body["document_id"] == handbook


def test_overview_of_a_named_document(client_factory, handbook_bytes):
    client, cafe, handbook = two_document_client(client_factory, handbook_bytes)
    body = ask(client, "Summarize the cafeteria menu").json()
    assert body["question_type"] == "summary"
    assert body["document_selection"]["mode"] == "filename_match"
    assert body["document_id"] == cafe
    assert "Weekly Menu" in body["answer"]


def test_overview_wording_about_a_specific_topic_is_answered_as_a_fact_question(client, handbook_bytes):
    upload(client, "employee_handbook.txt", handbook_bytes)
    body = ask(client, "Give me an overview of the remote work rules", include_trace=True).json()
    assert body["question_type"] == "fact"
    assert body["found"] is True and "2 days per week" in body["answer"]
    assert "retrieve_documents" in body["trace"]["path"]


def test_documents_persist_across_restarts(tmp_path):
    from fastapi.testclient import TestClient

    from app.core.container import build_container
    from app.main import create_app
    from tests.conftest import make_settings

    settings = make_settings(tmp_path)
    with TestClient(create_app(container=build_container(settings, llm_client=isolation_llm(), embedder=FakeEmbeddings()))) as first:
        doc_a = upload(first, "alpha.txt", POLICY_A).json()["document_id"]
        doc_b = upload(first, "beta.txt", POLICY_B).json()["document_id"]

    # A brand-new container = an application restart reading the same ./data directory.
    with TestClient(create_app(container=build_container(settings, llm_client=isolation_llm(), embedder=FakeEmbeddings()))) as second:
        listing = second.get("/documents").json()
        assert {d["document_id"] for d in listing["documents"]} == {doc_a, doc_b}
        assert listing["latest_document_id"] == doc_b
        assert ask(second, "How many days of annual leave?", document_id=doc_a).json()["found"] is True
        # Re-uploading A makes it the latest again (duplicate policy).
        assert upload(second, "alpha.txt", POLICY_A).json()["status"] == "duplicate"
        assert second.get("/documents").json()["latest_document_id"] == doc_a

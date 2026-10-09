"""End-to-end tests against a REAL local Ollama (opt-in).

Skipped unless RUN_OLLAMA_TESTS=1 and Ollama is running with the configured
models. They use the real models from your .env, a temporary vector store, and
the sample handbook. Answers come from a real LLM, so these tests check
behaviour (found / abstained, correct number present), not exact wording.

    $env:RUN_OLLAMA_TESTS = "1"; pytest -m ollama -v      # PowerShell
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.container import build_container
from app.main import create_app
from app.services.ollama_utils import check_ollama
from tests.conftest import SAMPLE_TXT, upload

pytestmark = pytest.mark.ollama

if os.getenv("RUN_OLLAMA_TESTS") != "1":
    pytest.skip("set RUN_OLLAMA_TESTS=1 to run tests against a real Ollama", allow_module_level=True)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    settings = Settings(vector_store_path=tmp_path_factory.mktemp("chroma-real"))
    status = check_ollama(settings.ollama_base_url, settings.llm_model, settings.embedding_model)
    if not (status.reachable and status.llm_model_available and status.embedding_model_available):
        pytest.skip("Ollama or a configured model is not available: " + "; ".join(status.hints))
    with TestClient(create_app(container=build_container(settings))) as test_client:
        assert upload(test_client, "employee_handbook.txt", SAMPLE_TXT.read_bytes()).status_code == 201
        yield test_client


def ask(client, question):
    response = client.post("/ask", json={"question": question, "include_trace": True})
    assert response.status_code == 200, response.text
    body = response.json()
    print(f"\nQ: {question}\nA: {body['answer']}\nfound={body['found']} path={' -> '.join(body['trace']['path'])}")
    return body


def test_direct_fact(client):
    body = ask(client, "How many days of annual leave do employees get per year?")
    assert body["found"] is True
    assert "20" in body["answer"]
    assert any("20 days" in s["content"] for s in body["sources"])


def test_fact_from_another_section(client):
    body = ask(client, "What are the core working hours?")
    assert body["found"] is True
    assert "10" in body["answer"] and "4" in body["answer"]


def test_colloquial_question(client):
    body = ask(client, "How much of my unused vacation can I roll over into next year?")
    assert body["found"] is True
    assert "5" in body["answer"]


def test_maternity_leave_abstains(client):
    body = ask(client, "How many weeks of maternity leave do employees get?")
    assert body["found"] is False
    assert body["sources"] == []


def test_document_overview(client):
    body = ask(client, "What is this pdf about?")
    assert body["found"] is True
    assert body["question_type"] == "summary"
    assert "leave" in body["answer"].lower()

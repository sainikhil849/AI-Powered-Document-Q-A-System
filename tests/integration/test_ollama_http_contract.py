"""Integration tests: the real LangChain Ollama clients over real HTTP.

These run the production ``OllamaChatClient`` and ``OllamaEmbeddings`` code
against ``OllamaStub`` (an HTTP imitation of the Ollama API). They verify the
wiring - request format, JSON-schema output format, streaming, error mapping -
but NOT the quality of answers from a real model. See test_real_ollama.py.
"""

from __future__ import annotations

import socket

import pytest
from fastapi.testclient import TestClient

from app.core.container import build_container
from app.main import create_app
from tests.conftest import SAMPLE_TXT, make_settings, upload
from tests.helpers.ollama_stub import OllamaStub


@pytest.fixture
def stub():
    server = OllamaStub().start()
    yield server
    server.stop()


def real_client(tmp_path, base_url: str, **overrides) -> TestClient:
    settings = make_settings(
        tmp_path,
        ollama_base_url=base_url,
        llm_model="llama3:latest",
        embedding_model="nomic-embed-text:latest",
        **overrides,
    )
    # No fakes injected: build_container creates ChatOllama + OllamaEmbeddings.
    return TestClient(create_app(container=build_container(settings)))


def test_full_pipeline_through_langchain_ollama_clients(tmp_path, stub):
    with real_client(tmp_path, stub.base_url) as client:
        assert upload(client, "employee_handbook.txt", SAMPLE_TXT.read_bytes()).status_code == 201
        body = client.post("/ask", json={"question": "How many days of annual leave do employees get?", "include_trace": True}).json()
        assert body["found"] is True and "20 days" in body["answer"]
        assert body["sources"] and "20 days of paid annual leave" in body["sources"][0]["content"]

        maternity = client.post("/ask", json={"question": "How long is maternity leave?"}).json()
        assert maternity["found"] is False and maternity["sources"] == []

        overview = client.post("/ask", json={"question": "What is this document about?"}).json()
        assert overview["found"] is True and overview["question_type"] == "summary"
        assert "Annual Leave" in overview["answer"]

        deps = client.get("/health/dependencies").json()
        assert deps["status"] == "ok" and deps["ollama"]["llm_model_available"] is True

    embed_inputs = [r["body"]["input"] for r in stub.requests if r["path"] == "/api/embed"]
    flat = [text for item in embed_inputs for text in (item if isinstance(item, list) else [item])]
    assert any(t.startswith("search_document: ") for t in flat)  # nomic task prefixes reach Ollama
    assert any(t.startswith("search_query: ") for t in flat)

    chats = stub.chat_requests()
    assert chats, "ChatOllama never called /api/chat"
    for request in chats:
        assert request["model"] == "llama3:latest"
        assert isinstance(request["format"], dict) and request["format"]["type"] == "object"  # JSON schema mode
        assert request["options"]["temperature"] == 0.0
        assert request["options"]["num_ctx"] == 4096


def test_plain_json_mode_for_older_ollama(tmp_path, stub):
    with real_client(tmp_path, stub.base_url, llm_structured_output="json") as client:
        upload(client, "employee_handbook.txt", SAMPLE_TXT.read_bytes())
        client.post("/ask", json={"question": "What are the core working hours?"})
    assert {r["format"] for r in stub.chat_requests()} == {"json"}


def test_missing_chat_model_returns_503_with_pull_command(tmp_path):
    server = OllamaStub(models=("nomic-embed-text:latest",)).start()
    try:
        with real_client(tmp_path, server.base_url) as client:
            upload(client, "employee_handbook.txt", SAMPLE_TXT.read_bytes())
            response = client.post("/ask", json={"question": "What are the core working hours?"})
        assert response.status_code == 503
        error = response.json()["error"]
        assert error["code"] == "MODEL_NOT_AVAILABLE"
        assert "ollama pull llama3:latest" in error["message"]
    finally:
        server.stop()


def test_missing_embedding_model_blocks_upload(tmp_path):
    server = OllamaStub(models=("llama3:latest",)).start()
    try:
        with real_client(tmp_path, server.base_url) as client:
            response = upload(client, "employee_handbook.txt", SAMPLE_TXT.read_bytes())
        assert response.status_code == 503
        assert "ollama pull nomic-embed-text:latest" in response.json()["error"]["message"]
    finally:
        server.stop()


def test_ollama_not_running_returns_503(tmp_path):
    with socket.socket() as sock:  # find a port with nothing listening
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with real_client(tmp_path, f"http://127.0.0.1:{port}") as client:
        response = upload(client, "employee_handbook.txt", SAMPLE_TXT.read_bytes())
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "OLLAMA_UNAVAILABLE"


def test_slow_model_returns_504(tmp_path, stub):
    with real_client(tmp_path, stub.base_url, llm_timeout_seconds=0.5) as client:
        upload(client, "employee_handbook.txt", SAMPLE_TXT.read_bytes())
        stub.delay_seconds = 2.0
        response = client.post("/ask", json={"question": "What are the core working hours?"})
        stub.delay_seconds = 0.0
    # The query embedding has a 60 s timeout, so it waits; the chat call times out at 0.5 s.
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "OLLAMA_TIMEOUT"

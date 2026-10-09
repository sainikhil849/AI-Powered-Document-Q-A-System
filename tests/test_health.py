"""Requirement 1: health endpoint (plus dependency diagnostics and error format)."""

from __future__ import annotations

from app.services import ollama_utils


def test_health_returns_exact_payload(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers.get("X-Request-ID")


def test_dependencies_report_unreachable_ollama_as_degraded(client):
    # The test settings point OLLAMA_BASE_URL at a closed port.
    response = client.get("/health/dependencies")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["ollama"]["reachable"] is False
    assert any("ollama serve" in hint for hint in body["ollama"]["hints"])
    assert body["vector_store"]["ok"] is True
    assert body["vector_store"]["document_count"] == 0


def test_dependencies_report_missing_model_with_pull_command(client, monkeypatch):
    class Reply:
        def raise_for_status(self):
            return None

        def json(self):
            return {"models": [{"model": "fake-embed:latest"}]}

    monkeypatch.setattr(ollama_utils.httpx, "get", lambda *a, **k: Reply())
    body = client.get("/health/dependencies").json()
    assert body["ollama"]["reachable"] is True
    assert body["ollama"]["embedding_model_available"] is True
    assert body["ollama"]["llm_model_available"] is False
    assert "ollama pull fake-llm" in " ".join(body["ollama"]["hints"])


def test_swagger_and_openapi_are_exposed(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/upload", "/ask", "/documents"} <= set(paths)


def test_unknown_route_uses_standard_error_body(client):
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


def test_graph_endpoint_returns_mermaid(client):
    mermaid = client.get("/graph").json()["mermaid"]
    for node in ("process_question", "retrieve_documents", "evaluate_evidence", "reformulate_question",
                 "generate_answer", "validate_grounding", "final_response", "insufficient_response"):
        assert node in mermaid

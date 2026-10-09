"""A tiny HTTP server that imitates the parts of the Ollama REST API this
project uses: ``GET /api/tags``, ``POST /api/embed`` and ``POST /api/chat``
(streaming NDJSON).

It is NOT a language model. It lets the integration tests (and a manual
end-to-end run without Ollama) exercise the real LangChain ``ChatOllama`` and
``OllamaEmbeddings`` clients over real HTTP: request formats, the JSON-schema
``format`` parameter, streaming, missing-model 404s, timeouts and connection
failures. Chat replies are produced by the same "oracle" rules as the unit-test
fake, so answers are only as good as those fixed rules.

Run standalone:  python -m tests.helpers.ollama_stub --port 11434
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from tests.helpers.fakes import FakeEmbeddings, oracle_llm

PREFIXES = ("search_document: ", "search_query: ")
TASK_MARKERS = {
    "describe what a document is about": "generate_summary",
    "evidence auditor": "evaluate_evidence",
    "improve search queries": "reformulate_question",
    "answer questions about uploaded documents": "generate_answer",
    "strict fact-checker": "validate_grounding",
}


def _normalize(model: str) -> str:
    return model if ":" in model else f"{model}:latest"


class OllamaStub:
    def __init__(self, models: tuple[str, ...] = ("llama3:latest", "nomic-embed-text:latest"), port: int = 0) -> None:
        self.models = [_normalize(m) for m in models]
        self.delay_seconds = 0.0
        self.requests: list[dict[str, Any]] = []
        self.embedder = FakeEmbeddings(dimension=768)
        self.llm = oracle_llm()
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep test output quiet
                return

            def _json(self, status: int, payload: dict) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                stub.requests.append({"path": self.path})
                if self.path == "/api/tags":
                    self._json(200, {"models": [{"name": m, "model": m, "size": 1} for m in stub.models]})
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                stub.requests.append({"path": self.path, "body": payload})
                if stub.delay_seconds:
                    time.sleep(stub.delay_seconds)
                model = _normalize(payload.get("model", ""))
                if model not in stub.models:
                    self._json(404, {"error": f'model "{payload.get("model")}" not found, try pulling it first'})
                    return
                if self.path == "/api/embed":
                    inputs = payload.get("input")
                    inputs = [inputs] if isinstance(inputs, str) else list(inputs or [])
                    cleaned = []
                    for text in inputs:
                        for prefix in PREFIXES:
                            if text.startswith(prefix):
                                text = text[len(prefix):]
                        cleaned.append(text)
                    self._json(200, {"model": model, "embeddings": stub.embedder.embed_documents(cleaned)})
                elif self.path == "/api/chat":
                    self._chat(model, payload)
                else:
                    self._json(404, {"error": "not found"})

            def _chat(self, model: str, payload: dict) -> None:
                messages = payload.get("messages", [])
                system = next((m["content"] for m in messages if m["role"] == "system"), "")
                user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
                task = next((t for marker, t in TASK_MARKERS.items() if marker in system), None)
                content = stub.llm.complete(task=task, system=system, user=user) if task else "{}"
                now = datetime.now(UTC).isoformat()
                if payload.get("stream", True):
                    lines = [
                        {"model": model, "created_at": now, "message": {"role": "assistant", "content": content}, "done": False},
                        {"model": model, "created_at": now, "message": {"role": "assistant", "content": ""},
                         "done": True, "done_reason": "stop", "total_duration": 1, "eval_count": 1, "prompt_eval_count": 1},
                    ]
                    body = b"".join(json.dumps(line).encode() + b"\n" for line in lines)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self._json(200, {"model": model, "created_at": now, "message": {"role": "assistant", "content": content},
                                     "done": True, "done_reason": "stop"})

        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.port = self._server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> OllamaStub:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def chat_requests(self) -> list[dict[str, Any]]:
        return [r["body"] for r in self.requests if r["path"] == "/api/chat"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ollama API stub (not a real model)")
    parser.add_argument("--port", type=int, default=11434)
    args = parser.parse_args()
    stub = OllamaStub(port=args.port)
    print(f"Ollama API stub listening on {stub.base_url} (models: {', '.join(stub.models)})")
    stub._server.serve_forever()


if __name__ == "__main__":
    main()

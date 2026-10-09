"""Helpers shared by the Ollama chat and embedding services.

* ``translate_ollama_error`` converts low-level client exceptions into the
  application's stable error types, so callers never see raw httpx/ollama errors.
* ``check_ollama`` powers the startup diagnostics and ``GET /health/dependencies``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import httpx

from app.core.config import normalize_model_name
from app.core.exceptions import (
    AppError,
    EmbeddingError,
    LLMGenerationError,
    ModelNotAvailableError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)

try:  # the ollama client is installed with langchain-ollama
    from ollama import ResponseError as OllamaResponseError
except ImportError:  # pragma: no cover
    OllamaResponseError = None  # type: ignore[assignment,misc]

# Approximate download sizes, used only in operator hints.
MODEL_SIZE_HINTS = {
    "llama3:latest": "about 4.7 GB on disk, roughly 8 GB of free RAM to run",
    "phi3:mini": "about 2.2 GB on disk, roughly 4 GB of free RAM to run",
    "nomic-embed-text:latest": "about 274 MB on disk",
}


def translate_ollama_error(exc: BaseException, *, model: str, operation: Literal["llm", "embedding"]) -> AppError:
    if isinstance(exc, AppError):
        return exc
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        return OllamaTimeoutError()
    # The ollama client raises the built-in ConnectionError when the server is down.
    if isinstance(exc, (ConnectionError, httpx.ConnectError, httpx.RemoteProtocolError)):
        return OllamaUnavailableError()
    if OllamaResponseError is not None and isinstance(exc, OllamaResponseError):
        text = str(getattr(exc, "error", exc)).lower()
        if getattr(exc, "status_code", None) == 404 or ("not found" in text and "model" in text):
            kind = "language" if operation == "llm" else "embedding"
            return ModelNotAvailableError(
                f"The configured {kind} model '{model}' is not installed in Ollama. "
                f"Install it with: ollama pull {model}"
            )
    if isinstance(exc, httpx.HTTPError):
        return OllamaUnavailableError()
    return LLMGenerationError() if operation == "llm" else EmbeddingError()


@dataclass
class OllamaStatus:
    reachable: bool
    llm_model: str
    embedding_model: str
    llm_model_available: bool = False
    embedding_model_available: bool = False
    installed_models: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)


def _model_installed(model: str, installed: list[str]) -> bool:
    wanted = normalize_model_name(model)
    return any(normalize_model_name(name) == wanted for name in installed)


def check_ollama(base_url: str, llm_model: str, embedding_model: str, timeout: float = 3.0) -> OllamaStatus:
    """Ask the local Ollama server which models are installed (GET /api/tags)."""
    status = OllamaStatus(reachable=False, llm_model=llm_model, embedding_model=embedding_model)
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        names = [item.get("model") or item.get("name") or "" for item in payload.get("models", [])]
        status.installed_models = sorted(name for name in names if name)
        status.reachable = True
    except Exception:
        status.hints.append(
            f"Ollama is not reachable at {base_url}. Start the Ollama app (or run `ollama serve`) "
            "and check OLLAMA_BASE_URL. /upload and /ask return 503 until it is available."
        )
        return status

    status.llm_model_available = _model_installed(llm_model, status.installed_models)
    status.embedding_model_available = _model_installed(embedding_model, status.installed_models)
    for model, available in ((llm_model, status.llm_model_available), (embedding_model, status.embedding_model_available)):
        if not available:
            size = MODEL_SIZE_HINTS.get(normalize_model_name(model))
            suffix = f" ({size})" if size else ""
            status.hints.append(f"Model '{model}' is not installed. Run: ollama pull {model}{suffix}")
    return status

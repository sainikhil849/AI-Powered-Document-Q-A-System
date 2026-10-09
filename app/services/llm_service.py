"""Transport layer for the local chat model.

``LLMClient`` is a tiny interface (one ``complete`` method) so the production
``OllamaChatClient`` (LangChain ``ChatOllama``) can be swapped for a
deterministic fake in tests. ``complete_json`` adds JSON parsing, schema
validation and a bounded repair retry on top of any client.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.core.exceptions import LLMOutputError
from app.services.ollama_utils import translate_ollama_error
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class LLMClient(Protocol):
    model_name: str

    def complete(self, *, task: str, system: str, user: str, json_schema: dict[str, Any] | None = None) -> str:
        """Return the raw text produced by the model for one system+user prompt."""
        ...


class OllamaChatClient:
    """Production client backed by LangChain's ``ChatOllama``."""

    def __init__(self, settings: Settings) -> None:
        from langchain_ollama import ChatOllama

        self.model_name = settings.llm_model
        self.structured_output = settings.llm_structured_output
        self._chat = ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
            num_ctx=settings.llm_num_ctx,
            num_predict=settings.llm_num_predict,  # bounds the length of every generation
            seed=42,  # with temperature 0 this makes runs repeatable
            client_kwargs={"timeout": settings.llm_timeout_seconds},
        )

    def complete(self, *, task: str, system: str, user: str, json_schema: dict[str, Any] | None = None) -> str:
        fmt: str | dict[str, Any] | None = None
        if json_schema is not None:
            fmt = json_schema if self.structured_output == "json_schema" else "json"
        try:
            message = self._chat.invoke([SystemMessage(content=system), HumanMessage(content=user)], format=fmt)
        except Exception as exc:
            logger.error("LLM call '%s' failed: %s", task, type(exc).__name__)
            raise translate_ollama_error(exc, model=self.model_name, operation="llm") from exc
        content = message.content
        return content if isinstance(content, str) else json.dumps(content)


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract one JSON object from model output (tolerates code fences and stray prose)."""
    cleaned = _FENCE.sub("", text.strip())
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise LLMOutputError("No JSON object found in model output.") from None
        try:
            value = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMOutputError("Model output is not valid JSON.") from exc
    if not isinstance(value, dict):
        raise LLMOutputError("Model output is JSON but not an object.")
    return value


def complete_json(
    client: LLMClient,
    *,
    task: str,
    system: str,
    user: str,
    schema: type[ModelT],
    retries: int = 1,
) -> ModelT:
    """Call the model and return a validated ``schema`` instance.

    Malformed output is retried ``retries`` times with an explicit correction
    note, then raises ``LLMOutputError``. Infrastructure errors (Ollama down,
    timeout, missing model) are raised immediately and never retried here.
    """
    json_schema = schema.model_json_schema()
    prompt = user
    last_error = "unknown"
    for attempt in range(retries + 1):
        raw = client.complete(task=task, system=system, user=prompt, json_schema=json_schema)
        try:
            return schema.model_validate(parse_json_object(raw))
        except (LLMOutputError, ValidationError) as exc:
            last_error = type(exc).__name__
            logger.warning("Malformed output for '%s' (attempt %d/%d): %s", task, attempt + 1, retries + 1, last_error)
            prompt = (
                f"{user}\n\nYour previous reply was not valid JSON for the required schema. "
                "Reply again with ONLY a JSON object that matches the schema."
            )
    raise LLMOutputError(f"Model output for '{task}' was invalid after {retries + 1} attempt(s): {last_error}")

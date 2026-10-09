"""Deterministic test doubles for the two model dependencies.

* ``FakeEmbeddings`` - a hashing bag-of-words embedder. Texts that share words
  get similar vectors, so real Chroma similarity search behaves sensibly in
  tests without any model.
* ``FakeLLM`` - implements the ``LLMClient`` interface. Each task
  (evaluate_evidence, reformulate_question, generate_answer, validate_grounding)
  is answered by a scripted value, a list of values, or a function of the prompt.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.embeddings import Embeddings

_TOKEN = re.compile(r"[a-z0-9]+")
SOURCE_BLOCK = re.compile(r'<source id="(S\d+)"[^>]*>\n(.*?)\n</source>', re.DOTALL)
HEADING = re.compile(r"^\d+\.\s+([A-Z][A-Z &]+)$", re.MULTILINE)


def _stem(token: str) -> str:
    return token[:-1] if len(token) > 3 and token.endswith("s") else token


class FakeEmbeddings(Embeddings):
    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []
        self.fail_on_documents: BaseException | None = None
        self.fail_on_query: BaseException | None = None

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in _TOKEN.findall(text.lower()):
            index = int(hashlib.md5(_stem(token).encode()).hexdigest(), 16) % self.dimension
            vector[index] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.fail_on_documents is not None:
            raise self.fail_on_documents
        self.document_calls.append(list(texts))
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        if self.fail_on_query is not None:
            raise self.fail_on_query
        self.query_calls.append(text)
        return self._vector(text)


@dataclass
class LLMCall:
    task: str
    system: str
    user: str
    json_schema: dict[str, Any] | None


Handler = str | dict | list | BaseException | Callable[[str], Any]


def parse_sources(prompt: str) -> dict[str, str]:
    return dict(SOURCE_BLOCK.findall(prompt))


def parse_question(prompt: str) -> str:
    match = re.search(r"QUESTION:\n(.*?)\n\n", prompt, re.DOTALL)
    return match.group(1) if match else ""


@dataclass
class FakeLLM:
    handlers: dict[str, Handler] = field(default_factory=dict)
    model_name: str = "fake-llm"
    calls: list[LLMCall] = field(default_factory=list)

    def complete(self, *, task: str, system: str, user: str, json_schema: dict[str, Any] | None = None) -> str:
        self.calls.append(LLMCall(task, system, user, json_schema))
        if task not in self.handlers:
            raise AssertionError(f"FakeLLM has no handler for task '{task}'")
        handler = self.handlers[task]
        if isinstance(handler, list):
            value = handler.pop(0) if len(handler) > 1 else handler[0]
        else:
            value = handler
        if isinstance(value, BaseException):
            raise value
        if callable(value):
            value = value(user)
        if isinstance(value, BaseException):
            raise value
        return value if isinstance(value, str) else json.dumps(value)

    def calls_for(self, task: str) -> list[LLMCall]:
        return [c for c in self.calls if c.task == task]


# ---------------------------------------------------------------------------
# An "oracle" LLM for the sample handbook: behaves like a perfect model for a
# fixed set of facts, so API tests exercise real retrieval and source mapping.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Fact:
    keyword: str  # word that identifies the question topic
    evidence: str  # exact phrase that must be present in a source
    answer: str


HANDBOOK_FACTS = [
    Fact("annual leave", "20 days of paid annual leave", "Full-time employees receive 20 days of paid annual leave per calendar year."),
    Fact("core working hours", "10:00 a.m. to 4:00 p.m.", "Core working hours are 10:00 a.m. to 4:00 p.m. on working days."),
    Fact("medical certificate", "longer than 3 consecutive days", "A medical certificate is required for sick leave longer than 3 consecutive days."),
    Fact("remote", "up to 2 days per week", "Eligible employees may work remotely for up to 2 days per week."),
]


def _fact_for(question: str, facts: list[Fact]) -> Fact | None:
    q = question.lower()
    return next((f for f in facts if f.keyword in q), None)


def oracle_llm(facts: list[Fact] | None = None) -> FakeLLM:
    facts = facts or HANDBOOK_FACTS

    def evaluate(prompt: str) -> dict:
        sources = parse_sources(prompt)
        fact = _fact_for(parse_question(prompt), facts)
        if fact is None:
            related = [label for label, text in sources.items() if "leave" in text.lower()]
            return {"relevant_sources": related, "reasoning": "Related policies only.", "sufficient": False,
                    "missing_information": "The sources do not cover the requested policy."}
        hits = [label for label, text in sources.items() if fact.evidence in text]
        return {"relevant_sources": hits, "reasoning": "checked", "sufficient": bool(hits),
                "missing_information": "" if hits else f"No source states '{fact.keyword}'."}

    def generate(prompt: str) -> dict:
        sources = parse_sources(prompt)
        fact = _fact_for(parse_question(prompt), facts)
        hits = [label for label, text in sources.items() if fact and fact.evidence in text]
        if not hits:
            return {"found": False, "answer": "", "citations": []}
        return {"found": True, "answer": fact.answer, "citations": hits[:1]}

    def summarize(prompt: str) -> dict:
        """Describe the document from the section headings found in the passages."""
        sources = parse_sources(prompt)
        headings = []
        for text in sources.values():
            for heading in HEADING.findall(text):
                if heading.title() not in headings:
                    headings.append(heading.title())
        if not headings:
            return {"found": False, "answer": "", "citations": []}
        answer = "This document is an employee handbook. It covers:\n" + "\n".join(f"- {h}" for h in headings)
        return {"found": True, "answer": answer, "citations": list(sources)}

    return FakeLLM(
        handlers={
            "generate_summary": summarize,
            "evaluate_evidence": evaluate,
            "reformulate_question": lambda prompt: {"query": f"policy about {parse_question(prompt).lower()}"},
            "generate_answer": generate,
            "validate_grounding": {"unsupported_claims": [], "grounded": True},
        }
    )

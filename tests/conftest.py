"""Shared fixtures.

Every test gets its own temporary Chroma directory, a fake embedder and a
fake LLM, so tests never need Ollama and never touch ./data/chroma.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.container import ServiceContainer, build_container
from app.main import create_app
from tests.helpers.fakes import FakeEmbeddings, FakeLLM, oracle_llm
from tests.helpers.pdf_factory import build_pdf, text_to_pages

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_TXT = ROOT / "sample_documents" / "employee_handbook.txt"


def make_settings(tmp_path: Path, **overrides) -> Settings:
    values = dict(
        vector_store_path=tmp_path / "chroma",
        llm_model="fake-llm",
        embedding_model="fake-embed",
        ollama_base_url="http://127.0.0.1:9",  # nothing listens here
        chunk_size=800,
        chunk_overlap=120,
        top_k=4,
        max_retries=2,
        max_grounding_retries=1,
        max_upload_size_mb=1,
        log_level="WARNING",
    )
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return oracle_llm()


@pytest.fixture
def container(settings: Settings, fake_llm: FakeLLM, fake_embeddings: FakeEmbeddings) -> ServiceContainer:
    return build_container(settings, llm_client=fake_llm, embedder=fake_embeddings)


@pytest.fixture
def client(container: ServiceContainer) -> Iterator[TestClient]:
    app = create_app(container=container)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_factory(tmp_path: Path) -> Callable[..., TestClient]:
    """Build a client with custom settings / fakes (closed automatically)."""
    opened: list[TestClient] = []

    def factory(*, llm: FakeLLM | None = None, embeddings: FakeEmbeddings | None = None, raise_server_exceptions=True, **overrides):
        container = build_container(
            make_settings(tmp_path, **overrides),
            llm_client=llm or oracle_llm(),
            embedder=embeddings or FakeEmbeddings(),
        )
        test_client = TestClient(create_app(container=container), raise_server_exceptions=raise_server_exceptions)
        test_client.__enter__()
        test_client.container = container  # type: ignore[attr-defined]
        opened.append(test_client)
        return test_client

    yield factory
    for test_client in opened:
        test_client.__exit__(None, None, None)


@pytest.fixture
def handbook_bytes() -> bytes:
    return SAMPLE_TXT.read_bytes()


@pytest.fixture
def handbook_pdf_bytes() -> bytes:
    return build_pdf(text_to_pages(SAMPLE_TXT.read_text(encoding="utf-8")))


def upload(client: TestClient, name: str, content: bytes, content_type: str = "application/octet-stream"):
    return client.post("/upload", files={"file": (name, content, content_type)})

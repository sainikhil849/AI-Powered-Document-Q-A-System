"""HTTP endpoints. Route functions stay thin: validate input, call a service,
shape the response. All errors are raised as AppError subclasses and turned
into the standard error body by the handlers in ``app/main.py``."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Request, Response, UploadFile, status

from app.api.schemas import (
    AskRequest,
    AskResponse,
    DependenciesResponse,
    DocumentListResponse,
    DocumentSelectionInfo,
    DocumentSummary,
    ErrorResponse,
    GraphResponse,
    HealthResponse,
    OllamaDependency,
    SourceItem,
    UploadResponse,
    VectorStoreDependency,
    WorkflowTrace,
)
from app.core.container import ServiceContainer
from app.core.exceptions import AppError, FileTooLargeError
from app.services.ollama_utils import check_ollama

router = APIRouter()


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.container


def _errors(*codes: int) -> dict[int | str, dict]:
    descriptions = {
        400: "Bad request (e.g. empty file)",
        404: "No documents uploaded, or unknown document_id",
        413: "File too large",
        415: "Unsupported file type",
        422: "Validation error or unprocessable document",
        500: "Internal, retrieval or workflow error",
        502: "Local model returned an error",
        503: "Ollama or the vector store is unavailable",
        504: "Local model timed out",
    }
    return {code: {"model": ErrorResponse, "description": descriptions[code]} for code in codes}


@router.get("/health", response_model=HealthResponse, tags=["health"], summary="Liveness check")
def health() -> HealthResponse:
    return HealthResponse()


@router.get(
    "/health/dependencies",
    response_model=DependenciesResponse,
    tags=["health"],
    summary="Check Ollama models and the vector store",
)
def dependencies(container: ServiceContainer = Depends(get_container)) -> DependenciesResponse:
    settings = container.settings
    ollama = check_ollama(settings.ollama_base_url, settings.llm_model, settings.embedding_model)
    try:
        documents = container.store.list_documents()
        store = VectorStoreDependency(
            ok=True,
            collection=settings.collection_name,
            document_count=len(documents),
            chunk_count=container.store.count(),
        )
    except AppError:
        store = VectorStoreDependency(ok=False, collection=settings.collection_name, document_count=None, chunk_count=None)
    healthy = ollama.reachable and ollama.llm_model_available and ollama.embedding_model_available and store.ok
    return DependenciesResponse(
        status="ok" if healthy else "degraded",
        ollama=OllamaDependency(
            reachable=ollama.reachable,
            base_url=settings.ollama_base_url,
            llm_model=ollama.llm_model,
            llm_model_available=ollama.llm_model_available,
            embedding_model=ollama.embedding_model,
            embedding_model_available=ollama.embedding_model_available,
            installed_models=ollama.installed_models,
            hints=ollama.hints,
        ),
        vector_store=store,
    )


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
    summary="Upload and index a PDF or TXT document",
    responses={200: {"model": UploadResponse, "description": "Identical content was already indexed (status=duplicate)"},
               **_errors(400, 413, 415, 422, 500, 502, 503, 504)},
)
def upload_document(
    response: Response,
    file: UploadFile = File(..., description="A .pdf or .txt file"),
    container: ServiceContainer = Depends(get_container),
) -> UploadResponse:
    limit = container.settings.max_upload_bytes
    container.processor.detect_file_type(file.filename)  # reject wrong types before reading the body
    content = file.file.read(limit + 1)
    if len(content) > limit:
        raise FileTooLargeError(f"The uploaded file exceeds the {container.settings.max_upload_size_mb:g} MB limit.")
    result = container.ingestion.ingest(file.filename, content)
    if result.status == "duplicate":
        response.status_code = status.HTTP_200_OK
    return UploadResponse(**result.model_dump())


@router.get("/documents", response_model=DocumentListResponse, tags=["documents"], summary="List indexed documents",
            responses=_errors(503))
def list_documents(container: ServiceContainer = Depends(get_container)) -> DocumentListResponse:
    documents = container.store.list_documents()
    return DocumentListResponse(
        documents=[DocumentSummary(**doc.model_dump(exclude={"embedding_model"})) for doc in documents],
        latest_document_id=documents[0].document_id if documents else None,
    )


@router.post(
    "/ask",
    response_model=AskResponse,
    tags=["questions"],
    summary="Ask a question; the relevant document is chosen automatically unless document_id is given",
    responses=_errors(404, 422, 500, 502, 503, 504),
)
def ask_question(request: AskRequest, container: ServiceContainer = Depends(get_container)) -> AskResponse:
    result = container.qa.ask(request.question, request.document_id)
    return AskResponse(
        answer=result.answer,
        found=result.found,
        question_type=result.question_type,
        sources=[SourceItem(**source) for source in result.sources],
        document_id=result.document_id,
        document_selection=DocumentSelectionInfo(**result.document_selection),
        trace=WorkflowTrace(**result.trace) if request.include_trace else None,
    )


@router.get("/graph", response_model=GraphResponse, tags=["workflow"], summary="LangGraph workflow as Mermaid")
def workflow_graph(container: ServiceContainer = Depends(get_container)) -> GraphResponse:
    return GraphResponse(mermaid=container.workflow.mermaid())

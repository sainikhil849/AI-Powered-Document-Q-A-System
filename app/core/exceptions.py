"""Application exception hierarchy.

Every error that can reach an API client is an ``AppError`` subclass with a
stable machine-readable ``code``, an HTTP ``status_code`` and a safe default
``message``. Internal details (stack traces, file paths, raw library errors)
are logged on the server and never placed in the client-facing message.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    status_code: int = 500
    code: str = "INTERNAL_ERROR"
    message: str = "An unexpected internal error occurred."

    def __init__(self, message: str | None = None, *, details: list[dict[str, Any]] | None = None):
        self.message = message or self.__class__.message
        self.details = details
        super().__init__(self.message)


# --- Upload / document validation (4xx) -------------------------------------
class UnsupportedFileTypeError(AppError):
    status_code = 415
    code = "UNSUPPORTED_FILE_TYPE"
    message = "Only PDF (.pdf) and plain-text (.txt) files are supported."


class FileTooLargeError(AppError):
    status_code = 413
    code = "FILE_TOO_LARGE"
    message = "The uploaded file exceeds the maximum allowed size."


class EmptyFileError(AppError):
    status_code = 400
    code = "EMPTY_FILE"
    message = "The uploaded file is empty."


class InvalidFileContentError(AppError):
    status_code = 422
    code = "INVALID_FILE_CONTENT"
    message = "The file content does not match its file type."


class CorruptedPDFError(AppError):
    status_code = 422
    code = "CORRUPTED_PDF"
    message = "The PDF file is corrupted or cannot be parsed."


class EncryptedPDFError(AppError):
    status_code = 422
    code = "ENCRYPTED_PDF"
    message = "The PDF is password-protected and cannot be read."


class TooManyPagesError(AppError):
    status_code = 422
    code = "TOO_MANY_PAGES"
    message = "The PDF has more pages than the configured maximum."


class NoExtractableTextError(AppError):
    status_code = 422
    code = "NO_EXTRACTABLE_TEXT"
    message = (
        "No extractable text was found in the document. "
        "Scanned or image-only PDFs need OCR, which this service does not perform."
    )


# --- Question / document lookup (4xx) -----------------------------------------
class InvalidQuestionError(AppError):
    status_code = 422
    code = "INVALID_QUESTION"
    message = "The question is empty or invalid."


class NoDocumentsError(AppError):
    status_code = 404
    code = "NO_DOCUMENTS"
    message = "No documents have been uploaded yet. Upload a PDF or TXT file with POST /upload first."


class DocumentNotFoundError(AppError):
    status_code = 404
    code = "DOCUMENT_NOT_FOUND"
    message = "No document exists with the given document_id."


# --- Local model service (5xx) ------------------------------------------------
class OllamaUnavailableError(AppError):
    status_code = 503
    code = "OLLAMA_UNAVAILABLE"
    message = "The local Ollama service is not reachable. Start Ollama and try again."


class ModelNotAvailableError(AppError):
    status_code = 503
    code = "MODEL_NOT_AVAILABLE"
    message = "A configured Ollama model is not installed."


class OllamaTimeoutError(AppError):
    status_code = 504
    code = "OLLAMA_TIMEOUT"
    message = "The local model did not respond in time. Try again or use a smaller model."


class EmbeddingError(AppError):
    status_code = 502
    code = "EMBEDDING_FAILED"
    message = "Generating embeddings with the local embedding model failed."


class EmbeddingDimensionMismatchError(AppError):
    status_code = 500
    code = "EMBEDDING_DIMENSION_MISMATCH"
    message = (
        "The embedding dimension does not match the existing vector index. "
        "The index was built with a different embedding model."
    )


class LLMGenerationError(AppError):
    status_code = 502
    code = "LLM_FAILED"
    message = "The local language model failed to produce a response."


class LLMOutputError(AppError):
    """Raised when the model output is not valid JSON for the expected schema.

    This error is normally handled inside the answer service (conservative
    fallback) and does not reach API clients.
    """

    status_code = 502
    code = "LLM_INVALID_OUTPUT"
    message = "The local language model returned output that could not be parsed."


# --- Storage / retrieval / workflow (5xx) ---------------------------------------
class VectorStoreError(AppError):
    status_code = 503
    code = "VECTOR_STORE_UNAVAILABLE"
    message = "The vector store is unavailable or failed to complete the operation."


class RetrievalError(AppError):
    status_code = 500
    code = "RETRIEVAL_FAILED"
    message = "Retrieving relevant passages from the document failed."


class WorkflowError(AppError):
    status_code = 500
    code = "WORKFLOW_ERROR"
    message = "The question-answering workflow failed."


def _all_subclasses(cls: type[AppError]) -> list[type[AppError]]:
    result = []
    for sub in cls.__subclasses__():
        result.append(sub)
        result.extend(_all_subclasses(sub))
    return result


def error_from_code(code: str, message: str | None = None) -> AppError:
    """Rebuild an AppError from its code (used to surface errors recorded in graph state)."""
    for cls in [AppError, *_all_subclasses(AppError)]:
        if cls.code == code:
            return cls(message)
    return WorkflowError(message)

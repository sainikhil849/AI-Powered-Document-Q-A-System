# Verification Report

Build environment: Linux (x86-64, 2 vCPU, 7 GB RAM), 9 October 2026. Ollama and its
model registry were **not reachable** from this environment (blocked by its network
policy), so no real `llama3` / `nomic-embed-text` inference could be run. Everything
that does not depend on real model output was executed and verified. Items that depend
on it are marked **NOT VERIFIED** with the exact command to verify them on your machine.

Status legend: **PASS** = executed with evidence below · **NOT VERIFIED** = not executed in
the build environment · **FAIL** = executed and failed (none).

---

## 1. Final project tree

```
document-qa-system/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/        __init__.py  routes.py  schemas.py
│   ├── core/       __init__.py  config.py  container.py  exceptions.py  models.py
│   ├── graph/      __init__.py  nodes.py  state.py  workflow.py
│   ├── services/   __init__.py  answer_service.py  document_processor.py  embedding_service.py
│   │               ingestion_service.py  llm_service.py  ollama_utils.py  qa_service.py
│   │               question_analysis.py  retrieval_service.py  vector_store.py
│   └── utils/      __init__.py  logging_config.py
├── tests/
│   ├── __init__.py  conftest.py
│   ├── test_ask.py  test_graph.py  test_health.py  test_retrieval.py  test_upload.py
│   ├── helpers/      __init__.py  fakes.py  ollama_stub.py  pdf_factory.py
│   └── integration/  __init__.py  test_ollama_http_contract.py  test_real_ollama.py
├── scripts/        benchmark_retrieval.py  export_graph.py  make_sample_pdf.py  smoke_test.py
├── sample_documents/ employee_handbook.txt  employee_handbook.pdf
├── docs/           TECHNICAL_WALKTHROUGH.md  VERIFICATION_REPORT.md  langgraph_workflow.mmd
├── data/.gitkeep
├── .env.example  .gitignore  pytest.ini  requirements.txt  README.md
```

## 2. Exact setup and run commands (Windows PowerShell)

```powershell
ollama list
ollama pull nomic-embed-text:latest      # only if missing (~274 MB)
ollama pull llama3:latest                # only if missing (~4.7 GB)

cd document-qa-system
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1             # if blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env

pytest                                   # expect: 173 passed, 1 skipped
uvicorn app.main:app --host 127.0.0.1 --port 8000
# second terminal (activate .venv again):
python scripts\smoke_test.py             # real-model end-to-end check
$env:RUN_OLLAMA_TESTS = "1"; pytest -m ollama -v -s
```

## 3. Actual test results

Fresh virtual environment per Python version, `pip install -r requirements.txt`, then `pytest`:

| Python | Result |
|---|---|
| 3.11.17 | 173 passed, 1 skipped |
| 3.12.3 | 173 passed, 1 skipped |
| 3.13.16 | 173 passed, 1 skipped |

- The 1 skipped item is the module `tests/integration/test_real_ollama.py` (5 tests), which needs `RUN_OLLAMA_TESTS=1` and a real Ollama. **NOT VERIFIED.** (Its code path was dry-run against the Ollama-API stub: 3 passed; the colloquial-wording test failed there as expected, because the stub's fixed rules do not understand paraphrases — a real model is needed for that test.)
- Line coverage of `app/`: 94 % (coverage.py).
- Mutation check: deliberately breaking the retry bound, the Chroma `where` filter, the number check, the original-question rule, one-based page numbering, duplicate detection, the grounding bound, page mapping, evaluator validation, summary routing, summary detection, overview sampling, focused-topic handling, the all-documents search or the multi-document `$in` filter made the suite fail in every case.
- All 4 Mermaid diagrams in README.md and `docs/langgraph_workflow.mmd` were parsed successfully with the Mermaid library.
- A script cross-checked the docs against the code: every setting in `Settings` appears in README and `.env.example`, every error code and endpoint is documented, and every test, file and function name referenced in the docs exists.

## 4. End-to-end verification

**4.1 Real server + real LangChain Ollama clients + Ollama-API stub (PASS for plumbing).**
`uvicorn app.main:app` was started for real; Ollama was replaced by `tests/helpers/ollama_stub.py`
on port 11434, which speaks the Ollama HTTP API but answers with fixed rules (it is not a model).

| Check | Result |
|---|---|
| Startup with Ollama **down** | Server starts; log: `STARTUP CHECK: Ollama is not reachable at http://localhost:11434 ...`; `/health` → `{"status":"ok"}`; `/upload` → `503 OLLAMA_UNAVAILABLE` |
| Startup with stub up | Log: `Ollama reachable ...; installed models: llama3:latest, nomic-embed-text:latest` and `Models ready` |
| `POST /upload` PDF | `201`, `page_count: 3`, `chunk_count: 7`, id `a532cd084d2318d8` |
| `POST /upload` TXT, then again | `201` then `200` with `status: "duplicate"`, same id `3f93b04f86b13651` |
| `GET /documents` | both documents, `latest_document_id` = the re-uploaded TXT |
| `POST /ask` annual leave (PDF id) | `found: true`, source `page: 1, pages: [1]`, content = stored chunk |
| `POST /ask` medical certificate | `found: true`, source `page: 2` |
| `POST /ask` maternity leave | `found: false`, `sources: []`, 3 queries tried, path ends in `insufficient_response` |
| `POST /ask` colloquial remote-work question | stub could not match the paraphrase → full retry path → abstained (expected for the stub; real model **NOT VERIFIED**) |
| Errors | `415 UNSUPPORTED_FILE_TYPE`, `404 DOCUMENT_NOT_FOUND`, `422 VALIDATION_ERROR`, `422 CORRUPTED_PDF` with the standard body and `request_id` |
| `GET /docs` | `200` |
| Restart on the same `VECTOR_STORE_PATH` | both documents listed again; latest default preserved; `/ask` answers |
| Log hygiene | `grep` for handbook phrases in the server logs: 0 matches |
| **Second round (after the fix for overviews and automatic selection)** — `cafeteria_menu.txt` and `employee_handbook.pdf` uploaded, no `document_id` in any question | |
| "explain me what is this pdf about" | `found: true`, `question_type: summary`, overview of all 9 handbook sections from all 7 passages (pages 1–3), mode `latest`, path through `gather_overview` |
| "explianme whatbis pdf about" (typos) | `found: true`, `question_type: summary` |
| "How many days of annual leave do employees get per year?" | `found: true`, source `employee_handbook.pdf` page 1; `document_selection.mode: relevance`, `searched_document_ids: null`, scores handbook 0.5235 vs menu 0.0991 |
| "What is served for lunch on Friday?" | selection scored the menu highest (0.3107 vs 0.131); the stub's fixed rules have no menu fact, so it abstained (expected for the stub) |
| "Summarize the cafeteria menu" | mode `filename_match`, overview of the menu |
| "How many weeks of maternity leave do employees get?" | `found: false`, `sources: []`, `document_id: null`, mode `relevance` |
| Explicit `document_id` + "What is this document about?" | mode `explicit`, overview of that document |
| Restart, then `GET /documents` | both documents listed |
| Requests sent to Ollama | `/api/embed` inputs carry `search_document: ` / `search_query: `; every `/api/chat` request has `format` = JSON schema, `temperature 0.0`, `num_ctx 4096` |

**4.2 Real models (NOT VERIFIED).** Answer quality, abstention behaviour and latency with
real `llama3:latest` + `nomic-embed-text:latest` were not executed. Verify with
`python scripts\smoke_test.py` (prints answers, sources, graph path and seconds per
question) and `pytest -m ollama`.

## 5. Requirement checklist

| Requirement | Status | Evidence |
|---|---|---|
| `GET /health` returns `{"status": "ok"}` | PASS | `test_health_returns_exact_payload`; live server |
| `POST /upload` accepts PDF and TXT (multipart) | PASS | upload tests; live server |
| `POST /ask` with question and optional `document_id` | PASS | ask tests; live server |
| No `document_id` needed: documents chosen by relevance, scores returned | PASS (pipeline) | selection tests; live run with two documents |
| "What is this PDF about?" answered with an overview | PASS (pipeline) / NOT VERIFIED (real model wording) | overview tests incl. typo wording; live run |
| Pydantic request/response models, consistent formats | PASS | `app/api/schemas.py`; one error body for all errors |
| Swagger UI at `/docs` | PASS | `test_swagger_and_openapi_are_exposed`; live `200` |
| Validation of extension, content, size, readability | PASS | 415/413/400/422 tests |
| Reject empty files and textless documents | PASS | `EMPTY_FILE`, `NO_EXTRACTABLE_TEXT` tests |
| PDF page-by-page extraction, one-based pages | PASS | `test_pdf_pages_are_one_based_and_preserved` |
| TXT deliberate encoding handling | PASS | `test_txt_encodings_are_detected` (UTF-8, BOM, UTF-16, cp1252) |
| Overlapping chunks with doc ID, filename, page, chunk index | PASS | metadata/offset/overlap tests |
| Chunks spanning pages handled explicitly, no fabricated pages | PASS | cross-page and unlocatable-chunk tests |
| Embeddings through Ollama | PASS (HTTP contract) / NOT VERIFIED (real model) | stub integration tests |
| Persistent Chroma, discoverable after restart | PASS | `test_documents_persist_across_restarts`; live restart |
| Stable document ID, duplicate policy, multiple documents | PASS | duplicate and isolation tests |
| Same embedding model for index and query | PASS | per-model collection; dimension tests |
| Question with `document_id` searches only that document | PASS | isolation tests |
| Default without `document_id` documented | PASS | `auto` (relevance) by default; `DEFAULT_DOCUMENT_SCOPE=latest` gives the assignment's latest-upload default (`test_default_scope_latest_only_searches_the_latest_upload`) |
| `ChatOllama` / `OllamaEmbeddings`, env-configured | PASS | `test_full_pipeline_through_langchain_ollama_clients` |
| No paid or hosted services; no data leaves the machine | PASS | code uses only `OLLAMA_BASE_URL`; tracing forced off; Chroma telemetry off |
| Startup diagnostics for missing Ollama / models | PASS | live startup log; `/health/dependencies` tests |
| No silent model switching | PASS | no fallback model in code |
| Configurable top-k, documented relevance strategy | PASS | README §24 |
| No arbitrary similarity threshold | PASS | default off; benchmark evidence (stand-in) |
| Separate relevance / sufficiency / grounding decisions | PASS | three separate calls and schemas |
| Structured evaluator output, malformed handling | PASS | parsing and malformed-output tests |
| Real `StateGraph`, typed state, conditional edges | PASS | `test_compiled_graph_has_the_required_nodes_and_conditional_edges` |
| Retry path executes with the reformulated query | PASS | `test_reformulation_retrieves_new_evidence_with_the_reformulated_query` |
| Max 2 reformulation retries by default, enforced | PASS | `test_retry_limit_is_enforced[0-3]` |
| Original question preserved; intent preserved | PASS | tests assert evaluation/generation use the original |
| Bounded grounding correction or abstention | PASS | grounding tests |
| All branches terminate | PASS | 15 branch tests + recursion-limit test |
| LLM/embedding/retrieval exceptions handled safely | PASS | failure tests (503/504/502/500) |
| Answer only from evidence; abstain when missing | PASS (pipeline) / NOT VERIFIED (real model) | mocked + stub tests |
| Sources are real text and metadata | PASS | content compared with the document in tests |
| Error handling list (section G) | PASS | each case has a test |
| No stack traces, secrets or paths in responses | PASS | leakage assertions in tests |
| No document contents in logs | PASS | live log grep |
| `.env`, indexes, caches git-ignored | PASS | `.gitignore` |
| Tests need no paid API and no running Ollama | PASS | default `pytest` run |
| Chunk-size benchmark | PASS (lexical stand-in) / NOT VERIFIED (nomic-embed-text) | `scripts/benchmark_retrieval.py` |
| Windows PowerShell instructions | NOT VERIFIED | written for Windows; not executed on Windows |
| README matches implementation | PASS | automated cross-check (section 3) |

## 6. Remaining limitations

1. Real-model behaviour and latency are unverified until you run the smoke test. The prompts were also made less strict (paraphrases and partial answers accepted) to reduce false "not enough information" answers; how much that helps with `llama3` needs the smoke test.
2. LLM-based evaluation and fact-checking reduce hallucinations but cannot guarantee correctness.
3. The deterministic number check can reject correct answers that rewrite numbers (→ abstention).
4. No OCR, no layout-aware PDF parsing; TXT has no page numbers.
5. Single process, no authentication, document listing scans chunk metadata.
6. Whole-document questions are detected by wording rules; long documents are summarised from a sample of passages; a generic "what is this pdf about?" with several uploads describes the latest upload.

## 7. Five design decisions to be ready to defend

1. **LangGraph with two bounded loops and a routing step.** `select_documents` routes whole-document questions to an overview path and fact questions to retrieval. Re-retrieval (`retry_count < max_retries`) and answer correction (`generation_attempts <= max_grounding_retries`) use counters that only grow, so every run terminates (≤ 15 node runs by default), with `recursion_limit` as a second guard.
2. **Similarity ranks, the LLM decides sufficiency.** Relevance = cosine similarity (question vs chunk; a document's score = its best chunk), returned to the client. The benchmark showed an unanswerable question scoring above a correct match, so no threshold is used by default; the structured evaluator decides, and malformed output counts as "insufficient".
3. **Reformulation only changes the search query.** Evaluation, generation and grounding always use the original question, so intent cannot drift into the answer.
4. **Citations are mapped in code, sources come from storage.** The model returns only labels; text, filename and pages come from Chroma; deterministic + LLM grounding checks with one bounded correction.
5. **Errors are never disguised as "not found".** Node errors travel in state to `handle_error` and come back as typed `503/504/502/500`; `found: false` means only "the document does not contain it".

## 8. Demonstration script (about 5 minutes)

1. Terminal: `ollama list` — show both models. Start: `uvicorn app.main:app --host 127.0.0.1 --port 8000`. Point at "Models ready" in the log.
2. Browser `http://127.0.0.1:8000/docs`: run `GET /health` → `{"status":"ok"}`; `GET /health/dependencies` → `status: ok`.
3. `POST /upload` with `sample_documents/employee_handbook.pdf` → `201`, 3 pages; upload again → `200 duplicate`, same id. Say: "content-hash IDs make uploads idempotent."
4. `POST /ask` `{"question": "What is this pdf about?"}` — no `document_id` → `question_type: summary`, an overview of every section, sources from pages 1–3.
5. `POST /ask` `{"question": "How many days of annual leave do employees get per year?", "include_trace": true}` → answer, `page: 1`, the `content` passage, `document_selection` with the relevance score, and the trace path through `validate_grounding`.
6. Upload any second `.txt`, ask the same question again → `candidates` shows both scores; the handbook wins on relevance, not upload order. Say: "relevance = cosine similarity of the question to each passage; a document's score is its best passage."
7. Ask "Can I work from my house some days, and how many?" with trace → if `retries_used > 0`, show `queries`: the reformulated query was used only for retrieval.
8. Ask "How many weeks of maternity leave do employees get?" with trace → `found: false`, `sources: []`, three queries, `abstain_reason: insufficient_evidence`. Say: "the generator was never called, so nothing could be invented."
9. Errors: upload any `.docx` → `415`; ask with `"document_id": "0000000000000000"` → `404`. Show the standard body with `request_id`.
10. `GET /graph` → conditional edges; terminal: `pytest -q` → 173 passed.
11. Close with the limitation: "These checks reduce hallucinations; they don't guarantee zero. When unsure, the system abstains."

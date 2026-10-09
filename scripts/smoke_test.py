"""End-to-end smoke test against a RUNNING server (real Ollama models).

Start the API first (uvicorn app.main:app --port 8000), then in a second terminal:

    python scripts/smoke_test.py                       # uses the sample PDF
    python scripts/smoke_test.py --file sample_documents/employee_handbook.txt
    python scripts/smoke_test.py --url http://127.0.0.1:8000

It uploads the sample handbook, asks the five demonstration questions (without
a document_id, so automatic document selection is exercised), and prints each
answer, its sources, the document-selection scores, the LangGraph path and the
time taken. The expectations are checks for a human reviewer, not unit-test assertions.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]

DEMO = [
    ("Whole-document question", "What is this pdf about?", True, None),
    ("Direct fact", "How many days of annual leave do employees get per year?", True, "20"),
    ("Different section", "When is a medical certificate required for sick leave?", True, "3"),
    ("Colloquial wording (may trigger reformulation)", "Can I work from my house some days, and how many?", True, "2"),
    ("Missing topic -> must abstain", "How many weeks of maternity leave do employees get?", False, None),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--file", default=str(ROOT / "sample_documents" / "employee_handbook.pdf"))
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    with httpx.Client(base_url=args.url, timeout=args.timeout) as client:
        try:
            print("GET /health ->", client.get("/health").json())
        except httpx.HTTPError as exc:
            print(f"Cannot reach the API at {args.url}: {exc}. Start it with: uvicorn app.main:app --port 8000")
            return 1
        deps = client.get("/health/dependencies").json()
        print("GET /health/dependencies -> status:", deps["status"])
        for hint in deps["ollama"]["hints"]:
            print("  hint:", hint)
        if deps["status"] != "ok":
            print("Fix the hints above, then run this script again.")
            return 1

        path = Path(args.file)
        with path.open("rb") as handle:
            response = client.post("/upload", files={"file": (path.name, handle)})
        print(f"\nPOST /upload {path.name} -> {response.status_code} {response.json()}")
        if response.status_code not in (200, 201):
            return 1

        failures = 0
        for title, question, expect_found, expect_text in DEMO:
            started = time.perf_counter()
            # No document_id: the service selects the relevant document itself.
            response = client.post("/ask", json={"question": question, "include_trace": True})
            seconds = time.perf_counter() - started
            body = response.json()
            print(f"\n=== {title} ({seconds:.1f} s) ===")
            print("Q:", question)
            if response.status_code != 200:
                print("ERROR:", response.status_code, body)
                failures += 1
                continue
            print("A:", body["answer"])
            print("found:", body["found"], "| type:", body["question_type"], "| path:", " -> ".join(body["trace"]["path"]))
            selection = body["document_selection"]
            scores = ", ".join(f"{c['filename']}={c['relevance_score']}" for c in selection["candidates"])
            print(f"document selection: {selection['mode']} ({selection['reason']}) | scores: {scores}")
            if body["trace"]["retries_used"]:
                print("queries tried:", body["trace"]["queries"])
            for source in body["sources"]:
                print(f"  source: {source['filename']} page={source['page']} pages={source['pages']} "
                      f"score={source['relevance_score']} | {source['content'][:90]!r}...")
            ok = body["found"] == expect_found and (expect_text is None or expect_text in body["answer"])
            print("CHECK:", "as expected" if ok else "REVIEW - differs from the expected behaviour")
            failures += not ok
    print(f"\n{len(DEMO) - failures}/{len(DEMO)} demonstration questions behaved as expected.")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())

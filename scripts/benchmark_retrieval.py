"""Benchmark chunk size / overlap on the sample handbook.

For every chunking configuration it reports:

* fact integrity - share of labelled facts that sit completely inside at least
  one chunk (independent of the embedding model);
* hit@1, hit@k and MRR - whether similarity search ranks a chunk that contains
  the fact at the top, within the top-k, and how high on average;
* similarity of the best correct chunk for answerable questions and of the
  top chunk for unanswerable ones - the data you need before choosing a
  MIN_SIMILARITY floor.

Usage (from the project root, with Ollama running):

    python scripts/benchmark_retrieval.py                      # nomic-embed-text via Ollama
    python scripts/benchmark_retrieval.py --compare-prefixes   # also measure without nomic task prefixes
    python scripts/benchmark_retrieval.py --embeddings fake    # no Ollama: lexical stand-in (only checks the script)
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import Settings  # noqa: E402
from app.services.document_processor import DocumentProcessor  # noqa: E402

ANSWERABLE = [
    ("How many days of annual leave do employees get per year?", "20 days of paid annual leave"),
    ("What are the core working hours?", "10:00 a.m. to 4:00 p.m."),
    ("When is a medical certificate required for sick leave?", "longer than 3 consecutive days"),
    ("How many days per week can I work from home?", "up to 2 days per week"),
    ("How much vacation can I roll over into next year?", "Up to 5 unused days"),
    ("What is the overtime pay rate?", "1.5 times the normal hourly rate"),
    ("How long is the lunch break?", "45 minutes"),
    ("How much is the internet allowance for remote workers?", "INR 1,500 per month"),
    ("How many paid sick days do employees have?", "12 days of paid sick leave"),
    ("How far in advance must leave requests be submitted?", "at least 14 days before"),
    ("When do new employees become eligible for remote work?", "90-day probation"),
    ("How often must passwords be changed?", "every 90 days"),
    ("How many public holidays are there?", "10 paid public holidays"),
    ("Who must approve a fully remote arrangement?", "Vice President"),
]
UNANSWERABLE = [
    "How many weeks of maternity leave do employees get?",
    "What is the dress code?",
    "Does the company offer stock options?",
    "How much paternity leave can fathers take?",
]
CONFIGS = [(400, 60), (600, 90), (800, 120), (1000, 150), (1200, 200)]


def squash(text: str) -> str:
    return " ".join(text.split())


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def make_embedder(kind: str, settings: Settings):
    if kind == "fake":
        from tests.helpers.fakes import FakeEmbeddings

        return FakeEmbeddings()
    from app.services.embedding_service import build_ollama_embeddings

    return build_ollama_embeddings(settings)


def run(kind: str, top_k: int, use_prefixes: bool, settings: Settings) -> None:
    embedder = make_embedder(kind, settings)
    use_prefixes = use_prefixes and kind != "fake"  # task prefixes only mean something to nomic-embed-text
    doc_prefix = settings.resolved_document_prefix if use_prefixes else ""
    query_prefix = settings.resolved_query_prefix if use_prefixes else ""
    text = (ROOT / "sample_documents" / "employee_handbook.txt").read_bytes()
    query_vectors = {q: embedder.embed_query(query_prefix + q) for q, _ in ANSWERABLE}
    query_vectors.update({q: embedder.embed_query(query_prefix + q) for q in UNANSWERABLE})

    label = "fake lexical stand-in" if kind == "fake" else settings.embedding_model
    print(f"\nEmbeddings: {label} | prefixes: {'on' if use_prefixes and doc_prefix else 'off'} | top_k={top_k}")
    print(f"{'size/overlap':>13} {'chunks':>6} {'integrity':>9} {'hit@1':>6} {'hit@k':>6} {'MRR':>5} "
          f"{'min sim (correct)':>17} {'max sim (unanswerable)':>22}")
    for size, overlap in CONFIGS:
        processor = DocumentProcessor(chunk_size=size, chunk_overlap=overlap, max_upload_bytes=10**7, max_pdf_pages=10)
        chunks = processor.process("handbook.txt", text).chunks
        chunk_texts = [squash(c.text) for c in chunks]
        vectors = embedder.embed_documents([doc_prefix + c.text for c in chunks])

        intact = sum(1 for _, fact in ANSWERABLE if any(fact in t for t in chunk_texts))
        hits1 = hitsk = 0
        rr = 0.0
        correct_sims = []
        for question, fact in ANSWERABLE:
            scored = sorted(((cosine(query_vectors[question], v), t) for v, t in zip(vectors, chunk_texts, strict=False)), reverse=True)
            ranks = [i for i, (_, t) in enumerate(scored, start=1) if fact in t]
            if ranks:
                rank = ranks[0]
                rr += 1.0 / rank
                hits1 += rank == 1
                hitsk += rank <= top_k
                correct_sims.append(scored[rank - 1][0])
        unanswerable_top = [max(cosine(query_vectors[q], v) for v in vectors) for q in UNANSWERABLE]
        n = len(ANSWERABLE)
        print(
            f"{size:>6}/{overlap:<6} {len(chunks):>6} {intact / n:>9.0%} {hits1 / n:>6.0%} {hitsk / n:>6.0%} "
            f"{rr / n:>5.2f} {min(correct_sims, default=float('nan')):>17.3f} {max(unanswerable_top):>22.3f}"
        )
    print(
        "\nReading the table: integrity below 100% means some facts are cut by chunk boundaries. "
        "A similarity floor is only safe if it is below 'min sim (correct)'; if 'max sim (unanswerable)' "
        "is close to or above it, similarity alone cannot separate answerable from unanswerable questions."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings", choices=["ollama", "fake"], default="ollama")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--compare-prefixes", action="store_true")
    args = parser.parse_args()
    settings = Settings()
    top_k = args.top_k or settings.top_k
    try:
        run(args.embeddings, top_k, True, settings)
        if args.compare_prefixes and args.embeddings == "ollama":
            run(args.embeddings, top_k, False, settings)
    except Exception as exc:  # most likely Ollama is not running
        print(f"\nBenchmark failed: {type(exc).__name__}: {exc}")
        print("Start Ollama and pull the embedding model, or use --embeddings fake to test the script.")
        sys.exit(1)


if __name__ == "__main__":
    main()

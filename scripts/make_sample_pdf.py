"""Generate sample_documents/employee_handbook.pdf from the TXT handbook.

The PDF version lets you demonstrate one-based page numbers in source
attribution. Usage (from the project root):

    python scripts/make_sample_pdf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.helpers.pdf_factory import build_pdf, text_to_pages  # noqa: E402


def main() -> None:
    source = ROOT / "sample_documents" / "employee_handbook.txt"
    target = ROOT / "sample_documents" / "employee_handbook.pdf"
    pages = text_to_pages(source.read_text(encoding="utf-8"))
    target.write_bytes(build_pdf(pages))
    print(f"Wrote {target.relative_to(ROOT)} ({len(pages)} pages)")


if __name__ == "__main__":
    main()

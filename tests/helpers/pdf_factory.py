"""Build small, valid PDF files without third-party libraries.

Used by the tests (multi-page PDFs, textless PDFs) and by
``scripts/make_sample_pdf.py`` to produce the sample handbook PDF. Text is
drawn with the standard Helvetica font, one line per ``Tj`` operator.
"""

from __future__ import annotations

import textwrap


def _escape(line: str) -> str:
    safe = line.encode("latin-1", "replace").decode("latin-1")
    return safe.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf(pages: list[list[str]], font_size: int = 10, leading: int = 14) -> bytes:
    """Return PDF bytes; ``pages`` is a list of pages, each a list of text lines.

    An empty list for a page produces a page with no text at all.
    """
    objects: list[bytes] = []
    page_count = len(pages)
    # Object numbers: 1 catalog, 2 pages tree, 3 font, then (page, content) pairs.
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(page_count))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    for i, lines in enumerate(pages):
        content_obj = 5 + 2 * i
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_obj} 0 R >>"
            ).encode()
        )
        if lines:
            ops = [f"BT /F1 {font_size} Tf {leading} TL 54 750 Td"]
            for n, line in enumerate(lines):
                ops.append(("" if n == 0 else "T* ") + f"({_escape(line)}) Tj")
            ops.append("ET")
            stream = "\n".join(ops).encode("latin-1")
        else:
            stream = b""
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode()
    return bytes(out)


def text_to_pages(text: str, width: int = 95, lines_per_page: int = 48) -> list[list[str]]:
    """Wrap plain text and paginate it (blank lines are kept as paragraph breaks)."""
    lines: list[str] = []
    for paragraph in text.splitlines():
        if not paragraph.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    return [lines[i : i + lines_per_page] for i in range(0, len(lines), lines_per_page)]

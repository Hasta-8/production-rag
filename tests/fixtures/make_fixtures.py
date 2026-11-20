"""Regenerate the test PDFs. The text is original, written for these tests.

Run with:  uv run --with reportlab python tests/fixtures/make_fixtures.py
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

HERE = Path(__file__).parent
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # has the ﬃ ligature glyph

PAGES = [
    [
        "Quokka Retrieval: A Sample Paper",
        "Introduction",
        "This sample document exists only for automated tests.",
        "The zebra protocol coordinates eﬃcient retrie-",
        "val across many shards of a document collection.",
    ],
    [
        "Method",
        "The quokka index stores dense vectors in a hierarchical graph.",
        "Each node keeps sixteen neighbours, and search walks the graph greedily.",
    ],
    [
        "Results",
        "On the marmot benchmark the system reaches 97.3 percent accuracy",
        "at a median latency of 41 milliseconds per query.",
    ],
]


def write_pdf(path: Path, pages: list[list[str]], title: str | None) -> None:
    c = canvas.Canvas(str(path), pagesize=A4, invariant=1)  # invariant: byte-stable output
    if title:
        c.setTitle(title)
    for lines in pages:
        c.setFont("DV", 11)
        y = A4[1] - 72
        for line in lines:
            c.drawString(72, y, line)
            y -= 16
        c.showPage()
    c.save()


def write_blank(path: Path) -> None:
    c = canvas.Canvas(str(path), pagesize=A4, invariant=1)
    c.rect(100, 100, 200, 200, fill=1)
    c.showPage()
    c.save()


if __name__ == "__main__":
    pdfmetrics.registerFont(TTFont("DV", FONT))
    write_pdf(HERE / "sample_paper.pdf", PAGES, "Quokka Retrieval: A Sample Paper")
    revised = [p[:] for p in PAGES]
    revised[2][1] = "On the marmot benchmark the system reaches 98.1 percent accuracy"
    write_pdf(HERE / "sample_paper_revised.pdf", revised, "Quokka Retrieval: A Sample Paper")
    write_blank(HERE / "image_only.pdf")
    print("wrote fixtures to", HERE)

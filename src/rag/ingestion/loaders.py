"""Load documents into per-page text, keeping page numbers for citations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}
MIN_CHARS_PER_PAGE = 20  # below this on average, the PDF has no usable text layer


class IngestionError(Exception):
    """A file could not be turned into text. The message says why."""


@dataclass(frozen=True)
class PageText:
    page: int  # 1-based, as printed in citations
    text: str


@dataclass
class LoadedDocument:
    path: Path
    pages: list[PageText]
    title: str
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.path.name


def load_pdf(path: Path) -> LoadedDocument:
    try:
        reader = PdfReader(path)
        pages = [
            PageText(page=i + 1, text=page.extract_text() or "")
            for i, page in enumerate(reader.pages)
        ]
    except (PdfReadError, OSError, ValueError) as exc:
        raise IngestionError(f"cannot read PDF: {exc}") from exc
    if not pages:
        raise IngestionError("PDF has no pages")
    total_chars = sum(len(p.text.strip()) for p in pages)
    if total_chars < MIN_CHARS_PER_PAGE * len(pages):
        raise IngestionError("PDF has no usable text layer (scanned or image-only?)")
    meta_title = ""
    if reader.metadata and reader.metadata.title:
        meta_title = str(reader.metadata.title).strip()
    return LoadedDocument(path=path, pages=pages, title=meta_title or path.stem)


def load_text(path: Path) -> LoadedDocument:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise IngestionError(f"cannot read text file: {exc}") from exc
    if not text.strip():
        raise IngestionError("file is empty")
    return LoadedDocument(path=path, pages=[PageText(page=1, text=text)], title=path.stem)


def load_document(path: Path) -> LoadedDocument:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise IngestionError(f"unsupported file type {suffix!r}")
    return load_pdf(path) if suffix == ".pdf" else load_text(path)

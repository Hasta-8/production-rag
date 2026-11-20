"""Ingest files into a versioned index: load, normalise, deduplicate, chunk, embed, store."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from rag.indexing.index import DocumentEntry, VersionedIndex
from rag.ingestion.chunkers import Chunk, chunk_id_for, chunk_pages, count_tokens
from rag.ingestion.loaders import IngestionError, load_document
from rag.ingestion.normalize import make_doc_id, normalize_text

MetadataLookup = Callable[[Path], dict[str, str | int | float | bool | None]]
EMBED_BATCH = 128


@dataclass
class FileResult:
    path: Path
    status: str  # "indexed", "replaced", "duplicate" or "error"
    doc_id: str | None = None
    pages: int = 0
    chunks: int = 0
    tokens: int = 0
    error: str | None = None


def _chroma_metadata(values: dict) -> dict[str, str | int | float | bool]:
    """Chroma accepts only scalar, non-null metadata values."""
    return {k: v for k, v in values.items() if isinstance(v, str | int | float | bool)}


def _to_documents(chunks: list[Chunk], base: dict, version: str) -> list[Document]:
    return [
        Document(
            page_content=c.text,
            metadata=_chroma_metadata(
                {
                    **base,
                    "doc_id": c.doc_id,
                    "chunk_id": c.chunk_id,
                    "page": c.page,
                    "chunk_index": c.chunk_index,
                    "token_count": c.token_count,
                    "content_hash": c.content_hash,
                    "index_version": version,
                }
            ),
        )
        for c in chunks
    ]


def ingest_file(
    path: Path,
    index: VersionedIndex,
    metadata_for: MetadataLookup = lambda _p: {},
    token_counter: Callable[[str], int] = count_tokens,
) -> FileResult:
    info = index.info
    try:
        loaded = load_document(path)
    except IngestionError as exc:
        return FileResult(path, "error", error=str(exc))

    pages = [(p.page, normalize_text(p.text)) for p in loaded.pages]
    doc_id = make_doc_id([text for _, text in pages])
    if doc_id in info.documents:
        entry = info.documents[doc_id]
        return FileResult(path, "duplicate", doc_id, entry.pages, entry.chunks, entry.tokens)

    chunks = chunk_pages(doc_id, pages, info.chunk_size, info.chunk_overlap, token_counter)
    if not chunks:
        return FileResult(path, "error", doc_id, error="no text left after cleaning")

    extra = metadata_for(path)
    title = str(extra.get("title") or loaded.title)
    base = {**extra, "source": loaded.source, "title": title}
    documents = _to_documents(chunks, base, info.version)
    ids = [c.chunk_id for c in chunks]
    try:
        for start in range(0, len(documents), EMBED_BATCH):
            index.store.add_documents(
                documents[start : start + EMBED_BATCH], ids=ids[start : start + EMBED_BATCH]
            )
    except Exception:
        # Roll back partial writes so no chunk is searchable without an index.json entry.
        index.store.delete(ids=ids)
        raise

    # A changed file with the same name replaces the old version's chunks.
    status = "indexed"
    old_id = info.doc_id_for_source(loaded.source)
    if old_id is not None and old_id != doc_id:
        old = info.documents.pop(old_id)
        index.store.delete(ids=[chunk_id_for(old_id, i) for i in range(old.chunks)])
        status = "replaced"

    tokens = sum(c.token_count for c in chunks)
    info.documents[doc_id] = DocumentEntry(
        doc_id=doc_id,
        source=loaded.source,
        title=title,
        pages=len(pages),
        chunks=len(chunks),
        tokens=tokens,
    )
    index.save_info()
    return FileResult(path, status, doc_id, len(pages), len(chunks), tokens)


def ingest_paths(
    paths: Iterable[Path],
    index: VersionedIndex,
    metadata_for: MetadataLookup = lambda _p: {},
    token_counter: Callable[[str], int] = count_tokens,
    on_result: Callable[[FileResult], None] = lambda _r: None,
) -> list[FileResult]:
    results = []
    for path in paths:
        result = ingest_file(path, index, metadata_for, token_counter)
        on_result(result)
        results.append(result)
    return results

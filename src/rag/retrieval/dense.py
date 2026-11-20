"""Baseline dense retrieval (Phase 1). Hybrid search and reranking arrive in Phase 3."""

from __future__ import annotations

from dataclasses import dataclass

from rag.indexing.index import VersionedIndex


@dataclass(frozen=True)
class SearchHit:
    rank: int
    chunk_id: str
    doc_id: str
    title: str
    source: str
    page: int
    score: float  # cosine similarity, higher is closer
    text: str


def dense_search(index: VersionedIndex, query: str, k: int = 5) -> list[SearchHit]:
    results = index.store.similarity_search_with_score(query, k=k)
    hits = []
    for rank, (doc, distance) in enumerate(results, start=1):
        meta = doc.metadata
        hits.append(
            SearchHit(
                rank=rank,
                # The vector store's own ID is the chunk ID; metadata may also carry it.
                chunk_id=str(meta.get("chunk_id") or doc.id or ""),
                doc_id=str(meta.get("doc_id", "")),
                title=str(meta.get("title", "")),
                source=str(meta.get("source", "")),
                page=int(meta.get("page", 0)),
                # Chroma returns cosine *distance* (lower is closer) for this collection.
                score=1.0 - float(distance),
                text=doc.page_content,
            )
        )
    return hits

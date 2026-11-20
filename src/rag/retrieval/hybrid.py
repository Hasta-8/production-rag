"""One retriever for every mode: dense, BM25, hybrid (RRF) and reranked variants.

Every candidate keeps its rank and score from each stage, so the trace (and later the
UI's trace panel and compare mode) can show why a chunk ended up where it did.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rag.indexing.index import VersionedIndex
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.dense import dense_search
from rag.retrieval.rerank import Reranker

MODES = ("dense", "bm25", "hybrid", "dense_rerank", "hybrid_rerank")


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]], k: int = 60, weights: Mapping[str, float] | None = None
) -> list[tuple[str, float]]:
    """RRF: score(d) = sum over rankings of weight / (k + rank of d), ranks starting at 1.

    Uses ranks only, so BM25 scores and cosine similarities never need a common scale.
    Ties are broken by first appearance (earlier rankings first) to stay deterministic.
    """
    scores: dict[str, float] = {}
    order: dict[str, int] = {}
    for name, ids in rankings.items():
        weight = (weights or {}).get(name, 1.0)
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
            order.setdefault(doc_id, len(order))
    return sorted(scores.items(), key=lambda item: (-item[1], order[item[0]]))


@dataclass
class Candidate:
    chunk_id: str
    doc_id: str
    title: str
    page: int
    text: str
    meta: dict
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    fused_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    rank: int = 0  # final position, 1-based

    @property
    def score(self) -> float:
        """The score the final ranking is based on."""
        for value in (self.rerank_score, self.rrf_score, self.dense_score, self.bm25_score):
            if value is not None:
                return value
        return 0.0

    def summary(self) -> dict:
        def r(x):
            return round(x, 4) if isinstance(x, float) else x

        return {
            "chunk_id": self.chunk_id,
            "title": self.title,
            "page": self.page,
            "rank": self.rank,
            "dense_rank": self.dense_rank,
            "dense_score": r(self.dense_score),
            "bm25_rank": self.bm25_rank,
            "bm25_score": r(self.bm25_score),
            "fused_rank": self.fused_rank,
            "rerank_score": r(self.rerank_score),
        }


class Retriever:
    def __init__(
        self,
        index: VersionedIndex,
        bm25: BM25Index,
        reranker: Reranker | None = None,
        mode: str = "hybrid_rerank",
        candidates: int = 20,
        rrf_k: int = 60,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown retrieval mode {mode!r}; choose from {MODES}")
        if mode.endswith("_rerank") and reranker is None:
            raise ValueError(f"mode {mode!r} needs a reranker")
        self.index = index
        self.bm25 = bm25
        self.reranker = reranker
        self.mode = mode
        self.candidates = candidates
        self.rrf_k = rrf_k
        self.weights = dict(weights or {})

    @property
    def uses_reranker(self) -> bool:
        return self.mode.endswith("_rerank")

    def _candidate(self, chunk_id: str) -> Candidate:
        stored = self.bm25.by_id[chunk_id]
        meta = stored.meta
        return Candidate(
            chunk_id=chunk_id,
            doc_id=str(meta.get("doc_id", "")),
            title=str(meta.get("title", "")),
            page=int(meta.get("page", 0)),
            text=stored.text,
            meta=meta,
        )

    def search(self, query: str, k: int = 10) -> list[Candidate]:
        pool = max(k, self.candidates)
        cands: dict[str, Candidate] = {}
        rankings: dict[str, list[str]] = {}

        if self.mode in ("dense", "hybrid", "dense_rerank", "hybrid_rerank"):
            hits = dense_search(self.index, query, k=pool)
            rankings["dense"] = [h.chunk_id for h in hits]
            for h in hits:
                c = cands.setdefault(h.chunk_id, self._candidate(h.chunk_id))
                c.dense_rank, c.dense_score = h.rank, h.score
        if self.mode in ("bm25", "hybrid", "hybrid_rerank"):
            hits = self.bm25.search(query, k=pool)
            rankings["bm25"] = [cid for cid, _ in hits]
            for rank, (cid, score) in enumerate(hits, start=1):
                c = cands.setdefault(cid, self._candidate(cid))
                c.bm25_rank, c.bm25_score = rank, score

        if len(rankings) > 1:
            fused = reciprocal_rank_fusion(rankings, k=self.rrf_k, weights=self.weights)
            for rank, (cid, score) in enumerate(fused, start=1):
                cands[cid].fused_rank, cands[cid].rrf_score = rank, score
            ordered = [cands[cid] for cid, _ in fused][:pool]
        else:
            (only,) = rankings.values()
            ordered = [cands[cid] for cid in only][:pool]

        if self.uses_reranker and ordered:
            scores = self.reranker.score(query, [c.text for c in ordered])
            for c, s in zip(ordered, scores, strict=True):
                c.rerank_score = s
            ordered.sort(key=lambda c: -c.rerank_score)

        for rank, c in enumerate(ordered, start=1):
            c.rank = rank
        return ordered[:k]

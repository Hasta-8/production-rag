"""Keyword search with BM25, built from the chunks already stored in the index.

BM25 indexes do not update incrementally, so this one is rebuilt whenever it is
opened. For ~3,000 chunks that takes well under a second.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

# Keep tokens like "nf4", "1.4t", "ms-marco" and "gpt-4" whole: exact terms are BM25's job.
_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
_STOP = set(
    "a an the of and or to in on for with by from at as is are was were be been it its this "
    "that these those which who what how why when where than then into we our their they "
    "does do did can could not no".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass(frozen=True)
class StoredChunk:
    chunk_id: str
    text: str
    meta: dict


class BM25Index:
    def __init__(self, chunks: list[StoredChunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        tokenized = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized, k1=k1, b=b) if chunks else None

    @classmethod
    def from_store(cls, store) -> BM25Index:
        data = store.get(include=["documents", "metadatas"])
        chunks = [
            StoredChunk(cid, text, meta)
            for cid, text, meta in zip(
                data["ids"], data["documents"], data["metadatas"], strict=True
            )
        ]
        chunks.sort(key=lambda c: c.chunk_id)  # deterministic order, so ties break the same way
        return cls(chunks)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
        return [(self.chunks[i].chunk_id, float(scores[i])) for i in order if scores[i] > 0]

"""Evidence focusing: give the model the best sentences instead of whole chunks.

A small model sometimes declines although the answer is in its context, buried in a
~1,000-character chunk full of table fragments and citations (P26, q23). The focus step
scores every sentence of the retrieved candidates against the question with the same
cross-encoder used for reranking, and keeps the best few. The prompt gets shorter and the
answering sentence comes first. Sentences are cheap to score: about 30 tokens each,
against about 250 for a chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.retrieval.hybrid import Candidate
from rag.retrieval.rerank import Reranker

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'])")
MIN_CHARS = 25  # shorter pieces are fragments (axis labels, "Table 3.")
MAX_CHARS = 400  # longer "sentences" are usually run-on table text; keep their start


def split_sentences(text: str) -> list[str]:
    flat = " ".join(text.split())
    out = []
    for s in _SENTENCE.split(flat):
        s = s.strip()
        if len(s) >= MIN_CHARS:
            out.append(s[:MAX_CHARS])
    return out


@dataclass
class FocusedChunk:
    candidate: Candidate
    sentences: list[str]  # in their order within the chunk
    best_score: float


def focus(
    question: str, candidates: list[Candidate], reranker: Reranker, n_sentences: int = 6
) -> list[FocusedChunk]:
    """The ``n_sentences`` best sentences, grouped by chunk, best chunk first.

    Each sentence is scored together with its paper title, so "our model" style sentences
    keep a hint of which paper they come from. Overlapping chunks repeat sentences; a
    repeated sentence is scored once, for its highest-ranked chunk.
    """
    pairs: list[tuple[int, int, str]] = []  # (candidate index, sentence index, sentence)
    seen: set[str] = set()
    for ci, cand in enumerate(candidates):
        for si, sentence in enumerate(split_sentences(cand.text)):
            if sentence not in seen:
                seen.add(sentence)
                pairs.append((ci, si, sentence))
    if not pairs:
        return []
    scores = reranker.score(question, [f"{candidates[ci].title}. {s}" for ci, _, s in pairs])
    top = sorted(zip(scores, pairs, strict=True), key=lambda x: -x[0])[:n_sentences]

    grouped: dict[int, list[tuple[int, str, float]]] = {}
    for score, (ci, si, sentence) in top:
        grouped.setdefault(ci, []).append((si, sentence, score))
    chunks = [
        FocusedChunk(
            candidates[ci],
            [s for _, s, _ in sorted(items)],
            max(score for _, _, score in items),
        )
        for ci, items in grouped.items()
    ]
    return sorted(chunks, key=lambda c: -c.best_score)

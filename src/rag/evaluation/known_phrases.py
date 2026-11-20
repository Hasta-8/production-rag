"""Known-phrase check (Phase 1 exit gate): does searching a sentence copied from the
corpus return the page it came from?

This is a sanity check of ingestion and indexing, not a retrieval-quality benchmark:
the queries are verbatim text, which is the easiest possible case for search. Real
retrieval quality is measured on the golden set and BEIR SciFact (Phase 3).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from rag.retrieval.hybrid import Retriever

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z]{3,}")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_VENUE = re.compile(r"Proceedings|arXiv|Conference|Journal|preprint|In Advances", re.IGNORECASE)


def looks_like_references(text: str) -> bool:
    """Heuristic: a chunk from a bibliography has many years and venue words."""
    return len(_YEAR.findall(text)) >= 4 and len(_VENUE.findall(text)) >= 2


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


@dataclass
class PhraseResult:
    phrase: str
    expected_doc: str
    expected_page: int
    expected_chunk: str
    title: str
    rank_of_page: int | None  # 1-based rank of the first hit on the right doc and page
    top_title: str
    top_page: int
    is_reference: bool  # the source chunk looks like a bibliography
    top_is_right_paper: bool
    top_contains_phrase: bool  # top hit holds the phrase verbatim (it may occur in several places)


def pick_phrase(text: str, rng: random.Random, min_words: int = 8, max_words: int = 25) -> str:
    """Pick a prose-like sentence (mostly real words, not table rows or references)."""
    flat = " ".join(text.split())
    candidates = []
    for sentence in _SENTENCE_SPLIT.split(flat):
        words = sentence.split()
        if not min_words <= len(words) <= max_words:
            continue
        wordlike = sum(bool(_WORD.fullmatch(w.strip(".,;:()"))) for w in words)
        if wordlike / len(words) >= 0.7:
            candidates.append(sentence)
    return rng.choice(candidates) if candidates else ""


def run_known_phrase_check(
    retriever: Retriever, samples: int = 50, k: int = 5, seed: int = 7
) -> list[PhraseResult]:
    data = retriever.index.store.get(include=["documents", "metadatas"])
    chunks = list(zip(data["ids"], data["documents"], data["metadatas"], strict=True))
    chunks.sort(key=lambda c: c[0])  # stable order, so the seed gives the same sample
    rng = random.Random(seed)
    rng.shuffle(chunks)

    results: list[PhraseResult] = []
    for chunk_id, text, meta in chunks:
        if len(results) >= samples:
            break
        phrase = pick_phrase(text, rng)
        if not phrase:
            continue
        hits = retriever.search(phrase, k=k)
        rank = next(
            (h.rank for h in hits if h.doc_id == meta["doc_id"] and h.page == meta["page"]),
            None,
        )
        results.append(
            PhraseResult(
                phrase=phrase,
                expected_doc=str(meta["doc_id"]),
                expected_page=int(meta["page"]),
                expected_chunk=chunk_id,
                title=str(meta.get("title", "")),
                rank_of_page=rank,
                top_title=hits[0].title if hits else "",
                top_page=hits[0].page if hits else 0,
                is_reference=looks_like_references(text),
                top_is_right_paper=bool(hits) and hits[0].doc_id == meta["doc_id"],
                top_contains_phrase=bool(hits) and _norm(phrase) in _norm(hits[0].text),
            )
        )
    return results


def summarize(results: list[PhraseResult], k: int) -> dict[str, float]:
    n = len(results) or 1
    at1 = sum(r.rank_of_page == 1 for r in results)
    atk = sum(r.rank_of_page is not None for r in results)
    mrr = sum(1 / r.rank_of_page for r in results if r.rank_of_page) / n
    exact = sum(r.rank_of_page == 1 or r.top_contains_phrase for r in results)
    paper = sum(r.top_is_right_paper for r in results)
    return {
        "samples": len(results),
        "page_at_1": at1 / n,
        f"page_at_{k}": atk / n,
        "mrr": mrr,
        "page_or_verbatim_at_1": exact / n,
        "paper_at_1": paper / n,
    }

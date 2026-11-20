"""Golden-set evaluation: retrieval metrics (no LLM) and answer metrics (with the LLM).

Retrieval and generation are measured separately: if the right page is never
retrieved, no prompt can fix the answer.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel

from rag.generation.answer import Answer, AnswerSettings, answer_question, relevance_signal
from rag.generation.llm import LLMClient
from rag.indexing.index import VersionedIndex
from rag.retrieval.hybrid import Retriever

KS = (1, 3, 5, 10)


class GoldenQuestion(BaseModel):
    id: str
    question: str
    answerable: bool = True
    arxiv_id: str | None = None
    evidence_phrases: list[str] = []  # v1 labels
    evidence_pages: list[int] = []  # v2 labels: every page that states the answer
    keywords: list[str] = []
    reference: str | None = None
    note: str | None = None


def load_golden(path: Path) -> list[GoldenQuestion]:
    data = yaml.safe_load(path.read_text())
    return [GoldenQuestion(**q) for q in data["questions"]]


def golden_version(path: Path) -> str:
    return str(yaml.safe_load(path.read_text()).get("version", "?"))


def page_texts(index: VersionedIndex) -> dict[tuple[str, int], str]:
    """(arxiv_id, page) -> lower-case, whitespace-normalised text of that page."""
    data = index.store.get(include=["documents", "metadatas"])
    pages: dict[tuple[str, int], str] = defaultdict(str)
    for text, meta in zip(data["documents"], data["metadatas"], strict=True):
        key = (str(meta.get("arxiv_id", "")), int(meta["page"]))
        pages[key] += " " + " ".join(text.split()).lower()
    return pages


LABELS = ("v2", "v1")


def resolve_evidence(
    questions: list[GoldenQuestion], pages: dict[tuple[str, int], str], labels: str = "v2"
) -> dict[str, set[int]]:
    """Evidence pages for each answerable question.

    v1: pages containing an evidence phrase. v2: the listed evidence_pages (falling back to
    v1 for a question without them). Both label sets are checked against the corpus, and a
    phrase or page that does not exist raises.
    """
    if labels not in LABELS:
        raise ValueError(f"unknown labels {labels!r}; choose from {LABELS}")
    resolved: dict[str, set[int]] = {}
    problems = []
    for q in questions:
        if not q.answerable:
            continue
        found: set[int] = set()
        for phrase in q.evidence_phrases:
            hits = {p for (aid, p), text in pages.items() if aid == q.arxiv_id and phrase in text}
            if not hits:
                problems.append(f"{q.id}: phrase {phrase!r} not found in {q.arxiv_id}")
            found |= hits
        for page in q.evidence_pages:
            if (q.arxiv_id, page) not in pages:
                problems.append(f"{q.id}: page {page} not found in {q.arxiv_id}")
        if labels == "v2" and q.evidence_pages:
            found = set(q.evidence_pages)
        resolved[q.id] = found
    if problems:
        raise ValueError("golden set does not match the corpus:\n" + "\n".join(problems))
    return resolved


def keyword_recall(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    low = text.lower()
    return sum(any(alt in low for alt in kw.split("|")) for kw in keywords) / len(keywords)


# ---------------------------------------------------------------- retrieval


@dataclass
class RetrievalRow:
    id: str
    answerable: bool
    best_score: float
    first_hit_rank: int | None = None  # rank of the first chunk on an evidence page
    paper_rank: int | None = None  # rank of the first chunk from the right paper


def evaluate_retrieval(
    retriever: Retriever,
    questions: list[GoldenQuestion],
    k: int = max(KS),
    thresholds: list[float] | None = None,
    labels: str = "v2",
) -> tuple[list[RetrievalRow], dict]:
    evidence = resolve_evidence(questions, page_texts(retriever.index), labels)
    rows = []
    for q in questions:
        hits = retriever.search(q.question, k=k)
        _, best = relevance_signal(hits, retriever.uses_reranker)
        row = RetrievalRow(q.id, q.answerable, round(best, 4) if best is not None else 0.0)
        if q.answerable:
            for h in hits:
                same_paper = str(h.meta.get("arxiv_id", "")) == q.arxiv_id
                if same_paper and row.paper_rank is None:
                    row.paper_rank = h.rank
                if same_paper and h.page in evidence[q.id] and row.first_hit_rank is None:
                    row.first_hit_rank = h.rank
        rows.append(row)

    ans = [r for r in rows if r.answerable]
    n = len(ans) or 1
    summary = {
        f"page_recall@{k_}": sum(
            r.first_hit_rank is not None and r.first_hit_rank <= k_ for r in ans
        )
        / n
        for k_ in KS
    }
    summary["page_mrr"] = sum(1 / r.first_hit_rank for r in ans if r.first_hit_rank) / n
    summary["paper_at_1"] = sum(r.paper_rank == 1 for r in ans) / n
    summary["calibration"] = threshold_table(rows, thresholds)
    return rows, summary


DENSE_THRESHOLDS = [x / 100 for x in range(50, 86, 5)]
RERANK_THRESHOLDS = [float(x) for x in range(-8, 9)]


def threshold_table(rows: list[RetrievalRow], thresholds: list[float] | None = None) -> list[dict]:
    """For each relevance threshold: share of answerable kept and unanswerable rejected."""
    ans = [r.best_score for r in rows if r.answerable]
    una = [r.best_score for r in rows if not r.answerable]
    table = []
    for t in thresholds or DENSE_THRESHOLDS:
        table.append(
            {
                "threshold": t,
                "answerable_kept": sum(s >= t for s in ans) / (len(ans) or 1),
                "unanswerable_rejected": sum(s < t for s in una) / (len(una) or 1),
            }
        )
    return table


# ---------------------------------------------------------------- answers


@dataclass
class AnswerRow:
    id: str
    answerable: bool
    abstained: bool
    correct_abstention: bool
    keyword_recall: float | None
    cited: list[str]
    citation_on_evidence: bool | None  # some cited source is on an evidence page
    citations_inferred: bool
    seconds: float
    answer: str
    sources: list[dict] = field(default_factory=list)
    model: str | None = None
    retries: int = 0  # agent corrective steps
    steps: list[str] = field(default_factory=list)  # trace step names, in order


def evaluate_answers(
    retriever: Retriever,
    llm: LLMClient,
    questions: list[GoldenQuestion],
    settings: AnswerSettings,
    on_row: Callable[[AnswerRow], None] = lambda _r: None,
    labels: str = "v2",
    answer_fn: Callable[[str], Answer] | None = None,
) -> tuple[list[AnswerRow], dict]:
    """``answer_fn`` answers one question (default: the single-pass pipeline)."""
    if answer_fn is None:

        def answer_fn(question: str) -> Answer:
            return answer_question(question, retriever, llm, settings)

    index: VersionedIndex = retriever.index
    pages = page_texts(index)
    evidence = resolve_evidence(questions, pages, labels)
    data = index.store.get(include=["metadatas"])
    arxiv_by_chunk = {
        cid: str(m.get("arxiv_id", ""))
        for cid, m in zip(data["ids"], data["metadatas"], strict=True)
    }
    rows = []
    for q in questions:
        start = time.perf_counter()
        ans: Answer = answer_fn(q.question)
        secs = round(time.perf_counter() - start, 1)
        on_evidence = None
        kw = None
        if q.answerable and not ans.abstained:
            on_evidence = any(
                arxiv_by_chunk.get(s.chunk_id) == q.arxiv_id and s.page in evidence[q.id]
                for s in ans.cited_sources
            )
            kw = round(keyword_recall(ans.text, q.keywords), 3)
        row = AnswerRow(
            id=q.id,
            answerable=q.answerable,
            abstained=ans.abstained,
            correct_abstention=ans.abstained != q.answerable,
            keyword_recall=kw,
            cited=ans.cited,
            citation_on_evidence=on_evidence,
            citations_inferred=ans.citations_inferred,
            seconds=secs,
            answer=ans.text,
            sources=[
                {"tag": s.tag, "title": s.title, "page": s.page, "score": s.score}
                for s in ans.sources
            ],
            model=ans.model,
            retries=ans.retries,
            steps=[s.name for s in ans.trace.steps],
        )
        on_row(row)
        rows.append(row)
    return rows, summarize_answers(rows)


def summarize_answers(rows: list[AnswerRow]) -> dict:
    answerable = [r for r in rows if r.answerable]
    unanswerable = [r for r in rows if not r.answerable]
    answered = [r for r in answerable if not r.abstained]
    summary = {
        "answerable_answered": len(answered) / (len(answerable) or 1),
        "unanswerable_declined": sum(r.abstained for r in unanswerable) / (len(unanswerable) or 1),
        "keyword_recall_mean": (
            sum(r.keyword_recall for r in answered) / len(answered) if answered else None
        ),
        "citation_on_evidence": (
            sum(bool(r.citation_on_evidence) for r in answered) / len(answered)
            if answered
            else None
        ),
        "answers_with_citations": (
            sum(bool(r.cited) for r in answered) / len(answered) if answered else None
        ),
        "citations_inferred": sum(r.citations_inferred for r in answered),
        "questions_with_retries": sum(r.retries > 0 for r in rows),
        "max_retries_used": max((r.retries for r in rows), default=0),
        "median_seconds": sorted(r.seconds for r in rows)[len(rows) // 2] if rows else None,
    }
    return summary


def rescore_answers(
    rows: list[dict],
    questions: list[GoldenQuestion],
    evidence: dict[str, set[int]],
    arxiv_by_title: dict[str, str],
) -> tuple[list[AnswerRow], dict]:
    """Re-score a saved answer run against other evidence labels, without the LLM.

    Saved rows keep each source's title and page and the cited tags, which is all the
    citation-on-evidence metric needs. Keyword recall is recomputed from the saved text.
    """
    by_id = {q.id: q for q in questions}
    out = []
    for raw in rows:
        r = AnswerRow(**raw)
        q = by_id[r.id]
        if q.answerable and not r.abstained:
            cited = [s for s in r.sources if str(s["tag"]) in {str(t) for t in r.cited}]
            r.citation_on_evidence = any(
                arxiv_by_title.get(s["title"]) == q.arxiv_id and s["page"] in evidence[q.id]
                for s in cited
            )
            r.keyword_recall = round(keyword_recall(r.answer, q.keywords), 3)
        out.append(r)
    return out, summarize_answers(out)


def save_results(path: Path, label: str, config: dict, summary: dict, rows: list) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    out = path / f"{stamp}_{label}.json"
    out.write_text(
        json.dumps(
            {
                "label": label,
                "date": stamp,
                "config": config,
                "summary": summary,
                "rows": [asdict(r) for r in rows],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return out

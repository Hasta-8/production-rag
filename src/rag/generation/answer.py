"""Baseline RAG (Phase 2): retrieve, check relevance, generate a cited answer, check citations.

Phase 3 adds hybrid retrieval and reranking; Phase 4 turns this into a LangGraph agent
with a rewrite loop and model fallback. The answer and trace shapes stay the same.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag.generation.citations import check_citations, clean_answer, is_refusal
from rag.generation.llm import LLMClient, TokenCallback
from rag.generation.prompts import PROMPT_VERSION, REFUSAL, PromptSource, build_messages
from rag.indexing.index import VersionedIndex
from rag.retrieval.dense import SearchHit, dense_search
from rag.tracing import Trace


@dataclass(frozen=True)
class AnswerSettings:
    retrieve_k: int = 10  # candidates retrieved (shown in the trace)
    context_k: int = 3  # sources passed to the LLM (D2: small prompts on a CPU)
    min_relevance: float = 0.60  # calibrated on the golden set: rejects off-topic only
    max_tokens: int = 150  # D3: short answers


@dataclass
class Source:
    tag: str
    chunk_id: str
    doc_id: str
    title: str
    page: int
    score: float
    text: str


@dataclass
class Answer:
    question: str
    text: str
    abstained: bool
    abstain_reason: str | None
    sources: list[Source]  # what the model was shown, tagged S1..Sn
    cited: list[str]  # tags actually cited
    citations_inferred: bool
    model: str | None
    prompt_version: str = PROMPT_VERSION
    trace: Trace = field(default_factory=Trace)

    @property
    def cited_sources(self) -> list[Source]:
        by_tag = {s.tag: s for s in self.sources}
        return [by_tag[t] for t in self.cited if t in by_tag]


_SENTENCE_END = re.compile(r"[.!?](?:\s*\[[^\]]*\])*(?=\s|$)|\][.!?]?(?=\s|$)")


def trim_incomplete_sentence(text: str) -> tuple[str, bool]:
    """Cut text back to its last complete sentence (a sentence may end in a citation tag).

    Returns the text and whether anything was removed. If no sentence is complete, the
    text is returned unchanged rather than emptied.
    """
    stripped = text.rstrip()
    ends = list(_SENTENCE_END.finditer(stripped))
    if not ends or ends[-1].end() == len(stripped):
        return stripped, False
    return stripped[: ends[-1].end()].rstrip(), True


def _hit_summary(h: SearchHit) -> dict:
    return {"chunk_id": h.chunk_id, "title": h.title, "page": h.page, "score": round(h.score, 4)}


def answer_question(
    question: str,
    index: VersionedIndex,
    llm: LLMClient,
    settings: AnswerSettings = AnswerSettings(),
    on_token: TokenCallback | None = None,
) -> Answer:
    trace = Trace()
    question = question.strip()

    with trace.step("retrieve", method="dense", k=settings.retrieve_k) as rec:
        hits = dense_search(index, question, k=settings.retrieve_k)
        rec["hits"] = [_hit_summary(h) for h in hits]

    best = hits[0].score if hits else 0.0
    with trace.step("grade", min_relevance=settings.min_relevance) as rec:
        relevant = best >= settings.min_relevance
        rec.update(best_score=round(best, 4), verdict="relevant" if relevant else "too weak")

    sources = [
        Source(f"S{i}", h.chunk_id, h.doc_id, h.title, h.page, round(h.score, 4), h.text)
        for i, h in enumerate(hits[: settings.context_k], start=1)
    ]

    if not relevant:
        reason = f"best relevance {best:.2f} is below the threshold {settings.min_relevance:.2f}"
        return Answer(question, REFUSAL, True, reason, sources, [], False, None, trace=trace)

    prompt_sources = [PromptSource(s.tag, s.title, s.page, s.text) for s in sources]
    with trace.step("generate", model=llm.name, max_tokens=settings.max_tokens) as rec:
        gen = llm.generate(build_messages(question, prompt_sources), settings.max_tokens, on_token)
        rec.update(
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
            first_token_ms=gen.first_token_ms,
            sources=[s.tag for s in sources],
        )

    text, removed = clean_answer(gen.text)
    if removed:
        trace.steps[-1].data["cleaned"] = removed

    if is_refusal(text):
        with trace.step("citations") as rec:
            rec["result"] = "model declined to answer"
        return Answer(
            question,
            REFUSAL,
            True,
            "the model found no answer in the sources",
            sources,
            [],
            False,
            gen.model,
            trace=trace,
        )

    if gen.completion_tokens >= settings.max_tokens:
        # Hit the token cap: drop a half-finished last sentence rather than show it.
        text, cut = trim_incomplete_sentence(text)
        trace.steps[-1].data["truncated"] = cut

    with trace.step("citations") as rec:
        check = check_citations(text, prompt_sources)
        rec.update(
            cited=check.cited,
            invalid=check.invalid,
            inferred=check.inferred,
            uncited_sentences=check.uncited_sentences,
        )

    return Answer(
        question,
        check.text,
        False,
        None,
        sources,
        check.cited,
        check.inferred,
        gen.model,
        trace=trace,
    )

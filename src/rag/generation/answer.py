"""Answering steps: retrieve, grade the evidence, generate a cited answer, check it.

``answer_question`` runs them once (the Phase 2–3 pipeline, kept for compare mode).
The Phase 4 agent (``rag.agent``) reuses the same steps inside a bounded correction loop,
so both produce the same ``Answer`` and trace shapes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag.generation.citations import check_citations, clean_answer, is_refusal
from rag.generation.entities import check_entities, question_entities
from rag.generation.llm import LLMClient, TokenCallback
from rag.generation.prompts import PROMPT_VERSION, REFUSAL, PromptSource, build_messages
from rag.retrieval.hybrid import Candidate, Retriever
from rag.tracing import Trace


@dataclass(frozen=True)
class AnswerSettings:
    retrieve_k: int = 10  # candidates retrieved (shown in the trace)
    context_k: int = 3  # sources passed to the LLM (D2: small prompts on a CPU)
    min_relevance: float = 0.60  # dense cosine threshold (modes without a reranker)
    min_rerank_score: float = 2.0  # cross-encoder logit threshold (reranked modes)
    max_tokens: int = 150  # D3: short answers
    # Agent loop (Phase 4)
    max_retries: int = 2  # corrective steps per question (query rewrites + focus retry)
    # Focus retry: off by default. It answered trap questions in E12 (u01) and in E12b's
    # held-out check (t02): the reranker scores near-miss facts about a similar model as
    # relevant, so neither the model's refusal nor the gate can be overridden safely.
    focus_retry: bool = False
    focus_sentences: int = 6  # sentences kept by the focus retry
    focus_gate: bool = True  # focused sentences must pass min_rerank_score (E12b)
    rewrite_max_tokens: int = 40
    # E14: every name in the question must appear in a cited source (P35)
    entity_check: bool = True


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
    pipeline: str = "single"  # "single" (one pass) or "agent" (Phase 4 loop)
    retries: int = 0  # corrective steps the agent took

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


def relevance_signal(cands: list[Candidate], reranked: bool) -> tuple[str, float | None]:
    """The score used to decide whether the evidence is good enough.

    With a reranker: the best cross-encoder score. Otherwise: the best dense cosine score
    among the candidates (BM25 scores are not comparable across queries). BM25-only
    retrieval has no usable signal, so it returns None and is not gated.
    """
    if not cands:
        return ("none", 0.0)
    if reranked:
        return ("rerank", max(c.rerank_score for c in cands))
    dense = [c.dense_score for c in cands if c.dense_score is not None]
    return ("dense", max(dense)) if dense else ("none", None)


def make_sources(hits: list[Candidate], k: int) -> list[Source]:
    return [
        Source(f"S{i}", h.chunk_id, h.doc_id, h.title, h.page, round(h.score, 4), h.text)
        for i, h in enumerate(hits[:k], start=1)
    ]


@dataclass
class Grade:
    relevant: bool
    signal: str
    best: float | None
    threshold: float

    @property
    def reason(self) -> str:
        shown = "none" if self.best is None else f"{self.best:.2f}"
        return f"best {self.signal} score {shown} is below the threshold {self.threshold:.2f}"


def grade_evidence(
    trace: Trace, hits: list[Candidate], reranked: bool, settings: AnswerSettings
) -> Grade:
    """Is the best retrieved evidence strong enough to try answering? (No LLM call, D7.)"""
    signal, best = relevance_signal(hits, reranked)
    threshold = settings.min_rerank_score if signal == "rerank" else settings.min_relevance
    with trace.step("grade", signal=signal, threshold=threshold) as rec:
        relevant = bool(hits) and (best is None or best >= threshold)
        rec.update(
            best_score=None if best is None else round(best, 4),
            verdict="relevant" if relevant else "too weak",
        )
    return Grade(relevant, signal, best, threshold)


@dataclass
class Attempt:
    """One generation and its check."""

    text: str
    refused: bool
    cited: list[str]
    inferred: bool
    model: str
    rejected: str | None = None  # why a check withdrew the answer (E14 entity check)


def generate_and_check(
    trace: Trace,
    question: str,
    sources: list[Source],
    llm: LLMClient,
    settings: AnswerSettings,
    on_token: TokenCallback | None = None,
) -> Attempt:
    """Generate from the given sources, clean the output, then check refusal and citations."""
    prompt_sources = [PromptSource(s.tag, s.title, s.page, s.text) for s in sources]
    with trace.step("generate", model=llm.name, max_tokens=settings.max_tokens) as rec:
        gen = llm.generate(build_messages(question, prompt_sources), settings.max_tokens, on_token)
        rec.update(
            model=gen.model,
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
            first_token_ms=gen.first_token_ms,
            sources=[s.tag for s in sources],
        )
        if gen.fallback_from:
            rec.update(fallback_from=gen.fallback_from, fallback_reason=gen.fallback_reason)
    gen_step = trace.steps[-1].data

    text, removed = clean_answer(gen.text)
    if removed:
        gen_step["cleaned"] = removed

    if is_refusal(text):
        with trace.step("check") as rec:
            rec["result"] = "model declined to answer"
        return Attempt(REFUSAL, True, [], False, gen.model)

    if gen.completion_tokens >= settings.max_tokens:
        # Hit the token cap: drop a half-finished last sentence rather than show it.
        text, cut = trim_incomplete_sentence(text)
        gen_step["truncated"] = cut

    with trace.step("check") as rec:
        check = check_citations(text, prompt_sources)
        rec.update(
            result="answered",
            cited=check.cited,
            invalid=check.invalid,
            inferred=check.inferred,
            uncited_sentences=check.uncited_sentences,
        )
    if settings.entity_check and question_entities(question):
        by_tag = {s.tag: s for s in sources}
        cited = [by_tag[t] for t in check.cited if t in by_tag] or sources
        with trace.step("entities") as rec:
            verdict = check_entities(question, [(s.tag, s.title, s.text) for s in cited])
            rec.update(
                entities=verdict.entities,
                supported=verdict.supported,
                verdict="passed" if verdict.passed else "rejected",
            )
            if not verdict.passed:
                rec["withdrawn_answer"] = check.text
        if not verdict.passed:
            return Attempt(REFUSAL, False, [], False, gen.model, rejected=verdict.reason)
    return Attempt(check.text, False, check.cited, check.inferred, gen.model)


def answer_question(
    question: str,
    retriever: Retriever,
    llm: LLMClient,
    settings: AnswerSettings = AnswerSettings(),
    on_token: TokenCallback | None = None,
) -> Answer:
    """Single pass (Phases 2–3): retrieve, grade, generate, check. No retries."""
    trace = Trace()
    question = question.strip()

    with trace.step("retrieve", mode=retriever.mode, k=settings.retrieve_k) as rec:
        hits = retriever.search(question, k=settings.retrieve_k)
        rec["hits"] = [h.summary() for h in hits]

    grade = grade_evidence(trace, hits, retriever.uses_reranker, settings)
    sources = make_sources(hits, settings.context_k)
    if not grade.relevant:
        return Answer(question, REFUSAL, True, grade.reason, sources, [], False, None, trace=trace)

    att = generate_and_check(trace, question, sources, llm, settings, on_token)
    if att.refused:
        reason = "the model found no answer in the sources"
        return Answer(question, REFUSAL, True, reason, sources, [], False, att.model, trace=trace)
    if att.rejected:
        return Answer(
            question, REFUSAL, True, att.rejected, sources, [], False, att.model, trace=trace
        )
    return Answer(
        question, att.text, False, None, sources, att.cited, att.inferred, att.model, trace=trace
    )

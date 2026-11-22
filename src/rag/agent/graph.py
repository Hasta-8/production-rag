"""Phase 4: the agentic correction loop, as a LangGraph state machine.

    retrieve -> grade --relevant--> generate -> check --answered--> done
                  |                    ^          |
                  | too weak           |          | model declined, evidence was relevant
                  v                    |          v
               rewrite -> retrieve    focus ------+      (otherwise: abstain)

* **grade** uses the reranker score, not an LLM call (D7: a 1.7B grader on a CPU is slow
  and weak).
* **rewrite** (evidence too weak): one short LLM call turns the question into a search
  query, then retrieval runs again.
* **focus** (optional, **off by default**; the model declined although the gate passed):
  the best sentences of the top-10 candidates replace the three chunks, and the model
  tries once more (P26). Measured unsafe: it fixed one over-refusal but made the model
  answer two trap questions with a fact about a similar model (E12 u01; E12b held-out
  t02, where "LLaMA used 2048 A100 GPUs" became the answer for Llama 2). The reranker
  scores such near-misses as relevant, so a score gate can't catch them.
* **entity check** (E14, on; inside the check step): every name in the question must
  appear in a cited source, or the answer is withdrawn without a retry. With focus on,
  it withdrew the Llama 2 answer above; it can't catch generic names ("Transformer").
* Every corrective step costs one unit of a fixed budget (``max_retries``, default 2), and
  focus runs at most once, so a question makes at most three answer-or-rewrite LLM calls.
  A LangGraph recursion limit backs this up.

The generation prompt and its refusal rule never change between attempts: a retry changes
what the model sees, never what it is allowed to say.
"""

from __future__ import annotations

import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from rag.generation.answer import (
    Answer,
    AnswerSettings,
    Attempt,
    Grade,
    Source,
    generate_and_check,
    grade_evidence,
    make_sources,
)
from rag.generation.focus import focus
from rag.generation.llm import LLMClient, TokenCallback
from rag.generation.prompts import REFUSAL
from rag.retrieval.hybrid import Candidate, Retriever
from rag.tracing import Trace

REWRITE_SYSTEM = (
    "You turn a user's question into a short search query for a library of AI and "
    "machine-learning research papers. Keep every name, acronym and number from the "
    "question. Add the technical terms a paper would use. Reply with the query only, on "
    "one line."
)

# Rewrite v2 (E16). v1's user turn ended in "Query:", and the client appends " /no_think"
# to the last user turn, so the model read "Query: /no_think" and answered slash-style
# ("/search_billion_vectors_quickly"). v2 ends the turn with a sentence and shows the
# expected output with two examples on topics outside the corpus.
REWRITE_SYSTEM_V2 = (
    "You turn a user's question into a search query for a library of AI and "
    "machine-learning research papers. Use the technical terms a paper would use for the "
    "idea in the question: method names, task names and standard jargon. Keep every name, "
    "acronym and number from the question. Reply with the query only: 4 to 12 plain words "
    "on one line, no slashes, no quotes."
)
_REWRITE_EXAMPLES = [
    (
        "How do they make big models smaller without losing much accuracy?",
        "model compression knowledge distillation quantization pruning accuracy",
    ),
    (
        "How do you stop a network from forgetting old tasks when it learns new ones?",
        "catastrophic forgetting continual learning regularization rehearsal",
    ),
]


def _rewrite_turn(question: str) -> str:
    return f"Question: {question}\nWrite the search query."


def rewrite_messages(question: str, version: str = "2") -> list[dict[str, str]]:
    if version == "1":
        return [
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": f"Question: {question}\nQuery:"},
        ]
    msgs = [{"role": "system", "content": REWRITE_SYSTEM_V2}]
    for q, a in _REWRITE_EXAMPLES:
        msgs += [{"role": "user", "content": _rewrite_turn(q)}, {"role": "assistant", "content": a}]
    return [*msgs, {"role": "user", "content": _rewrite_turn(question)}]


def clean_query(text: str, version: str = "2") -> str:
    line = next((ln for ln in text.strip().splitlines() if ln.strip()), "")
    line = line.strip().removeprefix("Query:").strip()
    line = line.strip("\"'` ").strip()
    if version == "1":
        return line
    line = re.sub(r"/?no_think\b", " ", line)  # an echoed Qwen3 switch
    line = re.sub(r"[/_]+", " ", line)  # "/search_billion_vectors" -> "search billion vectors"
    return " ".join(line.split()).strip(" .,;:")


class AgentState(TypedDict, total=False):
    question: str
    query: str  # what retrieval searches for (the question, or a rewrite of it)
    trace: Trace
    hits: list[Candidate]
    grade: Grade
    sources: list[Source]
    attempt: Attempt | None
    retries: int
    focused: bool
    rewrites: list[str]
    focus_skipped: str | None  # why the focus retry did not generate (evidence too weak)
    abstain_reason: str | None


class RAGAgent:
    """Answers questions with the correction loop. Build once, call ``answer`` per question."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        settings: AnswerSettings | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.settings = settings or AnswerSettings()
        self._on_token: TokenCallback | None = None
        self.graph = self._build()

    # ------------------------------------------------------------ nodes

    def _retrieve(self, state: AgentState) -> dict[str, Any]:
        s = self.settings
        with state["trace"].step("retrieve", mode=self.retriever.mode, k=s.retrieve_k) as rec:
            if state["query"] != state["question"]:
                rec["query"] = state["query"]
            hits = self.retriever.search(state["query"], k=s.retrieve_k)
            rec["hits"] = [h.summary() for h in hits]
        return {"hits": hits, "sources": make_sources(hits, s.context_k)}

    def _grade(self, state: AgentState) -> dict[str, Any]:
        grade = grade_evidence(
            state["trace"], state["hits"], self.retriever.uses_reranker, self.settings
        )
        return {"grade": grade}

    def _rewrite(self, state: AgentState) -> dict[str, Any]:
        retries = state["retries"] + 1
        with state["trace"].step("rewrite", retry=retries, budget=self.settings.max_retries) as rec:
            gen = self.llm.generate(
                rewrite_messages(state["question"], self.settings.rewrite_version),
                self.settings.rewrite_max_tokens,
            )
            query = clean_query(gen.text, self.settings.rewrite_version)
            tried = [state["question"], *state["rewrites"]]
            repeated = not query or query.lower() in {t.lower() for t in tried}
            rec.update(model=gen.model, query=query, repeated=repeated)
        if repeated:  # retrieving the same query again cannot change anything
            return {"retries": retries, "abstain_reason": "the query rewrite added nothing new"}
        return {"retries": retries, "query": query, "rewrites": [*state["rewrites"], query]}

    def _generate(self, state: AgentState) -> dict[str, Any]:
        att = generate_and_check(
            state["trace"],
            state["question"],
            state["sources"],
            self.llm,
            self.settings,
            self._on_token,
        )
        return {"attempt": att}

    def _focus(self, state: AgentState) -> dict[str, Any]:
        retries = state["retries"] + 1
        with state["trace"].step("focus", retry=retries, budget=self.settings.max_retries) as rec:
            chunks = focus(
                state["question"],
                state["hits"],
                self.retriever.reranker,
                self.settings.focus_sentences,
            )
            sources = [
                Source(
                    f"S{i}",
                    fc.candidate.chunk_id,
                    fc.candidate.doc_id,
                    fc.candidate.title,
                    fc.candidate.page,
                    round(fc.best_score, 4),
                    " … ".join(fc.sentences),
                )
                for i, fc in enumerate(chunks, start=1)
            ]
            rec["sources"] = [
                {
                    "tag": s.tag,
                    "chunk_id": s.chunk_id,
                    "page": s.page,
                    "score": s.score,
                    "sentences": len(fc.sentences),
                }
                for s, fc in zip(sources, chunks, strict=True)
            ]
            best = max((fc.best_score for fc in chunks), default=None)
            threshold = self.settings.min_rerank_score
            weak = self.settings.focus_gate and (best is None or best < threshold)
            rec.update(
                best_score=None if best is None else round(best, 4),
                threshold=threshold if self.settings.focus_gate else None,
                verdict="too weak, no retry" if weak else "retry",
            )
        if weak:  # E12b: the focused evidence must pass the same gate as the chunks
            shown = "none" if best is None else f"{best:.2f}"
            note = f"the best focused sentence ({shown}) is below the threshold {threshold:.2f}"
            return {"focused": True, "retries": retries, "focus_skipped": note}
        return {"sources": sources, "focused": True, "retries": retries}

    # ------------------------------------------------------------ routing

    def _after_grade(self, state: AgentState) -> str:
        if state["grade"].relevant:
            return "generate"
        if state["retries"] < self.settings.max_retries and not state.get("abstain_reason"):
            return "rewrite"
        return END

    def _after_rewrite(self, state: AgentState) -> str:
        return END if state.get("abstain_reason") else "retrieve"

    def _after_check(self, state: AgentState) -> str:
        att = state["attempt"]
        can_focus = (
            att.refused
            and self.settings.focus_retry
            and not state["focused"]
            and self.retriever.reranker is not None
            and state["retries"] < self.settings.max_retries
        )
        return "focus" if can_focus else END

    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("retrieve", self._retrieve)
        g.add_node("grade", self._grade)
        g.add_node("rewrite", self._rewrite)
        g.add_node("generate", self._generate)
        g.add_node("focus", self._focus)
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges("grade", self._after_grade, ["generate", "rewrite", END])
        g.add_conditional_edges("rewrite", self._after_rewrite, ["retrieve", END])
        g.add_conditional_edges("generate", self._after_check, ["focus", END])
        g.add_conditional_edges(
            "focus", lambda st: END if st.get("focus_skipped") else "generate", ["generate", END]
        )
        return g.compile()

    # ------------------------------------------------------------ entry point

    def answer(
        self, question: str, on_token: TokenCallback | None = None, trace: Trace | None = None
    ) -> Answer:
        question = question.strip()
        self._on_token = on_token
        state: AgentState = {
            "question": question,
            "query": question,
            "trace": trace if trace is not None else Trace(),
            "attempt": None,
            "retries": 0,
            "focused": False,
            "rewrites": [],
            "abstain_reason": None,
            "focus_skipped": None,
        }
        # Each loop visits at most 3 nodes; the limit is a backstop, never the real bound.
        final: AgentState = self.graph.invoke(state, {"recursion_limit": 25})
        return self._to_answer(final)

    def _to_answer(self, st: AgentState) -> Answer:
        common = dict(trace=st["trace"], pipeline="agent", retries=st["retries"])
        att = st.get("attempt")
        if att is None:  # never generated: the evidence stayed too weak
            # The weak evidence is the cause; a stopped loop is only why there was no retry.
            reason = st["grade"].reason
            if st.get("abstain_reason"):
                reason += f"; {st['abstain_reason']}"
            attempts = sum(step.name == "rewrite" for step in st["trace"].steps)
            if attempts:
                reason += f" (after {attempts} query rewrite attempt(s))"
            return Answer(
                st["question"], REFUSAL, True, reason, st["sources"], [], False, None, **common
            )
        if att.refused:
            reason = "the model found no answer in the sources"
            if st.get("focus_skipped"):
                reason += f"; no retry: {st['focus_skipped']}"
            elif st["focused"]:
                reason += ", also after focusing on the best sentences"
            return Answer(
                st["question"],
                REFUSAL,
                True,
                reason,
                st["sources"],
                [],
                False,
                att.model,
                **common,
            )
        if att.rejected:  # E14: the answer named the wrong entity; no retry
            return Answer(
                st["question"],
                REFUSAL,
                True,
                att.rejected,
                st["sources"],
                [],
                False,
                att.model,
                **common,
            )
        return Answer(
            st["question"],
            att.text,
            False,
            None,
            st["sources"],
            att.cited,
            att.inferred,
            att.model,
            **common,
        )

"""Phase 4 agent loop: every route through the graph, the retry budget and the fallback."""

import pytest
from helpers import HashingEmbeddings, ScriptedLLM, word_count

from rag.agent import RAGAgent, clean_query
from rag.generation.answer import AnswerSettings, answer_question
from rag.generation.focus import focus, split_sentences
from rag.generation.llm import FallbackLLM, FirstTokenTimeout, LLMError
from rag.generation.prompts import REFUSAL
from rag.indexing.index import VersionedIndex
from rag.ingestion.pipeline import ingest_file
from rag.retrieval.bm25 import BM25Index, tokenize
from rag.retrieval.hybrid import Candidate, Retriever

QUESTION = "What does the quokka index store in a hierarchical graph?"
ANSWER = "The quokka index stores dense vectors in a hierarchical graph [S1]."


class KeywordReranker:
    """Fake cross-encoder: number of query words in the passage, minus 5."""

    name = "fake-reranker"

    def score(self, query, passages):
        words = set(tokenize(query))
        return [float(len(words & set(tokenize(p))) - 5) for p in passages]


@pytest.fixture
def retriever(tmp_path, fixtures_dir):
    idx = VersionedIndex.open(tmp_path / "idx", HashingEmbeddings(), "hashing-test", 300, 50)
    ingest_file(fixtures_dir / "sample_paper.pdf", idx, token_counter=word_count)
    return Retriever(
        idx, BM25Index.from_store(idx.store), reranker=KeywordReranker(), mode="hybrid_rerank"
    )


def settings(**kw):
    return AnswerSettings(**{"min_rerank_score": -100.0, "focus_retry": True, **kw})


def steps(ans):
    return [s.name for s in ans.trace.steps]


# ---------------------------------------------------------------- routes


def test_first_time_answer_matches_the_single_pass_pipeline(retriever):
    agent_ans = RAGAgent(retriever, ScriptedLLM(ANSWER), settings()).answer(QUESTION)
    single = answer_question(QUESTION, retriever, ScriptedLLM(ANSWER), settings())
    assert agent_ans.text == single.text and agent_ans.cited == single.cited == ["S1"]
    assert agent_ans.pipeline == "agent" and agent_ans.retries == 0
    assert steps(agent_ans) == steps(single) == ["retrieve", "grade", "generate", "check"]


def test_refusal_with_relevant_evidence_gets_one_focus_retry(retriever):
    llm = ScriptedLLM(REFUSAL, ANSWER)
    ans = RAGAgent(retriever, llm, settings(focus_sentences=1)).answer(QUESTION)
    assert not ans.abstained and ans.retries == 1
    assert steps(ans) == ["retrieve", "grade", "generate", "check", "focus", "generate", "check"]
    # The retry shows sentences, not whole chunks, under the same prompt and refusal rule.
    first, second = llm.calls[0], llm.calls[1]
    assert first[0] == second[0]  # same system prompt
    assert "stores dense vectors in a hierarchical graph" in second[1]["content"]
    assert len(second[1]["content"]) < len(first[1]["content"])
    assert ans.cited_sources[0].page == 2


class ChunksStrongSentencesWeak:
    """Scores whole chunks 5.0 and focus sentences (sent as "Title. sentence") 1.0."""

    name = "stub-reranker"

    def score(self, query, passages):
        return [
            1.0 if p.startswith("Quokka Retrieval: A Sample Paper. ") else 5.0 for p in passages
        ]


@pytest.mark.parametrize("gate", [True, False])
def test_focus_retry_needs_focused_evidence_above_the_gate(retriever, gate):
    stub = Retriever(retriever.index, retriever.bm25, reranker=ChunksStrongSentencesWeak())
    llm = ScriptedLLM(REFUSAL, ANSWER)
    agent_settings = AnswerSettings(min_rerank_score=2.0, focus_retry=True, focus_gate=gate)
    ans = RAGAgent(stub, llm, agent_settings).answer(QUESTION)
    focus_step = ans.trace.get("focus").data
    assert focus_step["best_score"] == 1.0
    if gate:  # E12b: sentences below 2.0 are not shown to the model
        assert ans.abstained and len(llm.calls) == 1
        assert "best focused sentence (1.00) is below the threshold 2.00" in ans.abstain_reason
        assert steps(ans)[-1] == "focus" and focus_step["verdict"] == "too weak, no retry"
    else:  # E12 behaviour: retried anyway
        assert not ans.abstained and len(llm.calls) == 2


def test_focus_retry_is_off_by_default(retriever):
    llm = ScriptedLLM(REFUSAL)
    ans = RAGAgent(retriever, llm, AnswerSettings(min_rerank_score=-100.0)).answer(QUESTION)
    assert ans.abstained and ans.retries == 0 and len(llm.calls) == 1
    assert "focus" not in steps(ans)


def test_focus_runs_at_most_once(retriever):
    llm = ScriptedLLM(REFUSAL, REFUSAL)
    ans = RAGAgent(retriever, llm, settings()).answer(QUESTION)
    assert ans.abstained and ans.text == REFUSAL
    assert "also after focusing" in ans.abstain_reason
    assert len(llm.calls) == 2 and ans.retries == 1


def test_weak_evidence_is_rewritten_until_the_budget_runs_out(retriever):
    llm = ScriptedLLM("zebra protocol shards", "marmot benchmark latency")
    ans = RAGAgent(retriever, llm, settings(min_rerank_score=100.0)).answer(QUESTION)
    assert ans.abstained and ans.retries == 2 and ans.model is None
    assert "after 2 query rewrite attempt(s)" in ans.abstain_reason
    assert steps(ans) == ["retrieve", "grade", "rewrite"] * 2 + ["retrieve", "grade"]
    retrieves = [s for s in ans.trace.steps if s.name == "retrieve"]
    assert [r.data.get("query") for r in retrieves] == [
        None,
        "zebra protocol shards",
        "marmot benchmark latency",
    ]


def test_a_rewrite_can_rescue_a_vague_question(retriever):
    vague = "How does that small marsupial thing keep its data?"
    llm = ScriptedLLM("quokka index stores dense vectors hierarchical graph", ANSWER)
    ans = RAGAgent(retriever, llm, settings(min_rerank_score=0.0)).answer(vague)
    assert not ans.abstained and ans.retries == 1
    assert steps(ans) == ["retrieve", "grade", "rewrite", "retrieve", "grade", "generate", "check"]
    # The answer prompt uses the user's question, not the search query.
    assert vague in llm.calls[1][1]["content"]


def test_a_repeated_rewrite_stops_the_loop(retriever):
    llm = ScriptedLLM(f'"{QUESTION}"')
    ans = RAGAgent(retriever, llm, settings(min_rerank_score=100.0)).answer(QUESTION)
    assert ans.abstained and "added nothing new" in ans.abstain_reason
    assert ans.abstain_reason.startswith("best rerank score")  # the cause comes first
    assert "after 1 query rewrite attempt(s)" in ans.abstain_reason
    assert len(llm.calls) == 1


def test_the_budget_is_shared_between_rewrites_and_focus(retriever):
    vague = "How does that small marsupial thing keep its data?"
    rewrite = "quokka index stores dense vectors hierarchical graph"
    # Budget 2: one rewrite, then a refusal still allows the focus retry.
    llm = ScriptedLLM(rewrite, REFUSAL, ANSWER)
    no_gate = settings(min_rerank_score=0.0, focus_gate=False)  # this test is about the budget
    ans = RAGAgent(retriever, llm, no_gate).answer(vague)
    assert not ans.abstained and ans.retries == 2
    # Budget 1: the rewrite uses it up, so the refusal is final.
    llm = ScriptedLLM(rewrite, REFUSAL)
    ans = RAGAgent(retriever, llm, settings(min_rerank_score=0.0, max_retries=1)).answer(vague)
    assert ans.abstained and ans.retries == 1 and "focus" not in steps(ans)


def test_no_focus_without_a_reranker(tmp_path, fixtures_dir):
    idx = VersionedIndex.open(tmp_path / "idx", HashingEmbeddings(), "hashing-test", 300, 50)
    ingest_file(fixtures_dir / "sample_paper.pdf", idx, token_counter=word_count)
    dense = Retriever(idx, BM25Index.from_store(idx.store), mode="dense")
    llm = ScriptedLLM(REFUSAL)
    ans = RAGAgent(dense, llm, AnswerSettings(min_relevance=0.0)).answer(QUESTION)
    assert ans.abstained and ans.retries == 0 and len(llm.calls) == 1


# ---------------------------------------------------------------- pieces


def test_split_sentences_drops_fragments_and_caps_length():
    text = "Table 3. The quokka index stores dense vectors in a graph.\nIt is fast. " + "x" * 900
    out = split_sentences(text)
    assert out[0] == "The quokka index stores dense vectors in a graph."
    assert "It is fast." not in out  # too short to be evidence
    assert all(len(s) <= 400 for s in out)


def _cand(cid, text, page=1):
    return Candidate(cid, "d", "Quokka paper", page, text, {})


def test_focus_keeps_the_best_sentences_grouped_by_chunk():
    cands = [
        _cand("a", "Unrelated opening sentence about shards. The marmot benchmark is hard.", 1),
        _cand("b", "The quokka index stores dense vectors in a graph. The graph is greedy.", 2),
    ]
    chunks = focus("quokka index dense vectors graph", cands, KeywordReranker(), n_sentences=2)
    assert chunks[0].candidate.chunk_id == "b"
    assert chunks[0].sentences[0].startswith("The quokka index")
    assert sum(len(c.sentences) for c in chunks) == 2


def test_focus_scores_a_repeated_sentence_once():
    shared = "The quokka index stores dense vectors in a graph."
    cands = [_cand("a", shared), _cand("b", shared + " Another sentence about marmots here.")]
    chunks = focus("quokka index", cands, KeywordReranker(), n_sentences=5)
    assert [len(c.sentences) for c in chunks if c.candidate.chunk_id == "a"] == [1]
    assert all(shared not in c.sentences for c in chunks if c.candidate.chunk_id == "b")


@pytest.mark.parametrize(
    ("raw", "query"),
    [
        ("Query: RETRO retrieval database size\nextra", "RETRO retrieval database size"),
        ('"SciFact claims count"', "SciFact claims count"),
        ("\n\n  quokka index  \n", "quokka index"),
    ],
)
def test_clean_query(raw, query):
    assert clean_query(raw) == query


@pytest.mark.parametrize(
    ("raw", "query"),
    [  # real rewrite-v1 outputs from the live run (E16)
        (
            "/trick/model/think/out/loud/before/answering",
            "trick model think out loud before answering",
        ),
        ("/search_billion_vectors_quickly", "search billion vectors quickly"),
        ("vector search /no_think", "vector search"),
    ],
)
def test_clean_query_v2_repairs_slash_style_output(raw, query):
    assert clean_query(raw) == query
    assert clean_query(raw, version="1") == raw.strip()  # v1 kept for reproduction


def test_rewrite_prompt_v2_does_not_end_where_the_no_think_switch_goes():
    from rag.agent.graph import rewrite_messages

    v2 = rewrite_messages("How do you search a billion vectors quickly?")
    assert v2[-1]["role"] == "user" and v2[-1]["content"].endswith("Write the search query.")
    assert [m["role"] for m in v2] == ["system", "user", "assistant", "user", "assistant", "user"]
    v1 = rewrite_messages("q", version="1")
    assert v1[-1]["content"].endswith("Query:")


# ---------------------------------------------------------------- fallback gateway


class FailingLLM:
    name = "big-model"

    def __init__(self, exc):
        self.exc = exc

    def generate(self, messages, max_tokens, on_token=None):
        raise self.exc


@pytest.mark.parametrize(
    "exc", [FirstTokenTimeout("no first token within 90 s"), LLMError("model file not found")]
)
def test_fallback_answers_when_the_primary_fails_or_is_too_slow(retriever, exc):
    llm = FallbackLLM(FailingLLM(exc), ScriptedLLM(ANSWER, name="small-model"))
    ans = RAGAgent(retriever, llm, settings()).answer(QUESTION)
    assert not ans.abstained and ans.model == "small-model"
    gen = ans.trace.get("generate").data
    assert gen["fallback_from"] == "big-model" and gen["fallback_reason"] == str(exc)


def test_fallback_is_not_used_when_the_primary_works():
    small = ScriptedLLM(name="small-model")  # no replies: any call would fail the test
    gen = FallbackLLM(ScriptedLLM("fine", name="big-model"), small).generate([], 10)
    assert gen.model == "big-model" and gen.fallback_from is None and small.calls == []

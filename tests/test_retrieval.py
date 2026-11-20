import pytest
from helpers import FakeLLM, HashingEmbeddings, word_count

from rag.generation.answer import AnswerSettings, answer_question, relevance_signal
from rag.indexing.index import VersionedIndex
from rag.ingestion.pipeline import ingest_file
from rag.retrieval.bm25 import BM25Index, StoredChunk, tokenize
from rag.retrieval.hybrid import Retriever, reciprocal_rank_fusion


class KeywordReranker:
    """Fake cross-encoder: score = number of query words found in the passage, minus 5."""

    name = "fake-reranker"

    def score(self, query, passages):
        words = set(tokenize(query))
        return [float(len(words & set(tokenize(p))) - 5) for p in passages]


def _chunks(*texts):
    return [
        StoredChunk(f"d:{i:05d}", t, {"doc_id": "d", "title": "T", "page": i + 1})
        for i, t in enumerate(texts)
    ]


# ---------------------------------------------------------------- BM25 and RRF


def test_tokenize_keeps_exact_technical_terms():
    assert tokenize("NF4 uses 1.4T tokens, see ms-marco and GPT-4.") == [
        "nf4",
        "uses",
        "1.4t",
        "tokens",
        "see",
        "ms-marco",
        "gpt-4",
    ]


def test_bm25_finds_the_rare_exact_term():
    bm25 = BM25Index(
        _chunks(
            "dense vectors are compared by cosine similarity",
            "the nf4 data type quantizes weights",
            "vectors and weights and more vectors",
        )
    )
    hits = bm25.search("what is nf4", k=3)
    assert hits[0][0] == "d:00001"
    assert all(score > 0 for _, score in hits)


def test_bm25_returns_nothing_for_unknown_or_empty_queries():
    bm25 = BM25Index(_chunks("alpha beta", "gamma delta"))
    assert bm25.search("zeta", k=5) == []
    assert bm25.search("the of and", k=5) == []  # only stopwords


def test_rrf_scores_and_order():
    fused = dict(reciprocal_rank_fusion({"dense": ["a", "b", "c"], "bm25": ["c", "a"]}, k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert fused["b"] == pytest.approx(1 / 62)
    order = [d for d, _ in reciprocal_rank_fusion({"dense": ["a", "b", "c"], "bm25": ["c", "a"]})]
    assert order == ["a", "c", "b"]  # in both lists beats a single high rank


def test_rrf_weights_and_tie_break():
    fused = reciprocal_rank_fusion({"x": ["a"], "y": ["b"]}, weights={"y": 2.0})
    assert [d for d, _ in fused] == ["b", "a"]
    tie = reciprocal_rank_fusion({"x": ["a"], "y": ["b"]})
    assert [d for d, _ in tie] == ["a", "b"]  # equal scores: first appearance wins


# ---------------------------------------------------------------- the retriever


@pytest.fixture
def index(tmp_path, fixtures_dir):
    idx = VersionedIndex.open(tmp_path / "idx", HashingEmbeddings(), "hashing-test", 300, 50)
    ingest_file(fixtures_dir / "sample_paper.pdf", idx, token_counter=word_count)
    return idx


def _retriever(index, mode, reranker=None):
    return Retriever(index, BM25Index.from_store(index.store), reranker=reranker, mode=mode)


def test_modes_record_the_stages_they_ran(index):
    q = "marmot benchmark accuracy"
    dense = _retriever(index, "dense").search(q, k=3)
    assert dense[0].dense_rank == 1 and dense[0].bm25_rank is None and dense[0].fused_rank is None

    bm25 = _retriever(index, "bm25").search(q, k=3)
    assert bm25[0].bm25_rank == 1 and bm25[0].dense_rank is None

    hybrid = _retriever(index, "hybrid").search(q, k=3)
    assert hybrid[0].fused_rank == 1 and hybrid[0].rrf_score is not None
    assert hybrid[0].page == 3

    reranked = _retriever(index, "hybrid_rerank", KeywordReranker()).search(q, k=3)
    assert reranked[0].rerank_score is not None
    scores = [c.rerank_score for c in reranked]
    assert scores == sorted(scores, reverse=True)
    assert [c.rank for c in reranked] == [1, 2, 3]


def test_reranker_can_override_the_fused_order(index):
    class PreferPageTwo:
        name = "prefer-page-2"

        def score(self, query, passages):
            return [10.0 if "quokka index" in p else 0.0 for p in passages]

    hits = _retriever(index, "hybrid_rerank", PreferPageTwo()).search("marmot benchmark", k=3)
    assert hits[0].page == 2


def test_rerank_mode_without_a_reranker_is_rejected(index):
    with pytest.raises(ValueError, match="needs a reranker"):
        _retriever(index, "hybrid_rerank")
    with pytest.raises(ValueError, match="unknown retrieval mode"):
        _retriever(index, "magic")


# ---------------------------------------------------------------- grading with reranker scores


def test_relevance_signal_prefers_the_reranker(index):
    hits = _retriever(index, "hybrid_rerank", KeywordReranker()).search("marmot benchmark", k=3)
    signal, best = relevance_signal(hits, reranked=True)
    assert signal == "rerank" and best == max(h.rerank_score for h in hits)
    signal, _ = relevance_signal(_retriever(index, "bm25").search("marmot", k=3), reranked=False)
    assert signal == "none"  # BM25 scores are not comparable across queries


def test_low_rerank_score_abstains_without_the_llm(index):
    llm = FakeLLM("unused")
    retriever = _retriever(index, "hybrid_rerank", KeywordReranker())
    ans = answer_question(
        "completely unrelated penguins", retriever, llm, AnswerSettings(min_rerank_score=0.0)
    )
    assert ans.abstained and "rerank" in ans.abstain_reason and llm.calls == []
    assert ans.trace.get("grade").data["signal"] == "rerank"


# ---------------------------------------------------------------- reference filter

BIBLIOGRAPHY = (
    "[12] P. Lewis, E. Perez, A. Piktus, and D. Kiela. Retrieval-augmented generation for "
    "knowledge-intensive NLP tasks. In Advances in NeurIPS, 2020. [13] J. Smith and K. Lee. "
    "Dense retrieval at scale. arXiv preprint arXiv:2101.00001, 2021. [14] M. Chen. Sparse "
    "retrieval revisited. In Proceedings of SIGIR, 2019. [15] T. Brown. Language models. 2020."
)
CITING_PROSE = (
    "Retrieval-augmented generation (Lewis et al., 2020) combines a retriever with a "
    "generator. Later work (Izacard and Grave, 2021; Borgeaud et al., 2022) scaled the "
    "retrieval database, and Asai et al. (2023) added self-reflection to decide when to "
    "retrieve."
)


def test_reference_detector_separates_bibliography_from_citing_prose():
    from rag.retrieval.filters import looks_like_references

    assert looks_like_references(BIBLIOGRAPHY)
    assert not looks_like_references(CITING_PROSE)


def test_skip_references_drops_bibliography_chunks():
    # BM25's IDF is zero for a term in half the chunks, so add unrelated filler chunks.
    filler = [f"unrelated filler text number {i}" for i in range(4)]
    bm25 = BM25Index(_chunks(BIBLIOGRAPHY, CITING_PROSE, *filler))
    query = "retrieval-augmented generation retrieval"
    plain = Retriever(None, bm25, mode="bm25").search(query, k=5)
    filtered = Retriever(None, bm25, mode="bm25", skip_references=True).search(query, k=5)
    assert {c.chunk_id for c in plain} == {"d:00000", "d:00001"}
    assert [c.chunk_id for c in filtered] == ["d:00001"]
    assert filtered[0].rank == 1 and filtered[0].bm25_rank == 1

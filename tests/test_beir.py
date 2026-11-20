import json
import math

import pytest

from rag.evaluation import beir


def test_ndcg_perfect_and_hand_computed():
    rel = {"a": 1, "b": 1}
    assert beir.ndcg_at_k(["a", "b", "c"], rel, 10) == pytest.approx(1.0)
    # relevant docs at ranks 2 and 3
    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert beir.ndcg_at_k(["x", "a", "b"], rel, 10) == pytest.approx(dcg / idcg)
    assert beir.ndcg_at_k(["x", "y"], rel, 10) == 0.0


def test_ndcg_respects_cutoff_and_graded_relevance():
    assert beir.ndcg_at_k(["x", "a"], {"a": 1}, k=1) == 0.0
    graded = {"a": 2, "b": 1}
    assert beir.ndcg_at_k(["b", "a"], graded, 10) < beir.ndcg_at_k(["a", "b"], graded, 10)


def test_recall_at_k():
    assert beir.recall_at_k(["a", "x", "b"], {"a": 1, "b": 1, "c": 1}, 100) == pytest.approx(2 / 3)
    assert beir.recall_at_k(["a", "x", "b"], {"a": 1, "b": 1}, 1) == pytest.approx(1 / 2)


def test_load_reads_beir_layout_and_keeps_judged_queries_only(tmp_path):
    folder = tmp_path / "toy"
    (folder / "qrels").mkdir(parents=True)
    corpus = [
        {"_id": "d1", "title": "T1", "text": "alpha"},
        {"_id": "d2", "title": "", "text": "beta"},
    ]
    (folder / "corpus.jsonl").write_text("\n".join(json.dumps(c) for c in corpus))
    queries = [{"_id": "q1", "text": "alpha?"}, {"_id": "q2", "text": "unjudged"}]
    (folder / "queries.jsonl").write_text("\n".join(json.dumps(q) for q in queries))
    (folder / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td2\t0\n")
    ds = beir.load(folder)
    assert list(ds.queries) == ["q1"]
    assert ds.qrels == {"q1": {"d1": 1}}  # zero-relevance rows are dropped
    assert beir.doc_text(ds.corpus["d1"]) == "T1. alpha"
    assert beir.doc_text(ds.corpus["d2"]) == "beta"

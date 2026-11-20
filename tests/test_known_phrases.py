import random

from helpers import HashingEmbeddings, word_count

from rag.evaluation.known_phrases import pick_phrase, run_known_phrase_check, summarize
from rag.indexing.index import VersionedIndex
from rag.ingestion.pipeline import ingest_file


def test_pick_phrase_prefers_prose_over_numbers():
    text = (
        "12.3 45.6 78.9 10.1 11.2 13.4 15.6 17.8 19.0. "
        "The quokka index stores dense vectors in a hierarchical graph for search."
    )
    assert pick_phrase(text, random.Random(0)).startswith("The quokka index")


def test_pick_phrase_returns_empty_when_nothing_fits():
    assert pick_phrase("Too short.", random.Random(0)) == ""


def test_known_phrase_check_on_fixture(tmp_path, fixtures_dir):
    index = VersionedIndex.open(tmp_path / "idx", HashingEmbeddings(), "hashing-test", 300, 50)
    ingest_file(fixtures_dir / "sample_paper.pdf", index, token_counter=word_count)
    results = run_known_phrase_check(index, samples=10, k=3)
    assert results  # at least one prose sentence was found
    summary = summarize(results, 3)
    assert summary["page_at_1"] == 1.0
    assert summary["mrr"] == 1.0

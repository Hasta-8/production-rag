import shutil

import pytest
from helpers import HashingEmbeddings, word_count

from rag.indexing.embeddings import CachedEmbeddings
from rag.indexing.index import IndexMismatchError, VersionedIndex, compute_index_version
from rag.ingestion.pipeline import ingest_file
from rag.retrieval.dense import dense_search

MODEL = "hashing-test"


def _open(tmp_path, embeddings=None, model=MODEL, size=300, overlap=50):
    return VersionedIndex.open(
        tmp_path / "indexes", embeddings or HashingEmbeddings(), model, size, overlap
    )


def _ingest(path, index):
    return ingest_file(path, index, token_counter=word_count)


# ---------------------------------------------------------------- embedding cache


def test_embedding_cache_avoids_repeat_calls(tmp_path):
    inner = HashingEmbeddings()
    cached = CachedEmbeddings(inner, MODEL, tmp_path / "cache.sqlite")
    first = cached.embed_documents(["alpha beta", "gamma"])
    assert inner.calls == 2
    again = cached.embed_documents(["gamma", "alpha beta", "delta"])
    assert inner.calls == 3  # only "delta" was new
    assert again[0] == pytest.approx(first[1], abs=1e-6)
    assert (cached.hits, cached.misses) == (2, 3)


class PrefixedQueries(HashingEmbeddings):
    def embed_query(self, text):
        return super().embed_query("query instruction " + text)


def test_embedding_cache_keeps_documents_and_queries_apart(tmp_path):
    inner = PrefixedQueries()
    cached = CachedEmbeddings(inner, MODEL, tmp_path / "cache.sqlite")
    doc_vec = cached.embed_documents(["same text"])[0]
    query_vec = cached.embed_query("same text")
    assert inner.calls == 2  # the query was not served from the document entry
    assert query_vec != pytest.approx(doc_vec, abs=1e-6)


def test_embedding_cache_persists_and_is_keyed_by_model(tmp_path):
    path = tmp_path / "cache.sqlite"
    CachedEmbeddings(HashingEmbeddings(), MODEL, path).embed_query("hello")
    inner = HashingEmbeddings()
    CachedEmbeddings(inner, MODEL, path).embed_query("hello")
    assert inner.calls == 0
    CachedEmbeddings(inner, "other-model", path).embed_query("hello")
    assert inner.calls == 1


# ---------------------------------------------------------------- index versions


def test_index_version_changes_with_settings():
    base = compute_index_version("m", 1000, 150)
    assert base == compute_index_version("m", 1000, 150)
    assert base != compute_index_version("m2", 1000, 150)
    assert base != compute_index_version("m", 800, 150)


def test_reopen_with_different_model_in_same_folder_fails(tmp_path):
    index = _open(tmp_path)
    index.info.embedding_model = "something-else"
    index.save_info()
    with pytest.raises(IndexMismatchError):
        _open(tmp_path)


# ---------------------------------------------------------------- pipeline


def test_ingest_then_search_finds_the_right_page(tmp_path, fixtures_dir):
    """Phase 1 exit gate, in miniature: a known phrase returns the right page."""
    index = _open(tmp_path)
    result = _ingest(fixtures_dir / "sample_paper.pdf", index)
    assert result.status == "indexed" and result.pages == 3

    hits = dense_search(index, "quokka index stores dense vectors hierarchical graph", k=3)
    assert hits[0].page == 2
    assert hits[0].title == "Quokka Retrieval: A Sample Paper"
    assert hits[0].score > hits[-1].score

    hits = dense_search(index, "marmot benchmark accuracy latency", k=1)
    assert hits[0].page == 3


def test_ingested_text_is_normalised(tmp_path, fixtures_dir):
    index = _open(tmp_path)
    _ingest(fixtures_dir / "sample_paper.pdf", index)
    text = dense_search(index, "zebra protocol", k=1)[0].text
    assert "efficient retrieval" in text  # ligature expanded, line-break hyphen joined


def test_reingesting_the_same_content_is_a_duplicate(tmp_path, fixtures_dir):
    index = _open(tmp_path)
    first = _ingest(fixtures_dir / "sample_paper.pdf", index)
    renamed = tmp_path / "copy_with_other_name.pdf"
    shutil.copy(fixtures_dir / "sample_paper.pdf", renamed)
    second = _ingest(renamed, index)
    assert second.status == "duplicate" and second.doc_id == first.doc_id
    assert index.count() == first.chunks


def test_changed_file_with_same_name_replaces_old_chunks(tmp_path, fixtures_dir):
    index = _open(tmp_path)
    work = tmp_path / "paper.pdf"
    shutil.copy(fixtures_dir / "sample_paper.pdf", work)
    old = _ingest(work, index)
    shutil.copy(fixtures_dir / "sample_paper_revised.pdf", work)
    new = _ingest(work, index)
    assert new.status == "replaced" and new.doc_id != old.doc_id
    assert list(index.info.documents) == [new.doc_id]
    assert index.count() == new.chunks
    assert "98.1" in dense_search(index, "marmot benchmark accuracy", k=1)[0].text


class FailsAfterFirstBatch(HashingEmbeddings):
    def embed_documents(self, texts):
        if self.calls:
            raise RuntimeError("provider down")
        return super().embed_documents(texts)


def test_failed_embedding_rolls_back_partial_writes(tmp_path, fixtures_dir, monkeypatch):
    monkeypatch.setattr("rag.ingestion.pipeline.EMBED_BATCH", 1)  # force several batches
    index = _open(tmp_path, embeddings=FailsAfterFirstBatch())
    with pytest.raises(RuntimeError, match="provider down"):
        _ingest(fixtures_dir / "sample_paper.pdf", index)
    assert index.count() == 0
    assert index.info.documents == {}


def test_unreadable_file_is_reported_not_raised(tmp_path, fixtures_dir):
    index = _open(tmp_path)
    result = _ingest(fixtures_dir / "image_only.pdf", index)
    assert result.status == "error" and "text layer" in result.error
    assert index.count() == 0


def test_index_persists_across_reopen(tmp_path, fixtures_dir):
    index = _open(tmp_path)
    result = _ingest(fixtures_dir / "sample_paper.pdf", index)
    reopened = _open(tmp_path)
    assert result.doc_id in reopened.info.documents
    assert reopened.count() == result.chunks
    assert dense_search(reopened, "marmot benchmark", k=1)[0].page == 3


def test_chunk_metadata_supports_citations(tmp_path, fixtures_dir):
    index = _open(tmp_path)
    _ingest(fixtures_dir / "sample_paper.pdf", index)
    got = index.store.get(include=["metadatas"])
    meta = got["metadatas"][0]
    for key in ("doc_id", "chunk_id", "source", "title", "page", "content_hash", "index_version"):
        assert key in meta
    assert meta["index_version"] == index.info.version

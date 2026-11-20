import pytest
from helpers import word_count

from rag.ingestion.chunkers import chunk_id_for, chunk_pages
from rag.ingestion.loaders import IngestionError, load_document
from rag.ingestion.normalize import make_doc_id, normalize_text

# ---------------------------------------------------------------- normalisation


def test_normalize_joins_hyphenated_line_breaks():
    assert normalize_text("retrie-\nval works") == "retrieval works"


def test_normalize_expands_pdf_ligatures():
    assert normalize_text("eﬃcient ﬁle") == "efficient file"


def test_normalize_collapses_spaces_but_keeps_paragraphs():
    text = "a   b\t c\n\n\n\nnext   paragraph"
    assert normalize_text(text) == "a b c\n\nnext paragraph"


def test_doc_id_ignores_whitespace_differences():
    assert make_doc_id(["hello  world", "page two"]) == make_doc_id(["hello world\n", "page  two"])


def test_doc_id_changes_with_content():
    assert make_doc_id(["hello world"]) != make_doc_id(["hello there"])
    assert len(make_doc_id(["x"])) == 16


# ---------------------------------------------------------------- loading


def test_load_pdf_keeps_page_numbers_and_title(fixtures_dir):
    doc = load_document(fixtures_dir / "sample_paper.pdf")
    assert [p.page for p in doc.pages] == [1, 2, 3]
    assert "quokka index" in doc.pages[1].text
    assert doc.title == "Quokka Retrieval: A Sample Paper"


def test_image_only_pdf_is_rejected_with_a_reason(fixtures_dir):
    with pytest.raises(IngestionError, match="no usable text layer"):
        load_document(fixtures_dir / "image_only.pdf")


def test_unsupported_type_is_rejected(tmp_path):
    path = tmp_path / "slides.pptx"
    path.write_bytes(b"x")
    with pytest.raises(IngestionError, match="unsupported"):
        load_document(path)


def test_text_file_loads_as_one_page(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Title\n\nSome text.")
    doc = load_document(path)
    assert len(doc.pages) == 1 and doc.pages[0].page == 1


# ---------------------------------------------------------------- chunking


def _pages(n_words: int, pages: int = 2):
    return [(p, " ".join(f"w{p}x{i}." for i in range(n_words))) for p in range(1, pages + 1)]


def test_chunks_never_span_pages_and_respect_size():
    chunks = chunk_pages("d" * 16, _pages(400), 300, 50, word_count)
    assert len(chunks) > 4
    for c in chunks:
        assert len(c.text) <= 300
        assert c.text.split()[0].startswith(f"w{c.page}x")
        assert all(w.startswith(f"w{c.page}x") for w in c.text.split())


def test_chunk_ids_are_sequential_and_stable():
    a = chunk_pages("abcd" * 4, _pages(200), 300, 50, word_count)
    b = chunk_pages("abcd" * 4, _pages(200), 300, 50, word_count)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert [c.chunk_index for c in a] == list(range(len(a)))
    assert a[3].chunk_id == chunk_id_for("abcd" * 4, 3)


def test_consecutive_chunks_overlap():
    chunks = chunk_pages("d" * 16, _pages(400, pages=1), 300, 80, word_count)
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-1] in second


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_pages("d" * 16, _pages(10), 100, 100, word_count)

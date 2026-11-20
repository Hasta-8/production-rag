from pathlib import Path

import pytest
from helpers import HashingEmbeddings, word_count

from rag.evaluation.golden import (
    GoldenQuestion,
    RetrievalRow,
    keyword_recall,
    resolve_evidence,
    threshold_table,
)
from rag.generation.answer import AnswerSettings, answer_question, trim_incomplete_sentence
from rag.generation.citations import check_citations, find_tags, is_refusal, remove_tags
from rag.generation.llm import Generation, LlamaCppClient, LLMError, _ThinkFilter, strip_reasoning
from rag.generation.prompts import REFUSAL, PromptSource, build_messages
from rag.indexing.index import VersionedIndex
from rag.ingestion.pipeline import ingest_file


class FakeLLM:
    """Returns canned text and records what it was asked."""

    def __init__(self, text: str, completion_tokens: int = 20) -> None:
        self.name = "fake-llm"
        self.text = text
        self.completion_tokens = completion_tokens
        self.calls: list = []

    def generate(self, messages, max_tokens, on_token=None):
        self.calls.append(messages)
        if on_token:
            on_token(self.text)
        return Generation(self.text, self.name, 100, self.completion_tokens, 5.0, 1.0)


SOURCES = [
    PromptSource("S1", "Quokka paper", 2, "The quokka index stores dense vectors in a graph."),
    PromptSource("S2", "Quokka paper", 3, "The marmot benchmark accuracy is 97.3 percent."),
]

# ---------------------------------------------------------------- prompts and citations


def test_prompt_numbers_sources_with_title_and_page():
    messages = build_messages("What does the index store?", SOURCES)
    assert messages[0]["role"] == "system" and REFUSAL in messages[0]["content"]
    user = messages[1]["content"]
    assert "[S1] (Quokka paper, p. 2) The quokka index" in user
    assert "[S2] (Quokka paper, p. 3)" in user
    assert user.rstrip().endswith("Answer:")


@pytest.mark.parametrize(
    ("text", "tags"),
    [
        ("A [S1]. B [S2].", ["S1", "S2"]),
        ("A [S2][S1] and [S2].", ["S2", "S1"]),
        ("A [S1, S3].", ["S1", "S3"]),
        ("A [s1; S4].", ["S1", "S4"]),
        ("A [1] and [see below].", []),
    ],
)
def test_find_tags(text, tags):
    assert find_tags(text) == tags


def test_remove_tags_drops_invalid_and_empty_brackets():
    assert remove_tags("Fact [S1][S9]. Other [S9].", ["S9"]) == "Fact [S1]. Other."
    assert remove_tags("Mixed [S1, S9].", ["S9"]) == "Mixed [S1]."


def test_check_citations_removes_invented_tags():
    check = check_citations("The index stores vectors [S1][S7].", SOURCES)
    assert check.cited == ["S1"] and check.invalid == ["S7"]
    assert "[S7]" not in check.text and not check.inferred


def test_check_citations_infers_by_overlap_and_says_so():
    answer = (
        "The quokka index stores dense vectors in a graph. The benchmark accuracy is 97.3 percent."
    )
    check = check_citations(answer, SOURCES)
    assert check.inferred
    assert check.cited == ["S1", "S2"]
    assert check.text.endswith("[S2].")
    assert check.uncited_sentences == 0


def test_check_citations_leaves_unsupported_sentences_uncited():
    check = check_citations("Penguins live in Antarctica.", SOURCES)
    assert check.cited == [] and not check.inferred and check.uncited_sentences == 1


def test_refusal_detection():
    assert is_refusal(REFUSAL)
    assert is_refusal("Sorry, I don’t have enough information in the indexed documents.")
    assert not is_refusal("The index stores vectors [S1].")


@pytest.mark.parametrize(
    ("text", "expected", "cut"),
    [
        ("One fact [S1]. Two facts [S2]. And then the mod", "One fact [S1]. Two facts [S2].", True),
        ("Complete answer [S1].", "Complete answer [S1].", False),
        ("Ends with a tag [S1]", "Ends with a tag [S1]", False),
        ("No sentence end at all", "No sentence end at all", False),
    ],
)
def test_trim_incomplete_sentence(text, expected, cut):
    assert trim_incomplete_sentence(text) == (expected, cut)


# ---------------------------------------------------------------- LLM helpers


def test_strip_reasoning_removes_think_block():
    assert strip_reasoning("<think>\n\n</think>\n\nAnswer [S1].") == "Answer [S1]."


def test_think_filter_hides_streamed_think_block():
    out = []
    f = _ThinkFilter(out.append)
    for tok in ["<th", "ink>", "\n\n</think>", "\n\nThe ", "answer"]:
        f(tok)
    assert "".join(out) == "The answer"


def test_think_filter_passes_text_without_think_block():
    out = []
    f = _ThinkFilter(out.append)
    for tok in ["The", " answer"]:
        f(tok)
    assert "".join(out) == "The answer"


def test_missing_model_file_raises_llm_error(tmp_path):
    client = LlamaCppClient(tmp_path / "missing.gguf")
    with pytest.raises(LLMError, match="not found"):
        client.generate([{"role": "user", "content": "hi"}], max_tokens=5)


# ---------------------------------------------------------------- pipeline


@pytest.fixture
def index(tmp_path, fixtures_dir):
    idx = VersionedIndex.open(tmp_path / "idx", HashingEmbeddings(), "hashing-test", 300, 50)
    ingest_file(fixtures_dir / "sample_paper.pdf", idx, token_counter=word_count)
    return idx


QUESTION = "What does the quokka index store in a hierarchical graph?"


def test_answer_with_valid_citation(index):
    llm = FakeLLM("The quokka index stores dense vectors in a hierarchical graph [S1].")
    ans = answer_question(QUESTION, index, llm, AnswerSettings(min_relevance=0.0))
    assert not ans.abstained
    assert ans.cited == ["S1"] and not ans.citations_inferred
    assert ans.cited_sources[0].page == 2
    assert [s.name for s in ans.trace.steps] == ["retrieve", "grade", "generate", "citations"]
    assert "[S1]" in llm.calls[0][1]["content"]  # sources were in the prompt


def test_low_relevance_abstains_without_calling_the_llm(index):
    llm = FakeLLM("should not be used")
    ans = answer_question(QUESTION, index, llm, AnswerSettings(min_relevance=0.999))
    assert ans.abstained and "below the threshold" in ans.abstain_reason
    assert ans.text == REFUSAL and llm.calls == []
    assert [s.name for s in ans.trace.steps] == ["retrieve", "grade"]


def test_model_refusal_is_an_abstention(index):
    llm = FakeLLM(REFUSAL)
    ans = answer_question(QUESTION, index, llm, AnswerSettings(min_relevance=0.0))
    assert ans.abstained and ans.cited == [] and ans.model == "fake-llm"


def test_answer_hitting_token_cap_is_trimmed(index):
    llm = FakeLLM("It stores vectors [S1]. It also keeps sixteen neigh", completion_tokens=150)
    ans = answer_question(QUESTION, index, llm, AnswerSettings(min_relevance=0.0, max_tokens=150))
    assert ans.text == "It stores vectors [S1]."
    assert ans.trace.get("generate").data["truncated"] is True


def test_context_k_limits_sources(index):
    llm = FakeLLM("Answer [S1].")
    ans = answer_question(QUESTION, index, llm, AnswerSettings(min_relevance=0.0, context_k=2))
    assert [s.tag for s in ans.sources] == ["S1", "S2"]


# ---------------------------------------------------------------- golden-set helpers


def test_keyword_recall_with_alternatives():
    assert keyword_recall("Uses NF4 and GQA.", ["normalfloat|nf4", "gqa"]) == 1.0
    assert keyword_recall("Uses NF4.", ["normalfloat|nf4", "gqa"]) == 0.5


def test_resolve_evidence_finds_pages_and_rejects_missing_phrases():
    pages = {("1111.1111", 1): "intro text", ("1111.1111", 4): "the quokka index stores"}
    q = GoldenQuestion(
        id="q", question="?", arxiv_id="1111.1111", evidence_phrases=["quokka index"]
    )
    assert resolve_evidence([q], pages) == {"q": {4}}
    bad = q.model_copy(update={"evidence_phrases": ["not there"]})
    with pytest.raises(ValueError, match="not found"):
        resolve_evidence([bad], pages)


def test_threshold_table_counts_kept_and_rejected():
    rows = [
        RetrievalRow("a", True, 0.8),
        RetrievalRow("b", True, 0.6),
        RetrievalRow("u", False, 0.55),
    ]
    table = {r["threshold"]: r for r in threshold_table(rows)}
    assert table[0.6]["answerable_kept"] == 1.0 and table[0.6]["unanswerable_rejected"] == 1.0
    assert table[0.7]["answerable_kept"] == 0.5


def test_golden_file_is_valid():
    from rag.evaluation.golden import load_golden

    qs = load_golden(Path(__file__).parents[1] / "eval" / "golden.yaml")
    assert len({q.id for q in qs}) == len(qs)
    for q in qs:
        if q.answerable:
            assert q.arxiv_id and q.evidence_phrases and q.keywords
            assert all(p == p.lower() for p in q.evidence_phrases)
    assert sum(not q.answerable for q in qs) >= 5

"""E14 entity check: names in the question must appear in a cited source."""

import pytest
from helpers import ScriptedLLM
from test_agent import retriever, settings, steps  # noqa: F401  (fixture reuse)

from rag.agent import RAGAgent
from rag.generation.answer import answer_question
from rag.generation.entities import check_entities, entity_pattern, question_entities
from rag.generation.prompts import REFUSAL


@pytest.mark.parametrize(
    ("question", "names"),
    [
        ("How many GPUs were used to train Llama 2?", ["GPUs", "Llama 2"]),
        ("How many parameters does Claude 2.1 have?", ["Claude 2.1"]),
        ("What is the context window of Mixtral 8x22B?", ["Mixtral 8x22B"]),
        (
            "What is the embedding dimension of OpenAI's text-embedding-3-large?",
            ["OpenAI", "text-embedding-3-large"],
        ),
        ("What batch size was DINOv2 trained with?", ["DINOv2"]),
        ("Who won the 2022 FIFA World Cup?", ["FIFA World Cup"]),
        ("What sliding window size does Mistral 7B use?", ["Mistral 7B"]),
        ("What is the default pooling strategy of Sentence-BERT?", ["Sentence-BERT"]),
        ("What does the quokka index store in a hierarchical graph?", []),
    ],
)
def test_question_entities(question, names):
    assert question_entities(question) == names


@pytest.mark.parametrize(
    ("name", "text", "found"),
    [
        ("Llama 2", "compared with Llama 2 70B", True),
        ("Llama 2", "Llama-2 and llama2", True),
        ("Llama 2", "LLaMA, 2048 A100 GPUs", False),  # the t02 near-miss
        ("Llama 2", "LLaMA 2048 A100 GPUs", False),
        ("Llama 2", "LLaMA: Open and Efficient Foundation Language Models", False),
        ("GPUs", "on 2048 A100 GPU with 80GB", True),  # plural tolerance
        ("Sentence-BERT", "SentenceBERT embeddings", True),
        ("DINOv2", "DINO", False),
        ("Mixtral 8x22B", "Mixtral 8x7B", False),
        ("BERT", "RoBERTa", False),  # whole words only
    ],
)
def test_entity_pattern(name, text, found):
    assert bool(entity_pattern(name).search(text)) is found


def test_every_name_needs_one_cited_source():
    v = check_entities(
        "How many GPUs were used to train Llama 2?",
        [("S1", "LLaMA: Open and Efficient", "2048 A100 GPU"), ("S2", "Mixtral", "Llama 2 70B")],
    )
    assert v.supported == {"GPUs": ["S1"], "Llama 2": ["S2"]} and v.passed
    v = check_entities("How many GPUs were used to train Llama 2?", [("S1", "LLaMA", "2048 GPUs")])
    assert v.unsupported == ["Llama 2"] and v.reason == 'the cited sources never name "Llama 2"'


# ---------------------------------------------------------------- in both pipelines

NAMED = "What does the Quokka index store in a hierarchical graph?"
WRONG = "What does the Wombat index store in a hierarchical graph?"
ANSWER = "It stores dense vectors in a hierarchical graph [S1]."


def test_answer_kept_when_the_cited_source_names_the_entity(retriever):  # noqa: F811
    ans = RAGAgent(retriever, ScriptedLLM(ANSWER), settings()).answer(NAMED)
    assert not ans.abstained and steps(ans)[-1] == "entities"
    assert ans.trace.steps[-1].data["verdict"] == "passed"


@pytest.mark.parametrize("pipeline", ["agent", "single"])
def test_answer_withdrawn_when_no_cited_source_names_the_entity(retriever, pipeline):  # noqa: F811
    llm = ScriptedLLM(ANSWER)
    if pipeline == "agent":
        ans = RAGAgent(retriever, llm, settings()).answer(WRONG)
    else:
        ans = answer_question(WRONG, retriever, llm, settings())
    assert ans.abstained and ans.text == REFUSAL and ans.cited == []
    assert ans.abstain_reason == 'the cited sources never name "Wombat"'
    step = ans.trace.steps[-1]
    assert step.name == "entities" and step.data["withdrawn_answer"] == ANSWER
    assert len(llm.calls) == 1  # a withdrawn answer is not retried


def test_entity_check_can_be_turned_off(retriever):  # noqa: F811
    ans = RAGAgent(retriever, ScriptedLLM(ANSWER), settings(entity_check=False)).answer(WRONG)
    assert not ans.abstained and "entities" not in steps(ans)

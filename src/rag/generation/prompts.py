"""Prompt construction for grounded, cited answers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# Bump when the prompt changes; it becomes part of cache keys in Phase 5.
PROMPT_VERSION = "1"

REFUSAL = "I don't have enough information in the indexed documents to answer that."

SYSTEM_PROMPT = f"""You answer questions using only the numbered sources you are given.

Rules:
1. Use only facts stated in the sources. Never use outside knowledge.
2. End every sentence with the tag of the source it comes from, for example [S1] or [S2][S3].
3. If the sources do not contain the answer, reply with exactly this sentence and nothing else:
{REFUSAL}
4. Answer in at most three short sentences.

Example
Sources:
[S1] (Example paper, p. 3) The ferret index stores 384-dimensional vectors.
[S2] (Example paper, p. 5) Queries take 12 milliseconds on average.
Question: How large are the vectors in the ferret index?
Answer: The ferret index stores 384-dimensional vectors [S1]."""


@dataclass(frozen=True)
class PromptSource:
    tag: str  # "S1", "S2", ...
    title: str
    page: int
    text: str


def format_sources(sources: Sequence[PromptSource]) -> str:
    blocks = []
    for s in sources:
        text = " ".join(s.text.split())
        blocks.append(f"[{s.tag}] ({s.title}, p. {s.page}) {text}")
    return "\n\n".join(blocks)


def build_messages(question: str, sources: Sequence[PromptSource]) -> list[dict[str, str]]:
    user = f"Sources:\n{format_sources(sources)}\n\nQuestion: {question.strip()}\nAnswer:"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]

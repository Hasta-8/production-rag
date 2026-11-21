"""Entity check (E14): every name in the question must appear in a cited source.

The 1.7B model sometimes answers with a fact about a nearby model: "LLaMA used 2048 A100
GPUs" becomes the answer for Llama 2 (P35). The reranker scores such passages as highly
relevant, because a relevance score measures topic, not identity. This check is the
identity test: if the question asks about "Llama 2", at least one cited source must say
"Llama 2" (in its text or its paper's title), or the answer is withdrawn.

It is deliberately simple and deterministic (no LLM call). It cannot catch a generic name:
"the original Transformer" is satisfied by any paper that says "transformer" (u01).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_STRIP = "?.,;:!\"'()[]{}"
_VERSION = re.compile(r"^(?:v?\d+(?:\.\d+)*[a-z]?|\d+x\d+[a-z]?)$", re.IGNORECASE)


def _is_name(token: str, first: bool) -> bool:
    has_digit = any(c.isdigit() for c in token)
    has_alpha = any(c.isalpha() for c in token)
    if has_digit and has_alpha:  # GPT-4, 8x22B, DINOv2, A100
        return True
    if any(c.isupper() for c in token[1:]):  # ColPali, QLoRA, BERT, OpenAI
        return True
    return token[:1].isupper() and not first  # Mixtral, Transformer (not "What")


def _clean(token: str) -> str:
    token = token.strip(_STRIP)
    for suffix in ("'s", "’s"):
        token = token.removesuffix(suffix)
    return token.strip(_STRIP)


def question_entities(question: str) -> list[str]:
    """Names in the question, e.g. "How many GPUs were used to train Llama 2?" ->
    ["GPUs", "Llama 2"]."""
    tokens = [_clean(t) for t in question.split()]
    out: list[list[str]] = []
    prev_name = False
    for i, tok in enumerate(tokens):
        if not tok:
            prev_name = False
            continue
        if prev_name and _VERSION.match(tok):  # "Llama" + "2", "Claude" + "2.1"
            out[-1].append(tok)
            continue
        if _is_name(tok, first=(i == 0)):
            capitalised = tok[:1].isupper()
            if prev_name and capitalised and out[-1][-1][:1].isupper():
                out[-1].append(tok)  # "World Cup", "Mistral Large"
            else:
                out.append([tok])
            prev_name = True
        else:
            prev_name = False
    seen, names = set(), []
    for parts in out:
        name = " ".join(parts)
        if name.lower() not in seen:
            seen.add(name.lower())
            names.append(name)
    return names


def entity_pattern(name: str) -> re.Pattern[str]:
    """Whole-word, case-insensitive; parts may be joined by space, hyphen, dot or nothing.

    "Llama 2" matches "Llama 2", "Llama-2", "llama2", but not "LLaMA 2048" or "LLaMA".
    A trailing "s" is optional on either side ("GPUs" matches "GPU").
    """
    parts = [p for p in re.split(r"[^0-9A-Za-z]+", name) if p]
    if parts and len(parts[-1]) > 3 and parts[-1].lower().endswith("s"):
        parts[-1] = parts[-1][:-1]
    body = r"[\s\-_.]*".join(re.escape(p) for p in parts)
    return re.compile(rf"(?<![0-9a-z]){body}s?(?![0-9a-z])", re.IGNORECASE)


@dataclass
class EntityVerdict:
    entities: list[str]
    supported: dict[str, list[str]]  # name -> tags of cited sources that name it
    unsupported: list[str]

    @property
    def passed(self) -> bool:
        return not self.unsupported

    @property
    def reason(self) -> str:
        names = ", ".join(f'"{n}"' for n in self.unsupported)
        return f"the cited sources never name {names}"


def check_entities(question: str, cited: list[tuple[str, str, str]]) -> EntityVerdict:
    """``cited`` is (tag, title, text) for each cited source."""
    entities = question_entities(question)
    supported: dict[str, list[str]] = {}
    for name in entities:
        rx = entity_pattern(name)
        supported[name] = [tag for tag, title, text in cited if rx.search(f"{title}\n{text}")]
    unsupported = [n for n in entities if not supported[n]]
    return EntityVerdict(entities, supported, unsupported)

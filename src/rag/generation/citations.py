"""Parse, check and (when the model leaves them out) infer source citations."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from rag.generation.prompts import REFUSAL, PromptSource

_BRACKET = re.compile(r"\[([^\[\]]{1,40})\]")
_TAG_IN_BRACKET = re.compile(r"S\s*(\d+)", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_WORD = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")
_STOP = set(
    "a an the of and or to in on for with by from at as is are was were be been it its this "
    "that these those which who what how why when where than then into over under about can "
    "could may might not no do does did has have had their they them we our you your".split()
)
REFUSAL_MARKER = "don't have enough information"
# Softer ways the model says "not in the sources". Only a short reply made of such a
# statement counts as a refusal; a real answer that mentions a gap does not.
_SOFT_REFUSALS = (
    "not specified in the provided sources",
    "not mentioned in the provided sources",
    "not stated in the provided sources",
    "the sources do not",
    "the provided sources do not",
    "no information about",
)
# The model sometimes starts re-listing the sources after its answer: "[S1] (Title, p. 3) ...".
_SOURCE_ECHO = re.compile(r"\n\s*\[S\d+\]\s*\(")
_TRAILING_TAGS = re.compile(r"(?<=[.!?])(\s*\[S\d+(?:\s*[,;]\s*S\d+)*\])+\s*$", re.IGNORECASE)


def is_refusal(text: str) -> bool:
    low = text.lower().replace("’", "'")
    if REFUSAL_MARKER in low or text.strip() == REFUSAL:
        return True
    sentences = [s for s in _SENTENCE_END.split(text.strip()) if s.strip()]
    return len(sentences) <= 1 and any(marker in low for marker in _SOFT_REFUSALS)


def clean_answer(text: str) -> tuple[str, list[str]]:
    """Remove output artefacts. Returns the cleaned text and what was removed."""
    removed = []
    echo = _SOURCE_ECHO.search(text)
    if echo:
        text = text[: echo.start()]
        removed.append("echoed source list")
    stripped = _TRAILING_TAGS.sub("", text.rstrip())
    if stripped != text.rstrip():
        removed.append("stray trailing tags")
    return stripped.strip(), removed


def find_tags(text: str) -> list[str]:
    """Tags in order of first appearance: handles [S1], [S2][S3], [S1, S2] and [s1; S4]."""
    seen: list[str] = []
    for bracket in _BRACKET.findall(text):
        for number in _TAG_IN_BRACKET.findall(bracket):
            tag = f"S{int(number)}"
            if tag not in seen:
                seen.append(tag)
    return seen


def remove_tags(text: str, tags: Sequence[str]) -> str:
    """Drop the given tags (e.g. invalid ones) and any bracket left empty."""

    def fix(match: re.Match) -> str:
        kept = [
            f"S{int(n)}"
            for n in _TAG_IN_BRACKET.findall(match.group(1))
            if f"S{int(n)}" not in tags
        ]
        if not _TAG_IN_BRACKET.search(match.group(1)):
            return match.group(0)  # not a citation bracket
        return f"[{', '.join(kept)}]" if kept else ""

    return re.sub(r"\s+([.!?,])", r"\1", _BRACKET.sub(fix, text)).strip()


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


@dataclass
class CitationCheck:
    text: str  # answer text after repairs
    cited: list[str]  # valid tags, in order of appearance
    invalid: list[str]  # tags the model invented (not among the sources)
    inferred: bool  # True if citations were added by code, not the model
    uncited_sentences: int  # sentences still without a citation


def check_citations(
    answer: str, sources: Sequence[PromptSource], min_overlap: float = 0.3
) -> CitationCheck:
    """Validate the model's citations; if it gave none, attribute sentences by word overlap.

    Inferred citations are marked as such, never presented as the model's own.
    """
    valid = {s.tag for s in sources}
    tags = find_tags(answer)
    invalid = [t for t in tags if t not in valid]
    text = remove_tags(answer, invalid) if invalid else answer
    cited = [t for t in tags if t in valid]
    inferred = False

    sentences = [s for s in _SENTENCE_END.split(text) if s.strip()]
    if not cited and sources:
        source_words = {s.tag: _content_words(s.text) for s in sources}
        rebuilt = []
        for sentence in sentences:
            words = _content_words(sentence)
            best_tag, best = None, 0.0
            for tag, sw in source_words.items():
                overlap = len(words & sw) / len(words) if words else 0.0
                if overlap > best:
                    best_tag, best = tag, overlap
            if best_tag and best >= min_overlap:
                stripped = sentence.rstrip()
                end = stripped[-1] if stripped[-1:] in ".!?" else ""
                body = stripped[:-1] if end else stripped
                rebuilt.append(f"{body} [{best_tag}]{end}")
                if best_tag not in cited:
                    cited.append(best_tag)
            else:
                rebuilt.append(sentence)
        text = " ".join(rebuilt)
        inferred = bool(cited)
        sentences = rebuilt

    uncited = sum(1 for s in sentences if not find_tags(s))
    return CitationCheck(text, cited, invalid, inferred, uncited)

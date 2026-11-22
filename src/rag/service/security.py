"""Security layer: input screening, PII masking and output checks (design 5.10).

Regex catches common attacks quickly and without an LLM call, but a determined attacker can
get past it. It is one layer of several: the model has no tools that can do damage, the
prompt treats sources as data, answers must cite the sources, and outputs are checked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- injection

_I = re.IGNORECASE
INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "ignore_instructions",
        re.compile(
            r"\b(?:ignore|disregard|forget|override|bypass)\b[^.?!\n]{0,40}?"
            r"\b(?:previous|prior|above|earlier|preceding|all|any|your|the|these)\b"
            r"[^.?!\n]{0,20}?\b(?:instructions?|rules|prompts?|directions|guidelines)\b",
            _I,
        ),
    ),
    (
        "reveal_prompt",
        re.compile(
            r"\b(?:reveal|show|print|repeat|output|display|leak|tell me|what(?:'s| is| are))\b"
            r"[^.?!\n]{0,30}?\b(?:system|hidden|initial|original|developer)\s+"
            r"(?:prompt|instructions?|message|rules)\b",
            _I,
        ),
    ),
    (
        "role_play",
        re.compile(
            r"\b(?:you are now|from now on,? you|act as|pretend (?:to be|you are|you're)|"
            r"role-?play as|imagine you are)\b[^.?!\n]{0,60}?"
            r"\b(?:DAN|jailbroken|unrestricted|unfiltered|uncensored|evil|"
            r"no (?:rules|restrictions|limits|filters)|developer mode)\b",
            _I,
        ),
    ),
    ("jailbreak_persona", re.compile(r"\b(?:do anything now|developer mode enabled)\b", _I)),
]

# Chat-template tokens and fake role markers: a question must not be able to close the
# user turn and open a "system" one.
_DELIMITERS = re.compile(
    r"<\|[^|>]{1,30}\|>"  # <|im_start|>, <|im_end|>, <|endoftext|>
    r"|</?(?:system|assistant|user|think|tool|instructions?)\b[^>]{0,40}>"
    r"|\[/?INST\]|<</?SYS>>"
    r"|^\s*#{2,}\s*(?:system|instructions?|assistant)\b[^\n]*",
    _I | re.MULTILINE,
)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def find_injection(text: str) -> str | None:
    """The name of the first injection pattern found, or None."""
    for name, rx in INJECTION_PATTERNS:
        if rx.search(text):
            return name
    return None


def strip_delimiters(text: str) -> tuple[str, list[str]]:
    removed = [m.group(0).strip() for m in _DELIMITERS.finditer(text)]
    text = _DELIMITERS.sub(" ", text)
    if _CONTROL.search(text):
        removed.append("control characters")
        text = _CONTROL.sub("", text)
    return " ".join(text.split()), removed


# --------------------------------------------------------------------------- PII

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# Phones need a country code, an area code in brackets, or dash/dot separators: research
# text is full of space-separated numbers (table rows) that must not be masked.
_PHONE = re.compile(
    r"(?<![\w+])\+\d{1,3}[\s.-]?\(?\d{1,4}\)?(?:[\s.-]?\d{2,4}){2,3}(?!\w)"
    r"|(?<!\w)\(\d{3}\)\s?\d{3}[-.]\d{4}(?!\w)"
    r"|(?<![\w.-])\d{3}[-.]\d{3}[-.]\d{4}(?![\w.-])"
)
# Cards: 4-4-4-4(-3) or 4-6-5 groups with one consistent separator, or 13-19 digits in a
# row; then the Luhn checksum.
_CARD = re.compile(
    r"(?<![\d.,])(?:\d{4}([ -])\d{4}\1\d{4}\1\d{1,7}|\d{4}([ -])\d{6}\2\d{5}|\d{13,19})(?![\d.,])"
)
_NATIONAL_IDS = [
    ("ssn", re.compile(r"(?<![\w-])\d{3}-\d{2}-\d{4}(?![\w-])")),
    ("aadhaar", re.compile(r"(?<![\d.,])[2-9]\d{3}([ -]?)\d{4}\1\d{4}(?![\d.,])")),
    ("pan", re.compile(r"\b[A-Z]{3}[PCHABGJLFT][A-Z]\d{4}[A-Z]\b")),
    (
        "uk_nino",
        re.compile(
            r"\b(?!BG|GB|NK|KN|TN|NT|ZZ)[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b"
        ),
    ),
]


_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def _verhoeff(digits: str) -> bool:
    """Aadhaar's checksum: rejects about 90% of random 12-digit numbers (table rows)."""
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def mask_pii(text: str) -> tuple[str, list[str]]:
    """Replace emails, phone numbers, card numbers and national IDs with placeholders."""
    kinds: list[str] = []

    def sub(rx: re.Pattern[str], label: str, s: str, check=None) -> str:
        def repl(m: re.Match[str]) -> str:
            if check is not None and not check(m.group(0)):
                return m.group(0)
            kinds.append(label)
            return f"[{label.upper()}]"

        return rx.sub(repl, s)

    text = sub(_EMAIL, "email", text)
    for label, rx in _NATIONAL_IDS:  # before cards/phones: their digits overlap
        check = (lambda s: _verhoeff(_digits(s))) if label == "aadhaar" else None
        text = sub(rx, label, text, check)
    text = sub(_CARD, "card", text, check=lambda s: _luhn(_digits(s)))
    text = sub(_PHONE, "phone", text)
    return text, sorted(set(kinds))


# --------------------------------------------------------------------------- pipeline


@dataclass
class InputScreen:
    text: str  # what retrieval and the model see
    blocked: str | None = None  # injection pattern name, if any
    masked: list[str] = field(default_factory=list)
    stripped: list[str] = field(default_factory=list)


def screen_input(question: str) -> InputScreen:
    """Strip delimiter tricks, block injection phrases, mask PII (in that order)."""
    text, stripped = strip_delimiters(question)
    blocked = find_injection(text) or find_injection(question)
    text, masked = mask_pii(text)
    return InputScreen(text, blocked, masked, stripped)


def leaks_prompt(answer: str, system_prompt: str, min_chars: int = 40) -> bool:
    """True if the answer repeats a sentence-sized piece of the system prompt."""
    norm = " ".join(answer.lower().split())
    for line in system_prompt.splitlines():
        line = " ".join(line.lower().split())
        if len(line) >= min_chars and line in norm:
            return True
    return False


def scan_document(pages: list[str]) -> list[dict]:
    """Indirect injection: instructions planted in a document's text (flagged, not blocked)."""
    hits = []
    for page, text in enumerate(pages, start=1):
        name = find_injection(text)
        if name:
            hits.append({"page": page, "pattern": name})
    return hits

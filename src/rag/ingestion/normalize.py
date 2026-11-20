"""Text clean-up for extracted PDF text, and content-based document IDs."""

from __future__ import annotations

import hashlib
import re
import unicodedata

# Bump when normalisation changes, so indexes built with the old rules are rebuilt.
NORMALIZER_VERSION = "1"

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SPACES = re.compile(r"[ \t\f\v ]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_ALL_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Clean extracted text while keeping line and paragraph breaks for the splitter.

    - NFKC turns PDF ligatures (ﬁ, ﬂ, ﬃ) and compatibility characters into plain text.
    - Words hyphenated across a line break are joined ("retrie-\\nval" -> "retrieval").
      Real hyphenated compounds split at a line end lose their hyphen; that is an
      accepted trade-off of this simple rule.
    - Runs of spaces collapse to one; 3+ newlines collapse to a paragraph break.
    """
    text = unicodedata.normalize("NFKC", text).replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return _MANY_NEWLINES.sub("\n\n", text).strip()


def content_fingerprint(text: str) -> str:
    """SHA-256 of text with all whitespace removed from consideration."""
    canonical = _ALL_WHITESPACE.sub(" ", text).strip()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def make_doc_id(page_texts: list[str]) -> str:
    """Stable document ID from content, so re-ingesting the same file is a no-op.

    16 hex characters (64 bits) keeps chunk IDs readable; collisions are negligible at
    corpus sizes far beyond this project.
    """
    return content_fingerprint("\n".join(page_texts))[:16]

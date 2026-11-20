"""Chunk filters applied at query time.

Reference lists (bibliographies) are about 12% of the chunks in a paper corpus. They
match a question's keywords (paper titles contain the same terms), but they almost
never answer it: "Lewis et al. 2020. Retrieval-augmented generation..." is not an
explanation of RAG. The filter is applied at query time, not at ingestion, so the
index (and its version) stays the same and the effect can be measured both ways.
"""

from __future__ import annotations

import re

_YEAR = re.compile(r"\b(?:19|20)\d{2}[a-z]?\b")
_VENUE = re.compile(
    r"Proceedings|arXiv|Conference|Journal|preprint|In Advances|Transactions|Workshop|"
    r"\bIn:|\bURL\b|https?://|\bdoi\b|"
    r"\b(?:NeurIPS|NIPS|ICLR|ICML|CVPR|ICCV|ECCV|ACL|EMNLP|NAACL|EACL|COLING|SIGIR|KDD|AAAI|"
    r"IJCAI|TACL|JMLR|WWW|CIKM|WSDM|TREC)\b"
)
# "J. Smith", "Smith, J.", "Smith, J.,": author names with initials.
_AUTHOR = re.compile(r"\b[A-Z]\. ?(?:[A-Z]\. ?)?[A-Z][a-z]+|\b[A-Z][a-z]+, (?:[A-Z]\.){1,3}")
# Numbered entries: "[12] Name" or "12. Name, X."
_NUMBERED = re.compile(r"\[\d{1,3}\] [A-Z]|(?:^|\s)\d{1,3}\. [A-Z][a-z]+, [A-Z]\.")

FILTER_VERSION = "2"


def reference_signals(text: str) -> dict[str, int]:
    return {
        "years": len(_YEAR.findall(text)),
        "venues": len(_VENUE.findall(text)),
        "authors": len(_AUTHOR.findall(text)),
        "numbered": len(_NUMBERED.findall(text)),
    }


def looks_like_references(text: str) -> bool:
    """Heuristic: does this chunk come from a bibliography?

    A bibliography chunk has many years plus bibliographic structure (venues, author
    initials or numbered entries). Prose with in-text citations, "(Smith et al., 2020)",
    has years but none of that structure, so it is kept.
    """
    s = reference_signals(text)
    structure = s["venues"] + s["authors"] // 3 + s["numbered"]
    return s["years"] >= 4 and structure >= 3

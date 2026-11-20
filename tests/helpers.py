"""Test doubles that need no network or API keys."""

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings

_WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


class HashingEmbeddings(Embeddings):
    """Bag-of-words vectors via feature hashing.

    Texts sharing words get similar vectors, so retrieval tests behave like real
    (if crude) search, deterministically and offline. ``calls`` counts texts embedded.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.calls = 0

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in _WORD.findall(text.lower()):
            h = int.from_bytes(hashlib.sha256(word.encode()).digest()[:8], "big")
            vec[h % self.dim] += 1.0 if (h >> 63) == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += len(texts)
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return self._embed(text)


def word_count(text: str) -> int:
    """Stand-in token counter, so unit tests do not need tiktoken's download."""
    return len(text.split())

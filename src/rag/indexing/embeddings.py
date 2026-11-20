"""Embeddings with a persistent cache, so unchanged text is never embedded (or paid for) twice."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from array import array
from pathlib import Path

from langchain_core.embeddings import Embeddings


class CachedEmbeddings(Embeddings):
    """Wraps any LangChain ``Embeddings`` with a SQLite cache.

    The key is SHA-256 of (model name, text), so switching models never returns
    vectors from the wrong model. Vectors are stored as float32.
    """

    def __init__(self, underlying: Embeddings, model_name: str, cache_path: Path) -> None:
        self.underlying = underlying
        self.model_name = model_name
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(cache_path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vec BLOB)")
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _key(self, text: str, kind: str = "doc") -> str:
        # Documents and queries are separate: models like BGE embed queries with an
        # instruction prefix, so the same text has a different vector in each role.
        return hashlib.sha256(f"{self.model_name}\x00{kind}\x00{text}".encode()).hexdigest()

    def _get(self, keys: list[str]) -> dict[str, list[float]]:
        found: dict[str, list[float]] = {}
        for start in range(0, len(keys), 500):  # stay under SQLite's parameter limit
            batch = keys[start : start + 500]
            marks = ",".join("?" * len(batch))
            rows = self._db.execute(f"SELECT key, vec FROM vectors WHERE key IN ({marks})", batch)
            for key, blob in rows:
                found[key] = array("f", blob).tolist()
        return found

    def _put(self, items: list[tuple[str, list[float]]]) -> None:
        self._db.executemany(
            "INSERT OR REPLACE INTO vectors (key, vec) VALUES (?, ?)",
            [(k, array("f", v).tobytes()) for k, v in items],
        )
        self._db.commit()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key(t) for t in texts]
        with self._lock:
            cached = self._get(keys)
            missing = [i for i, k in enumerate(keys) if k not in cached]
            self.hits += len(texts) - len(missing)
            self.misses += len(missing)
            if missing:
                vectors = self.underlying.embed_documents([texts[i] for i in missing])
                new = [(keys[i], v) for i, v in zip(missing, vectors, strict=True)]
                self._put(new)
                cached.update(new)
        return [cached[k] for k in keys]

    def embed_query(self, text: str) -> list[float]:
        key = self._key(text, "query")
        with self._lock:
            cached = self._get([key])
            if key in cached:
                self.hits += 1
                return cached[key]
            self.misses += 1
            vector = self.underlying.embed_query(text)
            self._put([(key, vector)])
        return vector

    def close(self) -> None:
        self._db.close()


class FastEmbedEmbeddings(Embeddings):
    """Local embeddings through fastembed (ONNX Runtime, CPU, no PyTorch).

    ``query_prefix`` is prepended to queries only. BGE models recommend an instruction
    for short-query-to-passage retrieval; documents are embedded as they are.
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: Path,
        query_prefix: str = "",
        threads: int | None = None,
        batch_size: int = 32,
    ) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self._model = TextEmbedding(
            model_name=model_name, cache_dir=str(cache_dir), threads=threads
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.passage_embed(texts, batch_size=self.batch_size)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(self.query_prefix + text))).tolist()


def make_embeddings(
    provider: str,
    model: str,
    cache_dir: Path,
    query_prefix: str = "",
    threads: int | None = None,
    openai_api_key: str | None = None,
) -> CachedEmbeddings:
    """Build the configured embedding model, wrapped in the persistent cache.

    The cache key includes the query prefix, so changing it never reuses old query vectors.
    """
    if provider == "fastembed":
        underlying: Embeddings = FastEmbedEmbeddings(
            model, cache_dir / "fastembed", query_prefix=query_prefix, threads=threads
        )
    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        if not openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the openai embedding provider")
        underlying = OpenAIEmbeddings(model=model, api_key=openai_api_key)
        query_prefix = ""
    else:
        raise ValueError(f"unknown embedding provider {provider!r}")
    cache_name = f"{provider}:{model}|q={query_prefix}"
    return CachedEmbeddings(underlying, cache_name, cache_dir / "embeddings.sqlite")

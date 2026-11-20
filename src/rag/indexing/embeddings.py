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

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model_name}\x00{text}".encode()).hexdigest()

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
        key = self._key(text)
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


def openai_embeddings(model: str, api_key: str, cache_dir: Path) -> CachedEmbeddings:
    from langchain_openai import OpenAIEmbeddings

    underlying = OpenAIEmbeddings(model=model, api_key=api_key)
    return CachedEmbeddings(underlying, model, cache_dir / "embeddings.sqlite")

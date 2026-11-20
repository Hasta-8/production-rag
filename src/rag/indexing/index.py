"""The versioned index: a Chroma collection plus an ``index.json`` describing it.

The index version is a hash of everything that makes vectors comparable: the
embedding model, chunk settings and the normaliser and chunker versions. Changing
any of them gives a new version in a new folder, so old and new vectors never mix.
Which documents are inside is tracked in ``index.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from rag.ingestion.chunkers import CHUNKER_VERSION
from rag.ingestion.normalize import NORMALIZER_VERSION

COLLECTION = "chunks"

# Chroma sends anonymous usage telemetry by default; this project opts out.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


def compute_index_version(embedding_model: str, chunk_size: int, chunk_overlap: int) -> str:
    spec = {
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "normalizer": NORMALIZER_VERSION,
        "chunker": CHUNKER_VERSION,
    }
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
    return f"v-{digest}"


@dataclass
class DocumentEntry:
    doc_id: str
    source: str
    title: str
    pages: int
    chunks: int
    tokens: int


@dataclass
class IndexInfo:
    version: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    normalizer_version: str
    chunker_version: str
    created_at: str
    updated_at: str
    documents: dict[str, DocumentEntry] = field(default_factory=dict)

    def doc_id_for_source(self, source: str) -> str | None:
        for entry in self.documents.values():
            if entry.source == source:
                return entry.doc_id
        return None

    @property
    def total_chunks(self) -> int:
        return sum(d.chunks for d in self.documents.values())


class IndexMismatchError(RuntimeError):
    """The index on disk was built with different settings than the ones configured."""


class VersionedIndex:
    def __init__(self, root: Path, info: IndexInfo, embeddings: Embeddings) -> None:
        self.root = root
        self.info = info
        self.embeddings = embeddings
        self.store = Chroma(
            collection_name=COLLECTION,
            embedding_function=embeddings,
            persist_directory=str(root),
            collection_metadata={"hnsw:space": "cosine"},
        )

    @property
    def info_path(self) -> Path:
        return self.root / "index.json"

    @classmethod
    def open(
        cls,
        index_dir: Path,
        embeddings: Embeddings,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> VersionedIndex:
        version = compute_index_version(embedding_model, chunk_size, chunk_overlap)
        root = index_dir / version
        info_path = root / "index.json"
        if info_path.exists():
            info = _load_info(info_path)
            if info.embedding_model != embedding_model:
                raise IndexMismatchError(
                    f"index {version} was built with {info.embedding_model}, "
                    f"but {embedding_model} is configured"
                )
        else:
            now = datetime.now(UTC).isoformat(timespec="seconds")
            info = IndexInfo(
                version=version,
                embedding_model=embedding_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                normalizer_version=NORMALIZER_VERSION,
                chunker_version=CHUNKER_VERSION,
                created_at=now,
                updated_at=now,
            )
        root.mkdir(parents=True, exist_ok=True)
        return cls(root, info, embeddings)

    def save_info(self) -> None:
        self.info.updated_at = datetime.now(UTC).isoformat(timespec="seconds")
        data = asdict(self.info)
        tmp = self.info_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        tmp.replace(self.info_path)

    def count(self) -> int:
        return self.store._collection.count()


def _load_info(path: Path) -> IndexInfo:
    data = json.loads(path.read_text())
    docs = {k: DocumentEntry(**v) for k, v in data.pop("documents", {}).items()}
    return IndexInfo(**data, documents=docs)

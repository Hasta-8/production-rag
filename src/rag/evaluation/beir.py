"""Public retrieval benchmark: BEIR datasets (SciFact by default).

The dataset is downloaded for local evaluation only (git-ignored, never redistributed),
indexed in its own Chroma collection, and scored with the standard metrics: nDCG@10
(BEIR's main metric) and Recall@100, using the official relevance judgements (qrels).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import time
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import httpx
from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.retrieval.bm25 import BM25Index, StoredChunk
from rag.retrieval.hybrid import Retriever
from rag.retrieval.rerank import Reranker

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


@dataclass
class BeirDataset:
    name: str
    corpus: dict[str, dict]  # doc id -> {"title", "text"}
    queries: dict[str, str]  # query id -> text
    qrels: dict[str, dict[str, int]]  # query id -> {doc id: relevance}


def download(name: str, dest: Path, client: httpx.Client | None = None) -> tuple[Path, str]:
    """Download and unzip a BEIR dataset once. Returns its folder and the zip's SHA-256."""
    folder = dest / name
    marker = folder / ".sha256"
    if marker.exists():
        return folder, marker.read_text().strip()
    own_client = client is None
    client = client or httpx.Client(timeout=300, follow_redirects=True)
    try:
        response = client.get(BEIR_URL.format(name=name))
        response.raise_for_status()
    finally:
        if own_client:
            client.close()
    digest = hashlib.sha256(response.content).hexdigest()
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        zf.extractall(dest)  # the archive contains a top-level <name>/ folder
    marker.write_text(digest)
    return folder, digest


def load(folder: Path, split: str = "test") -> BeirDataset:
    corpus = {}
    for line in (folder / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        corpus[str(row["_id"])] = {"title": row.get("title", ""), "text": row.get("text", "")}
    queries = {}
    for line in (folder / "queries.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        queries[str(row["_id"])] = row["text"]
    qrels: dict[str, dict[str, int]] = {}
    with (folder / "qrels" / f"{split}.tsv").open(encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)  # header: query-id, corpus-id, score
        for qid, did, score in reader:
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)
    queries = {qid: queries[qid] for qid in qrels}  # evaluate only judged queries
    return BeirDataset(folder.name, corpus, queries, qrels)


def doc_text(doc: dict) -> str:
    """BEIR convention: title and text together."""
    return f"{doc['title']}. {doc['text']}".strip(". ") if doc["title"] else doc["text"]


# ---------------------------------------------------------------- metrics


def ndcg_at_k(ranked: list[str], relevant: dict[str, int], k: int = 10) -> float:
    """nDCG with linear gains (as in trec_eval / pytrec_eval ndcg_cut)."""
    dcg = sum(relevant.get(doc, 0) / math.log2(i + 2) for i, doc in enumerate(ranked[:k]))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall_at_k(ranked: list[str], relevant: dict[str, int], k: int = 100) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


# ---------------------------------------------------------------- index and run


class CorpusIndex:
    """Minimal stand-in for VersionedIndex: the retriever only needs `.store`."""

    def __init__(self, store: Chroma) -> None:
        self.store = store


def build_index(
    dataset: BeirDataset,
    persist_dir: Path,
    embeddings,
    batch: int = 256,
    on_progress: Callable[[int, int], None] = lambda _done, _total: None,
) -> tuple[CorpusIndex, BM25Index]:
    store = Chroma(
        collection_name=f"beir-{dataset.name}",
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )
    ids = sorted(dataset.corpus)
    present = set(store.get(include=[])["ids"])
    missing = [i for i in ids if i not in present]
    for start in range(0, len(missing), batch):
        part = missing[start : start + batch]
        store.add_documents(
            [
                Document(
                    page_content=doc_text(dataset.corpus[i]),
                    metadata={"doc_id": i, "title": dataset.corpus[i]["title"][:200], "page": 0},
                )
                for i in part
            ],
            ids=part,
        )
        on_progress(len(present) + start + len(part), len(ids))
    chunks = [
        StoredChunk(i, doc_text(dataset.corpus[i]), {"doc_id": i, "title": "", "page": 0})
        for i in ids
    ]
    return CorpusIndex(store), BM25Index(chunks)


def evaluate_mode(
    dataset: BeirDataset,
    index: CorpusIndex,
    bm25: BM25Index,
    mode: str,
    reranker: Reranker | None = None,
    depth: int = 100,
    rerank_depth: int = 100,
    queries: Iterable[str] | None = None,
) -> dict:
    retriever = Retriever(
        index, bm25, reranker=reranker, mode=mode, candidates=rerank_depth if reranker else depth
    )
    qids = list(queries or dataset.queries)
    ndcgs, recalls, times = [], [], []
    for qid in qids:
        start = time.perf_counter()
        ranked = [c.chunk_id for c in retriever.search(dataset.queries[qid], k=depth)]
        times.append(time.perf_counter() - start)
        ndcgs.append(ndcg_at_k(ranked, dataset.qrels[qid], 10))
        recalls.append(recall_at_k(ranked, dataset.qrels[qid], 100))
    times.sort()
    return {
        "mode": mode,
        "queries": len(qids),
        "ndcg@10": round(sum(ndcgs) / len(ndcgs), 4),
        "recall@100": round(sum(recalls) / len(recalls), 4),
        "p50_ms": round(times[len(times) // 2] * 1000, 1),
        "p95_ms": round(times[int(len(times) * 0.95) - 1] * 1000, 1),
    }

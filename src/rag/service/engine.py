"""Everything a request needs, built once at startup (the lifespan's job, design 5.11).

Tests build an ``Engine`` directly from fakes; ``build_engine`` builds the real one from
settings: the versioned index, one BM25 index and one reranker shared by all retrieval
modes, and the LLM gateway (Qwen3-1.7B with the 0.6B fallback).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from rag.generation.answer import Answer, AnswerSettings, answer_question
from rag.generation.llm import LLMClient
from rag.indexing.index import VersionedIndex
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.hybrid import MODES, Retriever
from rag.retrieval.rerank import Reranker

AnswerFn = Callable[..., Answer]  # (question, on_token=None, trace=None) -> Answer
SERVABLE_LICENCES = {"open", "noncommercial", "uploaded"}


@dataclass
class Engine:
    index: VersionedIndex
    llm: LLMClient
    opts: AnswerSettings
    reranker: Reranker | None
    default_mode: str = "hybrid_rerank"
    default_pipeline: str = "agent"
    pdf_dirs: list[Path] = field(default_factory=list)  # where indexed files can be served from
    doc_meta: dict[str, dict] = field(default_factory=dict)  # source file name -> manifest row
    retriever_options: dict = field(default_factory=dict)
    revision: int = 0  # bumped by every upload or delete; part of the cache key
    _retrievers: dict[str, Retriever] = field(default_factory=dict, init=False)
    _answerers: dict[tuple[str, str], AnswerFn] = field(default_factory=dict, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        self.refresh(bump=False)

    # ------------------------------------------------------------ building blocks

    def refresh(self, bump: bool = True) -> None:
        """Rebuild BM25 from the store (after an upload or delete) and every retriever."""
        bm25 = BM25Index.from_store(self.index.store)
        retrievers = {
            mode: Retriever(
                self.index,
                bm25,
                reranker=self.reranker if mode.endswith("_rerank") else None,
                mode=mode,
                **self.retriever_options,
            )
            for mode in MODES
            if self.reranker is not None or not mode.endswith("_rerank")
        }
        with self._lock:
            self._retrievers = retrievers
            self._answerers = {}
            if bump:
                self.revision += 1

    def retriever(self, mode: str | None) -> Retriever:
        mode = mode or self.default_mode
        try:
            return self._retrievers[mode]
        except KeyError:
            raise ValueError(f"retrieval mode {mode!r} is not available") from None

    def answerer(self, pipeline: str | None, mode: str | None) -> AnswerFn:
        pipeline = pipeline or self.default_pipeline
        mode = mode or self.default_mode
        with self._lock:
            fn = self._answerers.get((pipeline, mode))
        if fn is not None:
            return fn
        retriever = self.retriever(mode)
        if pipeline == "agent":
            from rag.agent import RAGAgent

            fn = RAGAgent(retriever, self.llm, self.opts).answer
        else:

            def fn(question, on_token=None, trace=None):
                return answer_question(question, retriever, self.llm, self.opts, on_token, trace)

        with self._lock:
            self._answerers[(pipeline, mode)] = fn
        return fn

    # ------------------------------------------------------------ documents

    def licence(self, source: str) -> str:
        return str(self.doc_meta.get(source, {}).get("licence_class", "uploaded"))

    def file_path(self, source: str) -> Path | None:
        """The indexed file behind ``source``, only from the allowed folders."""
        for folder in self.pdf_dirs:
            path = (folder / source).resolve()
            if path.parent == folder.resolve() and path.is_file():
                return path
        return None

    def servable(self, source: str) -> bool:
        return self.licence(source) in SERVABLE_LICENCES and self.file_path(source) is not None

    @property
    def models(self) -> dict[str, str | None]:
        primary = getattr(self.llm, "primary", self.llm)
        fallback = getattr(self.llm, "fallback", None)
        return {"primary": primary.name, "fallback": fallback.name if fallback else None}


def build_engine(settings) -> Engine:
    """The real engine: opens the index and loads the reranker and the main LLM."""
    from rag import corpus
    from rag.runtime import answer_settings, make_llm, make_reranker, open_index

    index, _ = open_index(settings)
    reranker = make_reranker(settings)
    llm = make_llm(settings)
    manifest = corpus.load_manifest(settings.corpus_manifest)
    engine = Engine(
        index=index,
        llm=llm,
        opts=answer_settings(settings),
        reranker=reranker,
        default_mode=settings.retrieval_mode,
        default_pipeline=settings.pipeline,
        pdf_dirs=[settings.pdf_dir, settings.uploads_dir],
        doc_meta={
            p.filename: {
                "licence_class": p.licence_class.value,
                "arxiv_id": p.arxiv_id,
                "abs_url": p.abs_url,
            }
            for p in manifest.papers
        },
        retriever_options={
            "candidates": settings.rerank_candidates,
            "rrf_k": settings.rrf_k,
            "skip_references": settings.skip_references,
        },
    )
    # Warm-up (D8): load the main model now, so /ready only turns 200 when the first
    # question won't wait minutes for a cold load. The fallback loads only if needed.
    primary = getattr(llm, "primary", llm)
    if hasattr(primary, "load"):
        primary.load()
    return engine

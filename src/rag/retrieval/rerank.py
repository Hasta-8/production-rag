"""Cross-encoder reranking: reads the question and each passage together and scores relevance.

Far more accurate than comparing separately computed embeddings, but it must run once
per (question, passage) pair, so it only rescores a short candidate list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """fastembed's ONNX cross-encoder (CPU, no PyTorch). Scores are raw logits."""

    def __init__(
        self,
        model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
        cache_dir: Path | None = None,
        threads: int | None = None,
        batch_size: int = 16,
    ) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.name = model_name
        self.batch_size = batch_size
        self._model = TextCrossEncoder(
            model_name=model_name, cache_dir=str(cache_dir) if cache_dir else None, threads=threads
        )

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        return [float(s) for s in self._model.rerank(query, passages, batch_size=self.batch_size)]

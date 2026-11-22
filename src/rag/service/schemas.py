"""API request and response models, and the one error shape (design 5.11)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Mode = Literal["dense", "bm25", "hybrid", "dense_rerank", "hybrid_rerank"]
Pipeline = Literal["agent", "single"]


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=4000)
    mode: Mode | None = None  # retrieval mode; default from settings (hybrid_rerank)
    pipeline: Pipeline | None = None  # default from settings (agent)
    include_trace: bool = True


class SourceOut(BaseModel):
    tag: str
    doc_id: str
    chunk_id: str
    title: str
    page: int
    score: float
    snippet: str
    cited: bool
    file_url: str | None  # opens the cited page; None if the licence forbids serving it


class SecurityOut(BaseModel):
    masked: list[str] = []  # PII kinds masked in the question
    stripped: int = 0  # delimiter tricks removed from the question
    output_masked: list[str] = []  # PII kinds masked in the answer


class QueryResponse(BaseModel):
    answer: str
    abstained: bool
    abstain_reason: str | None
    sources: list[SourceOut]
    cited: list[str]
    model_used: str | None
    fallback_reason: str | None
    pipeline: str
    mode: str
    retries: int
    cached: bool
    latency_ms: float
    request_id: str
    index_version: str
    prompt_version: str
    security: SecurityOut
    trace: list[dict[str, Any]] | None


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=4000)
    modes: list[Mode] = Field(default=["dense", "hybrid", "hybrid_rerank"], min_length=1)
    k: int = Field(default=5, ge=1, le=10)


class ErrorOut(BaseModel):
    error: str  # stable machine-readable code, e.g. "rate_limited"
    detail: str  # short human-readable message; never a stack trace or a host path
    request_id: str

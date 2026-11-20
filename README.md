# Production RAG

A question-answering service over a collection of openly licensed AI/ML papers.
It answers only from those documents and cites the page for every claim.
Retrieval is hybrid (BM25 plus embeddings, fused with Reciprocal Rank Fusion)
with a cross-encoder reranker. A LangGraph agent checks the evidence, rewrites
the query when retrieval is weak, and declines to answer rather than guess.
The pipeline runs behind a FastAPI service with production safeguards, and a
web UI shows each pipeline step next to the answer.

Everything runs **locally on a laptop CPU with no paid APIs**: embeddings use
`BAAI/bge-small-en-v1.5` (a quantized ONNX export) through fastembed (ONNX
Runtime, no PyTorch), and answers will come from Qwen3-1.7B (Q8_0 GGUF) through
llama.cpp, with Qwen3-0.6B as a faster fallback.

Model files live in a models folder set by `RAG_MODELS_DIR` (default
`.cache/models`), so several projects can share one copy of each model:
`llm/` holds the GGUF files and `fastembed/` holds the embedding model cache.

> **Status: early development.** Ingestion and a baseline question-answering
> pipeline with citations are built, tested and evaluated. Everything below
> marked PLANNED is not built yet.

## Capability status

| Capability | Status |
|---|---|
| Repository setup: secrets excluded, locked dependencies, lint and tests in CI | IMPLEMENTED |
| Corpus: licence check against arXiv metadata, manifest, hash-pinned downloads | IMPLEMENTED |
| Ingestion with page metadata, content-based IDs, duplicate detection | IMPLEMENTED |
| Page-scoped chunking, cached embeddings, versioned Chroma index | IMPLEMENTED |
| Dense search from the command line (local bge-small embeddings) | IMPLEMENTED |
| Known-phrase check of the index (`rag eval phrases`) | IMPLEMENTED |
| Baseline RAG: cited answers from Qwen3-1.7B, abstention, pipeline trace (`rag ask`) | IMPLEMENTED |
| Golden question set and evaluation harness (`rag eval golden`) | IMPLEMENTED |
| Hybrid retrieval (BM25 + dense, RRF) and cross-encoder reranking | PLANNED |
| BEIR SciFact benchmark | PLANNED |
| LangGraph correction loop with bounded retries and model fallback | PLANNED |
| FastAPI service: security, caching, rate limits, streaming, observability | PLANNED |
| Demo UI: chat, citation viewer, trace panel, compare mode | PLANNED |
| Local demo: Docker, start script, Cloudflare tunnel | PLANNED |

## Corpus

The demo corpus is a set of AI/ML papers from arXiv. Each paper's licence is
read from arXiv's metadata, and only papers whose licence allows redistribution
(Creative Commons) are downloaded and served: currently 28 of 78 candidates.
Many well-known papers use arXiv's default licence, which allows only arXiv to
distribute them, so they are not included.

```bash
uv run rag corpus check      # look up licences, write corpus/manifest.yaml
uv run rag corpus download   # fetch redistributable PDFs, pin their SHA-256
uv run rag ingest            # chunk, embed and index them locally (~14 min on a 4-core CPU)
uv run rag search "How does ColBERTv2 reduce the storage cost of late interaction?"
uv run rag eval phrases      # sanity check: do sampled sentences find their own page?
```

Current index: 28 papers, 2,835 chunks. Baseline dense-only results are in
the table below. They come from the known-phrase check: 200 sentences sampled
from the index (seed 7), each used as a query.

| Sentences | n | Right page at rank 1 | Right page in top 5 | Right paper at rank 1 |
|---|---|---|---|---|
| Prose | 164 | 54% | 70% | 74% |
| Reference-list entries | 36 | 19% | 39% | 33% |

Dense embeddings capture meaning rather than exact names and titles, so
reference-list entries do poorly. Hybrid search with BM25 is the planned fix.

## Asking questions

```bash
uv run rag ask "How does ColBERTv2 reduce the storage cost of late interaction?" --trace
```

The answer streams from Qwen3-1.7B on the CPU, with a `[S#]` tag after each
sentence pointing to a paper and page. `--trace` shows each step: retrieved
chunks and scores, the relevance check, generation timings, and the citation
check. If nothing relevant is retrieved, or the model finds no answer in the
sources, it declines instead of guessing.

## Baseline evaluation (dense retrieval)

`eval/golden.yaml` has 30 answerable questions across 24 papers and 5 that must
be declined. Several of those are traps: the corpus contains a similar-looking
fact, such as RETRO's dropout rate, that must not be given as the answer. Each
answerable question's evidence pages are found by searching the paper's text
for evidence phrases, not by the retriever. Results from
`eval/results/2026-11-20_baseline-dense-v2.json`:

| Metric | Baseline |
|---|---|
| Evidence page in top 3 / top 10 retrieved | 70% / 90% |
| Right paper ranked first | 87% |
| Unanswerable questions declined (all traps) | **5 / 5** |
| Answerable questions answered | 87% |
| Answers with citations (none invented, none inferred) | 100% |
| Answers citing an evidence page | 73% |
| Expected keywords present in answers | 0.75 |
| Median time per question (4-core laptop CPU) | 36 s |

Most failures are **retrieval misses**: the evidence page wasn't among the 3
chunks given to the model. In two of those cases the model combined unrelated
passages into a wrong answer instead of declining. Hybrid search, reranking and
an evidence-grading step are the planned fixes.

## Development

Requirements: Python 3.12+ and [uv](https://docs.astral.sh/uv/). No API keys
are needed; an optional `HF_TOKEN` speeds up the first model download.

```bash
uv sync                      # install dependencies from uv.lock
cp .env.example .env         # optional settings
uv run ruff check .          # lint
uv run ruff format --check . # formatting
uv run pytest                # tests
```

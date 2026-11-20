# Production RAG

A question-answering service over a collection of openly licensed AI/ML papers.
It answers only from those documents and cites the page for every claim.
Retrieval is hybrid (BM25 plus embeddings, fused with Reciprocal Rank Fusion)
with a cross-encoder reranker. A LangGraph agent checks the evidence, rewrites
the query when retrieval is weak, and declines to answer rather than guess.
The pipeline runs behind a FastAPI service with production safeguards, and a
web UI shows each pipeline step next to the answer.

> **Status: early development.** Corpus tooling and ingestion are built and
> tested. Everything below marked PLANNED is not built yet.

## Capability status

| Capability | Status |
|---|---|
| Repository setup: secrets excluded, locked dependencies, lint and tests in CI | IMPLEMENTED |
| Corpus: licence check against arXiv metadata, manifest, hash-pinned downloads | IMPLEMENTED |
| Ingestion with page metadata, content-based IDs, duplicate detection | IMPLEMENTED |
| Page-scoped chunking, cached embeddings, versioned Chroma index | IMPLEMENTED |
| Dense search from the command line | IMPLEMENTED |
| Baseline RAG with citations and abstention | PLANNED |
| Hybrid retrieval (BM25 + dense, RRF) and cross-encoder reranking | PLANNED |
| Evaluation: golden question set and BEIR SciFact benchmark | PLANNED |
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
uv run rag ingest            # chunk, embed and index them (needs OPENAI_API_KEY)
uv run rag search "dense retrieval with late interaction"
```

## Development

Requirements: Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # install dependencies from uv.lock
cp .env.example .env         # then add your API keys
uv run ruff check .          # lint
uv run ruff format --check . # formatting
uv run pytest                # tests
```

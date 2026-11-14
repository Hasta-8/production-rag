# Production RAG

A question-answering service over a collection of openly licensed AI/ML papers.
It answers only from those documents and cites the page for every claim.
Retrieval is hybrid (BM25 plus embeddings, fused with Reciprocal Rank Fusion)
with a cross-encoder reranker. A LangGraph agent checks the evidence, rewrites
the query when retrieval is weak, and declines to answer rather than guess.
The pipeline runs behind a FastAPI service with production safeguards, and a
web UI shows each pipeline step next to the answer.

> **Status: early development.** Only the repository setup is done. Everything
> below marked PLANNED is not built yet.

## Capability status

| Capability | Status |
|---|---|
| Repository setup: secrets excluded, locked dependencies, lint and tests in CI | IMPLEMENTED |
| Ingestion with page metadata, stable IDs, duplicate detection | PLANNED |
| Baseline RAG with citations and abstention | PLANNED |
| Hybrid retrieval (BM25 + dense, RRF) and cross-encoder reranking | PLANNED |
| Evaluation: golden question set and BEIR SciFact benchmark | PLANNED |
| LangGraph correction loop with bounded retries and model fallback | PLANNED |
| FastAPI service: security, caching, rate limits, streaming, observability | PLANNED |
| Demo UI: chat, citation viewer, trace panel, compare mode | PLANNED |
| Local demo: Docker, start script, Cloudflare tunnel | PLANNED |

## Development

Requirements: Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # install dependencies from uv.lock
cp .env.example .env         # then add your API keys
uv run ruff check .          # lint
uv run ruff format --check . # formatting
uv run pytest                # tests
```

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

> **Status: in development.** Ingestion, hybrid retrieval with reranking, cited
> answers and the agent loop are built, tested and evaluated. Everything below
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
| Hybrid retrieval (BM25 + dense, RRF) and cross-encoder reranking | IMPLEMENTED |
| BEIR SciFact benchmark (`rag eval beir`) | IMPLEMENTED |
| LangGraph correction loop with bounded retries and model fallback | IMPLEMENTED |
| Held-out question set for changes chosen after seeing results | IMPLEMENTED |
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
reference-list entries do poorly. With hybrid search and reranking (the default
since Phase 3) the same check gives 87% / 96% for prose and 72% / 89% for
reference entries.

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

## Retrieval quality (Phase 3)

The default retriever is now **hybrid + rerank**: BM25 and dense search fused
with Reciprocal Rank Fusion, then the top 20 rescored by a cross-encoder
(ms-marco-MiniLM-L-6-v2, ONNX on the CPU). Every candidate keeps its rank from
each stage, so `rag search` and `rag ask --trace` show why a chunk ranks where it
does. Golden set, v2 evidence labels (every page that states the answer):

| Retrieval mode | Evidence page at rank 1 | in top 3 | MRR | Right paper first |
|---|---|---|---|---|
| dense (Phase 2) | 60% | 73% | 0.704 | 87% |
| BM25 | 67% | 87% | 0.780 | 90% |
| hybrid (RRF) | 70% | 90% | 0.808 | 97% |
| dense + rerank | 80% | 97% | 0.890 | 97% |
| **hybrid + rerank** | **80%** | **100%** | **0.900** | **97%** |

Answers with hybrid + rerank (`eval/results/2026-11-20_p3-hybrid_rerank.json`):

| Metric | Phase 2 dense | Phase 3 hybrid + rerank |
|---|---|---|
| Unanswerable questions declined | 5 / 5 | 5 / 5 (3 by the relevance gate, before the LLM) |
| Answerable questions answered | 87% | 93% |
| Answers citing an evidence page (v2 labels) | 73% | **96%** |
| Expected keywords present | 0.75 | 0.90 |
| Wrong answers built from unrelated passages | 2 | 0 |

The relevance gate now uses the reranker score (threshold 2.0), which reads the
question and passage together and separates off-topic questions far better than
cosine similarity. The two remaining failures are refusals with the evidence in
hand, a generation problem left for the Phase 4 agent loop.

**Public benchmark, BEIR SciFact** (300 queries, 5,183 abstracts):

| Mode | nDCG@10 | Recall@100 | Published (BEIR paper, Table 2) |
|---|---|---|---|
| BM25 (no stemming) | 0.647 | 0.866 | BM25 0.665 |
| dense (bge-small, quantized) | **0.707** | 0.943 | |
| hybrid | 0.688 | **0.965** | |
| hybrid + rerank | 0.687 | 0.965 | BM25 + same cross-encoder 0.688 |

On SciFact, dense alone wins and the web-search-trained reranker adds nothing
(at 15 s/query for 100 candidates). On this project's papers, hybrid + rerank
wins clearly. The best configuration depends on the data, which is why both
benchmarks are run.

**Experiments recorded:** relabelled golden set (v1 single-phrase labels
understated answer quality: 82% vs 96% citing evidence, while the Phase 2
baseline scores 73% under both); a query-time bibliography filter (accurate,
but bibliography chunks almost never reach the top 3 once hybrid + rerank is
used, so it is off by default: `RAG_SKIP_REFERENCES=1`). `rag eval rescore`
re-scores saved answer runs against either label set without rerunning the LLM.

## Agent loop (Phase 4)

`rag ask` runs a LangGraph agent (`--pipeline single` runs the one-pass
pipeline, kept for comparison):

```
retrieve -> grade --relevant--> generate -> check -> answer or decline
              |
              +--too weak--> rewrite the query (LLM) -> retrieve again
```

- **Grade** uses the reranker score (threshold 2.0), not an LLM call: a 1.7B
  grader on a CPU would be slow and unreliable.
- **Rewrite**: when the evidence is weak, a short LLM call turns the question
  into a search query and retrieval runs again. At most 2 corrective steps per
  question; each one is a trace step with its budget.
- **Fallback**: if Qwen3-1.7B produces no first token within 90 s (normally
  35–47 s on this laptop) or fails, Qwen3-0.6B answers and the trace says why.
  The timeout really interrupts llama.cpp mid-prompt, through its abort callback.

**What didn't work, and is off by default.** A "focus" retry, for when the model
declines although good evidence was retrieved, gives it the best sentences of
the top 10 passages instead of 3 whole chunks. It recovered one over-refusal but
made the model answer two trap questions with a fact about a *similar* model:

| Trap question | Wrong answer after the retry | Where the fact came from |
|---|---|---|
| What dropout rate did the original Transformer base model use? | 0.1 | DETR's transformer |
| How many GPUs were used to train Llama 2? | 2048 A100s | the first LLaMA paper |

Requiring the focused sentences to pass the reranker gate fixed the first case,
but that rule was chosen after seeing the result. On held-out questions written
beforehand it failed: the reranker scored those near-miss sentences 4–5, far
above the gate. A relevance score measures topic, not identity. The retry
stays available (`RAG_FOCUS_RETRY=true`) to reproduce the experiments.

**Current results** (agent, re-indexed corpus):

| | Golden set (30 + 5 traps) | Held-out set (5 + 5 traps) |
|---|---|---|
| Answerable questions answered | 28 / 30 | 5 / 5 |
| Trap questions declined | 4 / 5 | 5 / 5 |
| Answers citing an evidence page | 96% | 80% (one answer cites a page that only says "e.g., 40") |
| Median seconds per question | 35.7 | 37.8 |

The golden-set trap now missed (u01) appeared after a text-extraction fix: the
RETRO paper's small-caps name ("R/e.sc /t.sc /r.sc/o.sc") is now repaired, which
made its hyperparameter table readable, and the model attributes RETRO's dropout
rate to the original Transformer. The fix stays: it is correct, and the earlier
5/5 relied partly on garbled text. Attributing facts to the wrong model is the
main open weakness; an entity check before answering is the planned experiment.

## Tracing

Every answer carries a trace of its steps (retrieve, grade, rewrite, focus,
generate, check), with timings, scores, retries and any switch to the fallback
model. `rag ask --trace` prints it, and `--json` includes it. The trace stays on
the machine.

LangSmith is supported but **off by default**, because it would send
questions, passages and answers to an outside service. To turn it on, set
`LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` in `.env`. LangGraph then
traces the agent's nodes automatically.

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

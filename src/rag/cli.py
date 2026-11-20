"""Command-line interface: ``uv run rag <command>``."""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from rag import corpus
from rag.config import get_settings


def _print(msg: str) -> None:
    print(msg, flush=True)


def cmd_corpus_check(args: argparse.Namespace) -> int:
    settings = get_settings()
    candidates = corpus.load_candidates(settings.corpus_candidates)
    existing = corpus.load_manifest(settings.corpus_manifest)
    _print(f"Checking {len(candidates)} candidates on arXiv (about 3 s each)...")
    with corpus.http_client() as client:
        manifest, failures = corpus.check_candidates(
            candidates, client, existing=existing, on_progress=_print
        )
    corpus.save_manifest(manifest, settings.corpus_manifest)
    counts = Counter(p.licence_class.value for p in manifest.papers)
    _print(f"\nWrote {settings.corpus_manifest}: {dict(counts)}")
    for cand, err in failures:
        _print(f"FAILED {cand.arxiv_id}: {err}")
    return 1 if failures else 0


def cmd_corpus_download(args: argparse.Namespace) -> int:
    settings = get_settings()
    manifest = corpus.load_manifest(settings.corpus_manifest)
    if not manifest.papers:
        _print("Manifest is empty. Run `rag corpus check` first.")
        return 1
    with corpus.http_client() as client:
        updated = corpus.download_papers(manifest, settings.pdf_dir, client, on_progress=_print)
    corpus.save_manifest(updated, settings.corpus_manifest)
    n = sum(1 for p in updated.papers if p.can_redistribute)
    _print(f"\n{n} redistributable papers present in {settings.pdf_dir}")
    return 0


EMBED_PRICE_PER_M_TOKENS = {"text-embedding-3-small": 0.02, "text-embedding-3-large": 0.13}


def _open_index(settings):
    from rag.indexing.embeddings import make_embeddings
    from rag.indexing.index import VersionedIndex

    settings.export_hf_token()
    embeddings = make_embeddings(
        settings.embedding_provider,
        settings.embedding_model,
        settings.cache_dir,
        query_prefix=settings.embedding_query_prefix,
        threads=settings.embedding_threads,
        openai_api_key=(
            settings.require_openai_key() if settings.embedding_provider == "openai" else None
        ),
        model_dir=settings.fastembed_dir,
    )
    index = VersionedIndex.open(
        settings.index_dir,
        embeddings,
        settings.embedding_model,
        settings.chunk_size,
        settings.chunk_overlap,
    )
    return index, embeddings


def _manifest_metadata(settings):
    """Map PDF file name -> paper metadata from the corpus manifest."""
    manifest = corpus.load_manifest(settings.corpus_manifest)
    by_file = {
        p.filename: {
            "title": p.title,
            "authors": p.authors,
            "year": p.year,
            "arxiv_id": p.arxiv_id,
            "arxiv_version": p.version,
            "abs_url": p.abs_url,
            "licence_class": p.licence_class.value,
            "topic": p.topic,
        }
        for p in manifest.papers
    }
    return lambda path: by_file.get(path.name, {})


def cmd_ingest(args: argparse.Namespace) -> int:
    from pathlib import Path

    from rag.ingestion.loaders import SUPPORTED_SUFFIXES
    from rag.ingestion.pipeline import ingest_paths

    settings = get_settings()
    roots = [Path(p) for p in args.paths] or [settings.pdf_dir]
    files = sorted(
        f
        for root in roots
        for f in ([root] if root.is_file() else root.rglob("*"))
        if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        _print(f"No supported files found in {', '.join(map(str, roots))}")
        return 1
    index, embeddings = _open_index(settings)
    _print(f"Index {index.info.version} ({settings.embedding_model}); {len(files)} files")

    def report(r):
        detail = r.error if r.error else f"{r.pages} pages, {r.chunks} chunks, {r.tokens} tokens"
        _print(f"{r.status:<9} {r.path.name}: {detail}")

    try:
        results = ingest_paths(files, index, _manifest_metadata(settings), on_result=report)
    except Exception as exc:  # provider errors (quota, auth, network) stop the whole run
        _print(
            f"\nStopped: the embedding provider failed ({type(exc).__name__}: {exc}).\n"
            f"Documents finished before this point are kept; rerun to continue."
        )
        return 2
    new_tokens = sum(r.tokens for r in results if r.status in ("indexed", "replaced"))
    price = EMBED_PRICE_PER_M_TOKENS.get(settings.embedding_model)
    cost = f", estimated embedding cost ${new_tokens * price / 1e6:.4f}" if price else ""
    errors = sum(r.status == "error" for r in results)
    _print(
        f"\nDone: {len(index.info.documents)} documents, {index.info.total_chunks} chunks in "
        f"index. New tokens {new_tokens}{cost}. Embedding cache: {embeddings.hits} hits, "
        f"{embeddings.misses} misses. Errors: {errors}"
    )
    return 1 if errors else 0


def _make_retriever(settings, index, mode: str | None = None):
    from rag.retrieval.bm25 import BM25Index
    from rag.retrieval.hybrid import Retriever

    mode = mode or settings.retrieval_mode
    reranker = None
    if mode.endswith("_rerank"):
        from rag.retrieval.rerank import CrossEncoderReranker

        reranker = CrossEncoderReranker(settings.reranker_model, cache_dir=settings.fastembed_dir)
    return Retriever(
        index,
        BM25Index.from_store(index.store),
        reranker=reranker,
        mode=mode,
        candidates=settings.rerank_candidates,
        rrf_k=settings.rrf_k,
        skip_references=settings.skip_references,
    )


def _fmt(x, spec=".3f"):
    return "-" if x is None else format(x, spec)


def cmd_search(args: argparse.Namespace) -> int:
    settings = get_settings()
    index, _ = _open_index(settings)
    if not index.info.documents:
        _print("The index is empty. Run `rag ingest` first.")
        return 1
    retriever = _make_retriever(settings, index, args.mode)
    _print(f"mode: {retriever.mode}")
    for c in retriever.search(args.query, k=args.k):
        snippet = " ".join(c.text.split())[:150]
        _print(
            f"{c.rank}. {c.title[:60]} — p. {c.page}\n"
            f"   dense #{_fmt(c.dense_rank, 'd')} ({_fmt(c.dense_score)}) | bm25 "
            f"#{_fmt(c.bm25_rank, 'd')} ({_fmt(c.bm25_score, '.1f')}) | fused "
            f"#{_fmt(c.fused_rank, 'd')} | rerank {_fmt(c.rerank_score, '.2f')}\n   {snippet}…"
        )
    return 0


def _make_llm(settings, fallback: bool = False):
    """The main model behind the fallback gateway, or (``--small``) the fallback model alone."""
    from rag.generation.llm import FallbackLLM, LlamaCppClient

    def client(path, timeout=None):
        return LlamaCppClient(
            path,
            n_ctx=settings.llm_n_ctx,
            n_threads=settings.llm_threads,
            first_token_timeout_s=timeout,
        )

    if fallback:
        return client(settings.llm_fallback_model_path)
    if not settings.llm_fallback:
        return client(settings.llm_model_path)
    return FallbackLLM(
        client(settings.llm_model_path, settings.llm_first_token_timeout_s),
        client(settings.llm_fallback_model_path),  # loaded only if it is ever needed
    )


def _make_answer_fn(settings, retriever, llm, pipeline: str | None = None):
    """question -> Answer, for the chosen pipeline ("single" pass or the "agent" loop)."""
    from rag.generation.answer import answer_question

    pipeline = pipeline or settings.pipeline
    answer_settings = _answer_settings(settings)
    if pipeline == "agent":
        from rag.agent import RAGAgent

        agent = RAGAgent(retriever, llm, answer_settings)
        return agent.answer
    if pipeline != "single":
        raise ValueError(f"unknown pipeline {pipeline!r}; choose single or agent")

    def single(question, on_token=None):
        return answer_question(question, retriever, llm, answer_settings, on_token=on_token)

    return single


def _answer_settings(settings):
    from rag.generation.answer import AnswerSettings

    return AnswerSettings(
        retrieve_k=settings.retrieve_k,
        context_k=settings.context_k,
        min_relevance=settings.min_relevance,
        min_rerank_score=settings.min_rerank_score,
        max_tokens=settings.answer_max_tokens,
        max_retries=settings.max_retries,
        focus_retry=settings.focus_retry,
        focus_sentences=settings.focus_sentences,
        focus_gate=settings.focus_gate,
    )


def cmd_ask(args: argparse.Namespace) -> int:
    import json

    settings = get_settings()
    index, _ = _open_index(settings)
    retriever = _make_retriever(settings, index, args.mode)
    llm = _make_llm(settings, fallback=args.small)
    answer_fn = _make_answer_fn(settings, retriever, llm, args.pipeline)
    streamed = []

    def show(token: str) -> None:
        streamed.append(token)
        if not args.json:
            print(token, end="", flush=True)

    if not args.json:
        _print(f"Q: {args.question}\n(retrieving, then generating with {llm.name}…)\n")
    answer = answer_fn(args.question, on_token=show)

    if args.json:
        from dataclasses import asdict

        data = asdict(answer)
        data["trace"] = answer.trace.to_list()
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0

    _print("\n\n--- final answer (after citation checks) ---")
    _print(answer.text)
    if answer.abstained:
        _print(f"\n[abstained: {answer.abstain_reason}]")
    if answer.retries:
        _print(f"[{answer.pipeline}: {answer.retries} corrective step(s); see --trace]")
    gen = answer.trace.steps and [s for s in answer.trace.steps if s.name == "generate"]
    if gen and gen[-1].data.get("fallback_from"):
        _print(f"[fallback model {answer.model}: {gen[-1].data['fallback_reason']}]")
    for s in answer.cited_sources:
        _print(f"  [{s.tag}] {s.title} — p. {s.page} (score {s.score:.3f})")
    if answer.citations_inferred:
        _print("  (citations inferred by word overlap: the model gave none)")
    if args.trace:
        _print("\n--- trace ---")
        for step in answer.trace.steps:
            data = {k: v for k, v in step.data.items() if k != "hits"}
            _print(f"{step.name:<10} {step.ms:>9.1f} ms  {data}")
            if step.name == "retrieve":
                for h in step.data["hits"]:
                    _print(
                        f"           #{h['rank']:<2} {h['title'][:44]} p. {h['page']:<3} "
                        f"dense #{_fmt(h['dense_rank'], 'd')} bm25 #{_fmt(h['bm25_rank'], 'd')} "
                        f"fused #{_fmt(h['fused_rank'], 'd')} "
                        f"rerank {_fmt(h['rerank_score'], '.2f')}"
                    )
        _print(f"total      {answer.trace.total_ms:>9.1f} ms")
    return 0


def cmd_index_info(args: argparse.Namespace) -> int:
    settings = get_settings()
    index, _ = _open_index(settings)
    info = index.info
    _print(
        f"Version {info.version}\n  model {info.embedding_model}, chunk {info.chunk_size}/"
        f"{info.chunk_overlap}\n  documents {len(info.documents)}, chunks {info.total_chunks} "
        f"(vector store holds {index.count()})\n  updated {info.updated_at}"
    )
    return 0


def cmd_eval_phrases(args: argparse.Namespace) -> int:
    from rag.evaluation.known_phrases import run_known_phrase_check, summarize

    settings = get_settings()
    index, _ = _open_index(settings)
    if not index.info.documents:
        _print("The index is empty. Run `rag ingest` first.")
        return 1
    retriever = _make_retriever(settings, index, args.mode)
    _print(f"mode: {retriever.mode}")
    results = run_known_phrase_check(retriever, samples=args.samples, k=args.k, seed=args.seed)
    if args.verbose:
        for r in results:
            if r.rank_of_page != 1:
                got = f"rank {r.rank_of_page}" if r.rank_of_page else f"not in top {args.k}"
                kind = "ref" if r.is_reference else "prose"
                _print(
                    f"MISS [{kind}] ({got}) expected {r.title[:40]} p. {r.expected_page}; top "
                    f"was {r.top_title[:40]} p. {r.top_page}\n   phrase: {r.phrase[:150]}"
                )
    groups = {
        "all": results,
        "prose": [r for r in results if not r.is_reference],
        "references": [r for r in results if r.is_reference],
    }
    _print(
        f"\n{'group':<11} {'n':>4} {'page@1':>7} {f'page@{args.k}':>7} {'MRR':>6} "
        f"{'page|verbatim@1':>16} {'paper@1':>8}"
    )
    for name, group in groups.items():
        if not group:
            continue
        s = summarize(group, args.k)
        _print(
            f"{name:<11} {s['samples']:>4} {s['page_at_1']:>7.0%} "
            f"{s[f'page_at_{args.k}']:>7.0%} {s['mrr']:>6.3f} "
            f"{s['page_or_verbatim_at_1']:>16.0%} {s['paper_at_1']:>8.0%}"
        )
    return 0


def cmd_eval_golden(args: argparse.Namespace) -> int:
    from pathlib import Path

    from rag.evaluation import golden

    settings = get_settings()
    index, _ = _open_index(settings)
    questions = golden.load_golden(Path(args.golden))
    if args.limit:
        questions = questions[: args.limit]
    retriever = _make_retriever(settings, index, args.mode)
    config = {
        "index_version": index.info.version,
        "embedding_model": settings.embedding_model,
        "retrieval": retriever.mode,
        "rerank_candidates": settings.rerank_candidates,
        "reranker": settings.reranker_model if retriever.uses_reranker else None,
        "rrf_k": settings.rrf_k,
        "skip_references": settings.skip_references,
        "golden": f"{args.golden} v{golden.golden_version(Path(args.golden))}",
        "labels": args.labels,
        "retrieve_k": settings.retrieve_k,
        "context_k": settings.context_k,
        "min_relevance": settings.min_relevance,
        "min_rerank_score": settings.min_rerank_score,
        "questions": len(questions),
    }

    thresholds = golden.RERANK_THRESHOLDS if retriever.uses_reranker else golden.DENSE_THRESHOLDS
    rows, summary = golden.evaluate_retrieval(
        retriever, questions, thresholds=thresholds, labels=args.labels
    )
    signal = "reranker logit" if retriever.uses_reranker else "dense cosine"
    _print(f"Retrieval ({len(questions)} questions, {retriever.mode}):")
    for key, value in summary.items():
        if key != "calibration":
            _print(f"  {key:<16} {value:.3f}")
    _print(f"  Abstention threshold calibration (best {signal} score):")
    _print("    threshold  answerable kept  unanswerable rejected")
    for row in summary["calibration"]:
        _print(
            f"    {row['threshold']:.2f}       {row['answerable_kept']:>6.0%}"
            f"           {row['unanswerable_rejected']:>6.0%}"
        )
    if args.verbose:
        for r in rows:
            _print(
                f"    {r.id} best={r.best_score:.3f} page_rank={r.first_hit_rank} "
                f"paper_rank={r.paper_rank}"
            )
    out_dir = Path("eval/results")
    golden.save_results(out_dir, f"{args.label}_retrieval", config, summary, rows)
    if args.retrieval_only:
        return 0

    llm = _make_llm(settings, fallback=args.small)
    pipeline = args.pipeline or settings.pipeline
    config["llm"] = llm.name
    config["max_tokens"] = settings.answer_max_tokens
    config["pipeline"] = pipeline
    if pipeline == "agent":
        config["max_retries"] = settings.max_retries
        config["focus_retry"] = settings.focus_retry
        config["focus_sentences"] = settings.focus_sentences
        config["focus_gate"] = settings.focus_gate
    if settings.llm_fallback and not args.small:
        config["fallback_llm"] = settings.llm_fallback_model_file
        config["first_token_timeout_s"] = settings.llm_first_token_timeout_s
    answer_fn = _make_answer_fn(settings, retriever, llm, pipeline)
    _print(f"\nAnswers with {llm.name}, {pipeline} pipeline (about a minute each)…")

    def show(r):
        status = (
            "declined" if r.abstained else f"kw={r.keyword_recall} cite_ok={r.citation_on_evidence}"
        )
        mark = (
            "ok " if (r.correct_abstention and (r.abstained or r.citation_on_evidence)) else "!! "
        )
        extra = f" retries={r.retries}" if r.retries else ""
        _print(f"  {mark}{r.id} {r.seconds:>6.1f}s {status}{extra}")

    arows, asummary = golden.evaluate_answers(
        retriever,
        llm,
        questions,
        _answer_settings(settings),
        on_row=show,
        labels=args.labels,
        answer_fn=answer_fn,
    )
    _print("\nAnswer summary:")
    for key, value in asummary.items():
        _print(f"  {key:<24} {value}")
    out = golden.save_results(out_dir, args.label, config, {**summary, **asummary}, arows)
    _print(f"\nSaved {out}")
    return 0


def cmd_eval_rescore(args: argparse.Namespace) -> int:
    """Re-score saved answer runs against v1 and v2 evidence labels (no LLM)."""
    import json
    from pathlib import Path

    from rag.evaluation import golden

    settings = get_settings()
    index, _ = _open_index(settings)
    questions = golden.load_golden(Path(args.golden))
    pages = golden.page_texts(index)
    meta = index.store.get(include=["metadatas"])["metadatas"]
    arxiv_by_title = {str(m.get("title", "")): str(m.get("arxiv_id", "")) for m in meta}
    evidence = {lab: golden.resolve_evidence(questions, pages, lab) for lab in golden.LABELS}
    keys = ("answerable_answered", "citation_on_evidence", "keyword_recall_mean")
    _print(f"{'run':<34} {'labels':<6} {'answered':>8} {'cite_ok':>8} {'kw':>6}  flagged")
    for name in args.files:
        data = json.loads(Path(name).read_text())
        for lab in golden.LABELS:
            rows, summary = golden.rescore_answers(
                data["rows"], questions, evidence[lab], arxiv_by_title
            )
            answered, cite, kw = (summary[k] for k in keys)
            flagged = [
                r.id
                for r in rows
                if r.answerable and not r.abstained and not r.citation_on_evidence
            ]
            _print(
                f"{data['label']:<34} {lab:<6} {answered:>8.0%} {cite:>8.0%} {kw:>6.2f}  "
                f"{' '.join(flagged) or '-'}"
            )
    return 0


def cmd_eval_beir(args: argparse.Namespace) -> int:
    import json
    from datetime import UTC, datetime
    from pathlib import Path

    from rag.evaluation import beir
    from rag.indexing.embeddings import make_embeddings

    settings = get_settings()
    settings.export_hf_token()
    folder, sources = beir.download(args.dataset, Path("data/beir"))
    ds = beir.load(folder)
    _print(
        f"{args.dataset}: {len(ds.corpus)} documents, {len(ds.queries)} test queries "
        f"(source: Hugging Face BeIR/{args.dataset})"
    )
    embeddings = make_embeddings(
        settings.embedding_provider,
        settings.embedding_model,
        settings.cache_dir,
        query_prefix=settings.embedding_query_prefix,
        threads=settings.embedding_threads,
        model_dir=settings.fastembed_dir,
    )
    persist = (
        settings.index_dir / f"beir-{args.dataset}" / settings.embedding_model.replace("/", "_")
    )

    def progress(done, total):
        _print(f"  embedded {done}/{total}")

    index, bm25 = beir.build_index(ds, persist, embeddings, on_progress=progress)
    queries = sorted(ds.queries)[: args.limit] if args.limit else None
    results = []
    for mode in args.modes.split(","):
        reranker = None
        if mode.endswith("_rerank"):
            from rag.retrieval.rerank import CrossEncoderReranker

            reranker = CrossEncoderReranker(
                settings.reranker_model, cache_dir=settings.fastembed_dir
            )
        res = beir.evaluate_mode(ds, index, bm25, mode, reranker, queries=queries)
        results.append(res)
        _print(
            f"  {mode:<14} nDCG@10 {res['ndcg@10']:.4f}  Recall@100 {res['recall@100']:.4f}  "
            f"p50 {res['p50_ms']:.0f} ms  p95 {res['p95_ms']:.0f} ms"
        )
    out = Path("eval/results") / f"{datetime.now(UTC):%Y-%m-%d}_beir-{args.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "sources_sha256": sources,
                "documents": len(ds.corpus),
                "embedding_model": settings.embedding_model,
                "reranker": settings.reranker_model,
                "results": results,
            },
            indent=2,
        )
    )
    _print(f"Saved {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag", description="Production RAG command line")
    sub = parser.add_subparsers(dest="command", required=True)

    corpus_p = sub.add_parser("corpus", help="Manage the paper corpus")
    corpus_sub = corpus_p.add_subparsers(dest="corpus_command", required=True)
    corpus_sub.add_parser(
        "check", help="Look up candidates on arXiv and write the manifest"
    ).set_defaults(func=cmd_corpus_check)
    corpus_sub.add_parser(
        "download", help="Download redistributable papers and pin their hashes"
    ).set_defaults(func=cmd_corpus_download)

    ingest_p = sub.add_parser("ingest", help="Ingest files into the index (default: the corpus)")
    ingest_p.add_argument("paths", nargs="*", help="Files or folders (default: corpus PDFs)")
    ingest_p.set_defaults(func=cmd_ingest)

    search_p = sub.add_parser("search", help="Dense search over the index")
    search_p.add_argument("query")
    search_p.add_argument("-k", type=int, default=5)
    search_p.add_argument(
        "--mode",
        help="Retrieval mode (default: settings): dense, bm25, hybrid, dense_rerank, hybrid_rerank",
    )
    search_p.set_defaults(func=cmd_search)

    ask_p = sub.add_parser("ask", help="Answer a question with citations (local LLM)")
    ask_p.add_argument("question")
    ask_p.add_argument("--trace", action="store_true", help="Show every pipeline step")
    ask_p.add_argument("--json", action="store_true", help="Print the full answer as JSON")
    ask_p.add_argument("--small", action="store_true", help="Use the fallback model (0.6B)")
    ask_p.add_argument(
        "--pipeline", choices=("single", "agent"), help="Single pass or the agent loop"
    )
    ask_p.add_argument(
        "--mode",
        help="Retrieval mode (default: settings): dense, bm25, hybrid, dense_rerank, hybrid_rerank",
    )
    ask_p.set_defaults(func=cmd_ask)

    index_p = sub.add_parser("index", help="Index information")
    index_sub = index_p.add_subparsers(dest="index_command", required=True)
    index_sub.add_parser("info", help="Show the active index").set_defaults(func=cmd_index_info)

    eval_p = sub.add_parser("eval", help="Evaluation checks")
    eval_sub = eval_p.add_subparsers(dest="eval_command", required=True)
    phrases_p = eval_sub.add_parser(
        "phrases", help="Known-phrase check: verbatim sentences should return their own page"
    )
    phrases_p.add_argument("--samples", type=int, default=50)
    phrases_p.add_argument("-k", type=int, default=5)
    phrases_p.add_argument("--seed", type=int, default=7)
    phrases_p.add_argument("-v", "--verbose", action="store_true", help="List every miss")
    phrases_p.add_argument(
        "--mode",
        help="Retrieval mode (default: settings): dense, bm25, hybrid, dense_rerank, hybrid_rerank",
    )
    phrases_p.set_defaults(func=cmd_eval_phrases)
    golden_p = eval_sub.add_parser("golden", help="Golden question set: retrieval and answers")
    golden_p.add_argument("--golden", default="eval/golden.yaml")
    golden_p.add_argument("--label", default="run")
    golden_p.add_argument("--retrieval-only", action="store_true", help="Skip the LLM (fast)")
    golden_p.add_argument("--limit", type=int, default=0)
    golden_p.add_argument("--small", action="store_true", help="Use the fallback model")
    golden_p.add_argument("--pipeline", choices=("single", "agent"), help="Answer pipeline")
    golden_p.add_argument(
        "--mode",
        help="Retrieval mode (default: settings): dense, bm25, hybrid, dense_rerank, hybrid_rerank",
    )
    golden_p.add_argument("-v", "--verbose", action="store_true")
    golden_p.add_argument("--labels", choices=("v2", "v1"), default="v2", help="Evidence labels")
    golden_p.set_defaults(func=cmd_eval_golden)
    rescore_p = eval_sub.add_parser(
        "rescore", help="Re-score saved golden answer runs with v1 and v2 labels (no LLM)"
    )
    rescore_p.add_argument("files", nargs="+")
    rescore_p.add_argument("--golden", default="eval/golden.yaml")
    rescore_p.set_defaults(func=cmd_eval_rescore)
    beir_p = eval_sub.add_parser("beir", help="Public benchmark (BEIR): nDCG@10, Recall@100")
    beir_p.add_argument("--dataset", default="scifact")
    beir_p.add_argument("--modes", default="bm25,dense,hybrid,hybrid_rerank")
    beir_p.add_argument("--limit", type=int, default=0, help="Evaluate only the first N queries")
    beir_p.set_defaults(func=cmd_eval_beir)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

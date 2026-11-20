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


def cmd_search(args: argparse.Namespace) -> int:
    from rag.retrieval.dense import dense_search

    settings = get_settings()
    index, _ = _open_index(settings)
    if not index.info.documents:
        _print("The index is empty. Run `rag ingest` first.")
        return 1
    for hit in dense_search(index, args.query, k=args.k):
        snippet = " ".join(hit.text.split())[:160]
        _print(f"{hit.rank}. [{hit.score:.3f}] {hit.title} — p. {hit.page}\n   {snippet}…")
    return 0


def _make_llm(settings, fallback: bool = False):
    from rag.generation.llm import LlamaCppClient

    path = settings.llm_fallback_model_path if fallback else settings.llm_model_path
    return LlamaCppClient(path, n_ctx=settings.llm_n_ctx, n_threads=settings.llm_threads)


def _answer_settings(settings):
    from rag.generation.answer import AnswerSettings

    return AnswerSettings(
        retrieve_k=settings.retrieve_k,
        context_k=settings.context_k,
        min_relevance=settings.min_relevance,
        max_tokens=settings.answer_max_tokens,
    )


def cmd_ask(args: argparse.Namespace) -> int:
    import json

    from rag.generation.answer import answer_question

    settings = get_settings()
    index, _ = _open_index(settings)
    llm = _make_llm(settings, fallback=args.small)
    streamed = []

    def show(token: str) -> None:
        streamed.append(token)
        if not args.json:
            print(token, end="", flush=True)

    if not args.json:
        _print(f"Q: {args.question}\n(retrieving, then generating with {llm.name}…)\n")
    answer = answer_question(args.question, index, llm, _answer_settings(settings), on_token=show)

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
                    _print(f"           {h['score']:.3f}  {h['title'][:50]} p. {h['page']}")
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
    results = run_known_phrase_check(index, samples=args.samples, k=args.k, seed=args.seed)
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
    config = {
        "index_version": index.info.version,
        "embedding_model": settings.embedding_model,
        "retrieval": "dense",
        "retrieve_k": settings.retrieve_k,
        "context_k": settings.context_k,
        "min_relevance": settings.min_relevance,
        "questions": len(questions),
    }

    rows, summary = golden.evaluate_retrieval(index, questions)
    _print(f"Retrieval ({len(questions)} questions, dense):")
    for key, value in summary.items():
        if key != "calibration":
            _print(f"  {key:<16} {value:.3f}")
    _print("  Abstention threshold calibration (best dense score):")
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
    config["llm"] = llm.name
    config["max_tokens"] = settings.answer_max_tokens
    _print(f"\nAnswers with {llm.name} (about a minute each)…")

    def show(r):
        status = (
            "declined" if r.abstained else f"kw={r.keyword_recall} cite_ok={r.citation_on_evidence}"
        )
        mark = (
            "ok " if (r.correct_abstention and (r.abstained or r.citation_on_evidence)) else "!! "
        )
        _print(f"  {mark}{r.id} {r.seconds:>6.1f}s {status}")

    arows, asummary = golden.evaluate_answers(
        index, llm, questions, _answer_settings(settings), on_row=show
    )
    _print("\nAnswer summary:")
    for key, value in asummary.items():
        _print(f"  {key:<24} {value}")
    out = golden.save_results(out_dir, args.label, config, {**summary, **asummary}, arows)
    _print(f"\nSaved {out}")
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
    search_p.set_defaults(func=cmd_search)

    ask_p = sub.add_parser("ask", help="Answer a question with citations (local LLM)")
    ask_p.add_argument("question")
    ask_p.add_argument("--trace", action="store_true", help="Show every pipeline step")
    ask_p.add_argument("--json", action="store_true", help="Print the full answer as JSON")
    ask_p.add_argument("--small", action="store_true", help="Use the fallback model (0.6B)")
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
    phrases_p.set_defaults(func=cmd_eval_phrases)
    golden_p = eval_sub.add_parser("golden", help="Golden question set: retrieval and answers")
    golden_p.add_argument("--golden", default="eval/golden.yaml")
    golden_p.add_argument("--label", default="run")
    golden_p.add_argument("--retrieval-only", action="store_true", help="Skip the LLM (fast)")
    golden_p.add_argument("--limit", type=int, default=0)
    golden_p.add_argument("--small", action="store_true", help="Use the fallback model")
    golden_p.add_argument("-v", "--verbose", action="store_true")
    golden_p.set_defaults(func=cmd_eval_golden)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

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
    from rag.indexing.embeddings import openai_embeddings
    from rag.indexing.index import VersionedIndex

    embeddings = openai_embeddings(
        settings.embedding_model, settings.require_openai_key(), settings.cache_dir
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

    index_p = sub.add_parser("index", help="Index information")
    index_sub = index_p.add_subparsers(dest="index_command", required=True)
    index_sub.add_parser("info", help="Show the active index").set_defaults(func=cmd_index_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

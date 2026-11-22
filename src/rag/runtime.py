"""Building the pipeline from settings: index, retrievers, LLM gateway, answer function.

Shared by the CLI and the API service, so both run exactly the same pipeline.
"""

from __future__ import annotations

from rag import corpus


def open_index(settings):
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


def manifest_metadata(settings):
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


def make_reranker(settings):
    from rag.retrieval.rerank import CrossEncoderReranker

    return CrossEncoderReranker(settings.reranker_model, cache_dir=settings.fastembed_dir)


def make_retriever(settings, index, mode: str | None = None, bm25=None, reranker=None):
    """A retriever for ``mode``. Pass ``bm25``/``reranker`` to share them between modes."""
    from rag.retrieval.bm25 import BM25Index
    from rag.retrieval.hybrid import Retriever

    mode = mode or settings.retrieval_mode
    if mode.endswith("_rerank") and reranker is None:
        reranker = make_reranker(settings)
    return Retriever(
        index,
        bm25 if bm25 is not None else BM25Index.from_store(index.store),
        reranker=reranker if mode.endswith("_rerank") else None,
        mode=mode,
        candidates=settings.rerank_candidates,
        rrf_k=settings.rrf_k,
        skip_references=settings.skip_references,
    )


def make_llm(settings, fallback: bool = False):
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


def make_answer_fn(settings, retriever, llm, pipeline: str | None = None):
    """question -> Answer, for the chosen pipeline ("single" pass or the "agent" loop)."""
    from rag.generation.answer import answer_question

    pipeline = pipeline or settings.pipeline
    opts = answer_settings(settings)
    if pipeline == "agent":
        from rag.agent import RAGAgent

        agent = RAGAgent(retriever, llm, opts)
        return agent.answer
    if pipeline != "single":
        raise ValueError(f"unknown pipeline {pipeline!r}; choose single or agent")

    def single(question, on_token=None, trace=None):
        return answer_question(question, retriever, llm, opts, on_token=on_token, trace=trace)

    return single


def answer_settings(settings):
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
        entity_check=settings.entity_check,
        rewrite_version=settings.rewrite_version,
    )

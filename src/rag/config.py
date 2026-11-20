"""Application settings, loaded once from environment variables and `.env`."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration in one validated place.

    App settings use the ``RAG_`` prefix (for example ``RAG_CHUNK_SIZE``).
    Provider keys use their standard names (``OPENAI_API_KEY``).
    """

    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    hf_token: SecretStr | None = Field(default=None, validation_alias="HF_TOKEN")

    # Corpus
    corpus_candidates: Path = Path("corpus/candidates.yaml")
    corpus_manifest: Path = Path("corpus/manifest.yaml")
    pdf_dir: Path = Path("corpus/pdfs")

    # Shared model store. Set RAG_MODELS_DIR in .env to the machine's shared folder
    # (for example ../ai_models); the default keeps models inside the project.
    models_dir: Path = Path(".cache/models")
    llm_model_file: str = "llm/Qwen3-1.7B-Q8_0.gguf"  # relative to models_dir
    llm_fallback_model_file: str = "llm/Qwen3-0.6B-Q8_0.gguf"
    llm_threads: int = 8  # measured faster than 4 on the 4-core/8-thread laptop CPU
    llm_n_ctx: int = 4096

    # Retrieval (Phase 3)
    retrieval_mode: str = "hybrid_rerank"  # dense | bm25 | hybrid | dense_rerank | hybrid_rerank
    rerank_candidates: int = 20  # pool size fused and reranked before taking the top k
    skip_references: bool = False  # drop bibliography chunks at query time (Phase 3 experiment)
    rrf_k: int = 60
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # Answering
    retrieve_k: int = 10
    context_k: int = 3
    min_relevance: float = 0.60  # dense cosine threshold; see eval/results calibration
    min_rerank_score: float = 2.0  # cross-encoder logit; golden calibration keeps 100% answerable
    answer_max_tokens: int = 150

    # Indexing
    index_dir: Path = Path("indexes")
    cache_dir: Path = Path(".cache")
    embedding_provider: str = "fastembed"  # "fastembed" (local, default) or "openai"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # BGE's recommended instruction for retrieval queries (documents get no prefix).
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    embedding_threads: int | None = None  # None lets ONNX Runtime decide
    chunk_size: int = Field(default=1000, gt=100)
    chunk_overlap: int = Field(default=150, ge=0)

    @property
    def fastembed_dir(self) -> Path:
        return self.models_dir / "fastembed"

    @property
    def llm_model_path(self) -> Path:
        return self.models_dir / self.llm_model_file

    @property
    def llm_fallback_model_path(self) -> Path:
        return self.models_dir / self.llm_fallback_model_file

    def export_hf_token(self) -> None:
        """Hugging Face's downloader reads HF_TOKEN from the process environment."""
        import os

        if self.hf_token is not None and self.hf_token.get_secret_value().strip():
            os.environ.setdefault("HF_TOKEN", self.hf_token.get_secret_value().strip())

    def require_openai_key(self) -> str:
        if self.openai_api_key is None or not self.openai_api_key.get_secret_value().strip():
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        return self.openai_api_key.get_secret_value().strip()


@lru_cache
def get_settings() -> Settings:
    return Settings()

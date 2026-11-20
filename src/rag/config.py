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

    # Corpus
    corpus_candidates: Path = Path("corpus/candidates.yaml")
    corpus_manifest: Path = Path("corpus/manifest.yaml")
    pdf_dir: Path = Path("corpus/pdfs")

    # Indexing
    index_dir: Path = Path("indexes")
    cache_dir: Path = Path(".cache")
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = Field(default=1000, gt=100)
    chunk_overlap: int = Field(default=150, ge=0)

    def require_openai_key(self) -> str:
        if self.openai_api_key is None or not self.openai_api_key.get_secret_value().strip():
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        return self.openai_api_key.get_secret_value().strip()


@lru_cache
def get_settings() -> Settings:
    return Settings()

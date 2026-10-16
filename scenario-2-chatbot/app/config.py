from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Engineering Knowledge Assistant"
    app_env: str = "development"
    log_level: str = "INFO"
    database_url: str = "postgresql+psycopg://chatbot:chatbot@localhost:5432/chatbot"
    openai_api_key: SecretStr | None = None
    openai_chat_model: str = "gpt-4.1-mini"
    openai_vision_model: str = "gpt-4.1-mini"
    openai_embed_model: str = "text-embedding-3-small"
    openai_embed_dimensions: int = 1536
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_enabled: bool = True
    retrieval_candidates: int = Field(default=20, ge=5, le=100)
    rrf_k: int = Field(default=60, ge=1, le=500)
    rerank_top_k: int = Field(default=6, ge=1, le=20)
    min_evidence_score: float = Field(default=0.20, ge=0, le=1)
    chunk_size: int = Field(default=1400, ge=300, le=4000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    data_dir: Path = Path("data")
    max_upload_mb: int = Field(default=50, ge=1, le=500)

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


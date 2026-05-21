"""
config.py — Central configuration for CodeSense backend.

All settings are loaded from environment variables (via .env).
Import this module anywhere in the codebase:

    from config import settings
    print(settings.groq_api_key)

Never import os.getenv() directly in other modules — always go through settings.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, model_validator
from functools import lru_cache
from typing import List


class Settings(BaseSettings):
    """
    All configuration values for CodeSense.
    Values are loaded from environment variables or the .env file.
    Pydantic validates types and raises on startup if required vars are missing.
    """

    # ── Groq ────────────────────────────────────────────────────
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    groq_model: str = Field(
        default="llama-3.1-70b-versatile",
        env="GROQ_MODEL"
    )
    # Rate limit handling — free tier is ~30 req/min
    groq_retry_delay: float = Field(default=2.0, env="GROQ_RETRY_DELAY")
    groq_max_retries: int = Field(default=3, env="GROQ_MAX_RETRIES")

    # ── GitHub ───────────────────────────────────────────────────
    github_token: str = Field(..., env="GITHUB_TOKEN")

    # ── Qdrant ───────────────────────────────────────────────────
    qdrant_host: str = Field(default="localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    qdrant_api_key: str = Field(default="", env="QDRANT_API_KEY")
    qdrant_collection_prefix: str = Field(
        default="codesense",
        env="QDRANT_COLLECTION_PREFIX"
    )

    # ── Redis ─────────────────────────────────────────────────────
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: str = Field(default="", env="REDIS_PASSWORD")
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL"
    )

    # ── Celery ────────────────────────────────────────────────────
    celery_broker_url: str = Field(
        default="redis://localhost:6379/0",
        env="CELERY_BROKER_URL"
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/0",
        env="CELERY_RESULT_BACKEND"
    )

    # ── Embedding Models ─────────────────────────────────────────
    # NOTE: Embeddings are local — no API key needed.
    # CodeBERT downloads ~500MB on first run, cached to transformers_cache.
    codebert_model_name: str = Field(
        default="microsoft/codebert-base",
        env="CODEBERT_MODEL_NAME"
    )
    transformers_cache: str = Field(
        default="./.model_cache",
        env="TRANSFORMERS_CACHE"
    )
    semantic_embedding_dim: int = Field(
        default=768,
        env="SEMANTIC_EMBEDDING_DIM"
    )
    structural_embedding_dim: int = Field(
        default=128,
        env="STRUCTURAL_EMBEDDING_DIM"
    )

    # ── Retrieval Settings ────────────────────────────────────────
    retrieval_confidence_threshold: float = Field(
        default=0.65,
        env="RETRIEVAL_CONFIDENCE_THRESHOLD"
    )
    retrieval_top_k: int = Field(default=20, env="RETRIEVAL_TOP_K")
    reranker_top_n: int = Field(default=5, env="RERANKER_TOP_N")

    # Hybrid weights — must sum to 1.0
    # Default comes from Day 8 experiment: 0.7/0.3 gave best Precision@5
    hybrid_semantic_weight: float = Field(
        default=0.7,
        env="HYBRID_SEMANTIC_WEIGHT"
    )
    hybrid_structural_weight: float = Field(
        default=0.3,
        env="HYBRID_STRUCTURAL_WEIGHT"
    )

    # ── Re-ranker ─────────────────────────────────────────────────
    reranker_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        env="RERANKER_MODEL_NAME"
    )

    # ── Ingestion Settings ────────────────────────────────────────
    repo_clone_dir: str = Field(
        default="./cloned_repos",
        env="REPO_CLONE_DIR"
    )
    ingestion_warn_threshold: int = Field(
        default=50_000,
        env="INGESTION_WARN_THRESHOLD"
    )
    ingestion_max_files: int = Field(
        default=100_000,
        env="INGESTION_MAX_FILES"
    )
    ingestion_skip_extensions: List[str] = Field(
        default=[".min.js", ".min.css", ".lock", ".sum", ".mod"],
        env="INGESTION_SKIP_EXTENSIONS"
    )

    # Supported languages → file extensions mapping
    # Not an env var — change here if adding language support
    supported_extensions: dict = {
        "python":     [".py"],
        "javascript": [".js", ".mjs", ".cjs"],
        "typescript": [".ts", ".tsx"],
        "java":       [".java"],
    }

    # ── FastAPI Backend ───────────────────────────────────────────
    backend_host: str = Field(default="0.0.0.0", env="BACKEND_HOST")
    backend_port: int = Field(default=8000, env="BACKEND_PORT")
    backend_reload: bool = Field(default=True, env="BACKEND_RELOAD")
    rate_limit_per_minute: int = Field(
        default=30,
        env="RATE_LIMIT_PER_MINUTE"
    )

    # ── Logging ───────────────────────────────────────────────────
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="./logs/codesense.log", env="LOG_FILE")

    # ── Environment ───────────────────────────────────────────────
    environment: str = Field(default="development", env="ENVIRONMENT")

    # ── Validators ────────────────────────────────────────────────

    @model_validator(mode="after")
    def validate_hybrid_weights(self) -> "Settings":
        """Ensure hybrid weights sum to 1.0 on startup."""
        total = self.hybrid_semantic_weight + self.hybrid_structural_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"HYBRID_SEMANTIC_WEIGHT + HYBRID_STRUCTURAL_WEIGHT must equal 1.0, "
                f"got {total:.4f}"
            )
        return self

    # ── Derived Properties ────────────────────────────────────────

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    def qdrant_collection_name(self, owner: str, repo: str) -> str:
        """
        Generate a sanitized Qdrant collection name for a given repo.

        Example:
            qdrant_collection_name("tiangolo", "fastapi")
            → "codesense_tiangolo_fastapi"
        """
        safe_owner = owner.lower().replace("-", "_").replace(".", "_")
        safe_repo  = repo.lower().replace("-", "_").replace(".", "_")
        return f"{self.qdrant_collection_prefix}_{safe_owner}_{safe_repo}"

    def is_supported_file(self, file_path: str) -> bool:
        """Returns True if the file should be parsed (supported extension, not skipped)."""
        from pathlib import Path
        ext = Path(file_path).suffix.lower()
        if ext in self.ingestion_skip_extensions:
            return False
        all_supported = [
            e for exts in self.supported_extensions.values() for e in exts
        ]
        return ext in all_supported

    def language_for_file(self, file_path: str) -> str | None:
        """Returns the language name for a file path, or None if unsupported."""
        from pathlib import Path
        ext = Path(file_path).suffix.lower()
        for lang, exts in self.supported_extensions.items():
            if ext in exts:
                return lang
        return None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_list_separator = ","   # Allows INGESTION_SKIP_EXTENSIONS=.min.js,.lock


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached singleton Settings instance.

    Use in FastAPI dependency injection:
        from config import get_settings
        from fastapi import Depends

        def my_route(settings: Settings = Depends(get_settings)):
            ...

    Or import directly for non-FastAPI modules:
        from config import settings
    """
    return Settings()


# Module-level singleton for direct imports
settings = get_settings()
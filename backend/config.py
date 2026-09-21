"""
config.py — Central configuration for CodeSense backend.

All settings are loaded from environment variables (via .env).
Import this module anywhere in the codebase:

    from config import settings
    print(settings.groq_api_key)

Never import os.getenv() directly in other modules — always go through settings.
"""

from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import Field, model_validator
from functools import lru_cache
from typing import List

# .env lives at the project root (one level above backend/), but this module
# is imported from many different working directories (uvicorn run from
# backend/, pytest run from repo root, Celery workers, etc.). Resolving the
# path relative to this file — not the process's CWD — means it's found
# consistently no matter where the app is launched from.
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    """
    All configuration values for CodeSense.
    Values are loaded from environment variables or the .env file.
    Pydantic validates types and raises on startup if required vars are missing.
    """

    # ── Groq ────────────────────────────────────────────────────
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    groq_model: str = Field(
        # llama-3.1-70b-versatile (the model this project was originally
        # benchmarked against, see Architecture-notes.md §3) was decommissioned
        # by Groq after this project's initial development. openai/gpt-oss-120b
        # is Groq's current large general-purpose model as of testing this
        # against a live account — verify against
        # https://console.groq.com/docs/models if this starts erroring with
        # "model_decommissioned" again, Groq's catalog changes over time.
        default="openai/gpt-oss-120b",
        env="GROQ_MODEL"
    )
    # Groq exposes an OpenAI-compatible chat completions endpoint, so the
    # `openai` SDK client can be reused as-is by pointing base_url here
    # instead of at api.openai.com (see Architecture-notes.md §3). This is
    # the ONLY thing that makes a Groq call different from an OpenAI call.
    groq_base_url: str = Field(
        default="https://api.groq.com/openai/v1",
        env="GROQ_BASE_URL"
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
    # Field name kept as "codebert_model_name" for .env backward-compat even
    # though the default below is no longer CodeBERT — see the rationale in
    # embeddings/code_embedder.py's module docstring: CodeBERT's raw
    # mean-pooled embeddings measured as near-random for retrieval (all
    # candidates for a query landed in a flat 0.94-0.965 cosine band,
    # including irrelevant chunks) on this project's real indexed repos.
    # BAAI/bge-small-en-v1.5 measured 56% Recall@5 on the same corpora
    # (up from near-zero) in the evaluation session that made this change —
    # re-run evaluation/run_eval.py before changing this again, don't swap
    # on spot checks.
    # bge-small downloads ~130MB on first run, cached to transformers_cache.
    codebert_model_name: str = Field(
        default="BAAI/bge-small-en-v1.5",
        env="CODEBERT_MODEL_NAME"
    )
    transformers_cache: str = Field(
        default="./.model_cache",
        env="TRANSFORMERS_CACHE"
    )
    # bge-small-en-v1.5 is 384-dim (CLS-pooled), not 768 like CodeBERT.
    # embeddings/code_embedder.py.get_embedding_dim() reads this from the
    # loaded model itself rather than trusting this constant, but Qdrant
    # collections still need re-creating (not just re-upserting) if you
    # change to a model with a different dimension — see qdrant_client.py.
    semantic_embedding_dim: int = Field(
        default=384,
        env="SEMANTIC_EMBEDDING_DIM"
    )
    structural_embedding_dim: int = Field(
        default=128,
        env="STRUCTURAL_EMBEDDING_DIM"
    )

    # ── Retrieval Settings ────────────────────────────────────────
    # 0.65 was calibrated for microsoft/codebert-base, whose raw mean-pooled
    # cosine scores were anisotropic (nearly everything landed in a flat
    # 0.94-0.965 band regardless of relevance — see code_embedder.py). That
    # made 0.65 an effectively meaningless gate for that model, but it is a
    # HARD gate for bge-small-en-v1.5's genuinely discriminative scores:
    # measured directly against this project's real indexed repos, a clearly
    # irrelevant control query ("what's the weather on Mars") topped out at
    # 0.43 cosine similarity, while confirmed-correct answers for real eval
    # queries scored as low as 0.56-0.61 — i.e. a 0.65 floor was silently
    # discarding correct chunks before the reranker ever saw them, entirely
    # independent of reranker quality. 0.5 keeps a wide safety margin above
    # the measured irrelevant-query ceiling (0.43) while admitting those
    # correct-but-not-top-cosine matches. Re-measure before changing this
    # again if the embedding model changes (see code_embedder.py docstring).
    retrieval_confidence_threshold: float = Field(
        default=0.5,
        env="RETRIEVAL_CONFIDENCE_THRESHOLD"
    )
    # Raised from 20: with the same real queries, some correct answers
    # ranked in the 30s-50s by raw cosine similarity alone (bge-small isn't
    # code-specialized) — reranking can only recover a correct answer if
    # it's actually in the candidate pool handed to it. 30 is a measured
    # middle ground, not the ceiling — some real queries needed 50+ to
    # include the correct chunk; going that high wasn't pursued further
    # since it stops paying off at larger repo scale (every query reranks
    # a much bigger slice of the corpus).
    retrieval_top_k: int = Field(default=30, env="RETRIEVAL_TOP_K")
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
    # pydantic-settings tries to JSON-decode any List[...]-typed field read
    # from a .env value, but .env stores this as a plain comma-separated
    # string (INGESTION_SKIP_EXTENSIONS=.min.js,.min.css,...), which isn't
    # valid JSON — that decode happens before field_validators ever run, so
    # it can't be fixed with a validator. Instead the env-backed field is a
    # plain str, and `ingestion_skip_extensions` below exposes it as a list.
    ingestion_skip_extensions_csv: str = Field(
        default=".min.js,.min.css,.lock,.sum,.mod",
        # Field(env=...) is pydantic v1 syntax and has no effect in pydantic
        # v2 / pydantic-settings — env-var matching there is by uppercased
        # field name, or (since the field name itself was renamed to
        # _csv to dodge the List[...] JSON-decode issue above) an explicit
        # validation_alias, which IS respected.
        validation_alias="INGESTION_SKIP_EXTENSIONS",
    )

    @property
    def ingestion_skip_extensions(self) -> List[str]:
        return [ext.strip() for ext in self.ingestion_skip_extensions_csv.split(",") if ext.strip()]

    # Supported languages → file extensions mapping
    # Not an env var — change here if adding language support
    supported_extensions: dict = {
        "python":     [".py"],
        "javascript": [".js", ".mjs", ".cjs"],
        "typescript": [".ts", ".tsx"],
        "java":       [".java"],
        "go":         [".go"],
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
        env_file = str(_ENV_FILE)
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
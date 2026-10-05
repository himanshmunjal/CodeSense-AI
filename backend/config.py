"""
config.py — Central configuration for CodeSense backend.

All settings are loaded from environment variables (via .env).
Import this module anywhere in the codebase:

    from config import settings
    print(settings.groq_api_key)

Never import os.getenv() directly in other modules — always go through settings.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
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
    groq_api_key: str = Field(...)
    groq_model: str = Field(
        # llama-3.1-70b-versatile (the model this project was originally
        # benchmarked against, see Architecture-notes.md §3) was decommissioned
        # by Groq after this project's initial development. openai/gpt-oss-120b
        # is Groq's current large general-purpose model as of testing this
        # against a live account — verify against
        # https://console.groq.com/docs/models if this starts erroring with
        # "model_decommissioned" again, Groq's catalog changes over time.
        default="openai/gpt-oss-120b"
    )
    # Groq exposes an OpenAI-compatible chat completions endpoint, so the
    # `openai` SDK client can be reused as-is by pointing base_url here
    # instead of at api.openai.com (see Architecture-notes.md §3). This is
    # the ONLY thing that makes a Groq call different from an OpenAI call.
    groq_base_url: str = Field(
        default="https://api.groq.com/openai/v1"
    )
    # Rate limit handling — free tier is ~30 req/min
    groq_retry_delay: float = Field(default=2.0)
    groq_max_retries: int = Field(default=3)

    # ── GitHub ───────────────────────────────────────────────────
    github_token: str = Field(...)

    # ── Qdrant ───────────────────────────────────────────────────
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_api_key: str = Field(default="")
    qdrant_collection_prefix: str = Field(
        default="codesense"
    )

    # ── Redis ─────────────────────────────────────────────────────
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    redis_password: str = Field(default="")
    redis_url: str = Field(
        default="redis://localhost:6379/0"
    )

    # ── Celery ────────────────────────────────────────────────────
    celery_broker_url: str = Field(
        default="redis://localhost:6379/0"
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/0"
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
        default="BAAI/bge-small-en-v1.5"
    )
    transformers_cache: str = Field(
        default="./.model_cache"
    )
    # bge-small-en-v1.5 is 384-dim (CLS-pooled), not 768 like CodeBERT.
    # embeddings/code_embedder.py.get_embedding_dim() reads this from the
    # loaded model itself rather than trusting this constant, but Qdrant
    # collections still need re-creating (not just re-upserting) if you
    # change to a model with a different dimension — see qdrant_client.py.
    semantic_embedding_dim: int = Field(
        default=384
    )
    structural_embedding_dim: int = Field(
        default=128
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
        default=0.5
    )
    # Raised from 20: with the same real queries, some correct answers
    # ranked in the 30s-50s by raw cosine similarity alone (bge-small isn't
    # code-specialized) — reranking can only recover a correct answer if
    # it's actually in the candidate pool handed to it. 30 is a measured
    # middle ground, not the ceiling — some real queries needed 50+ to
    # include the correct chunk; going that high wasn't pursued further
    # since it stops paying off at larger repo scale (every query reranks
    # a much bigger slice of the corpus).
    retrieval_top_k: int = Field(default=30)
    reranker_top_n: int = Field(default=5)

    # Hybrid weights — must sum to 1.0
    # Default comes from Day 8 experiment: 0.7/0.3 gave best Precision@5
    hybrid_semantic_weight: float = Field(
        default=0.7
    )
    hybrid_structural_weight: float = Field(
        default=0.3
    )

    # ── Re-ranker ─────────────────────────────────────────────────
    reranker_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # ── Ingestion Settings ────────────────────────────────────────
    repo_clone_dir: str = Field(
        default="./cloned_repos"
    )
    ingestion_warn_threshold: int = Field(
        default=50_000
    )
    ingestion_max_files: int = Field(
        default=100_000
    )
    # pydantic-settings tries to JSON-decode any List[...]-typed field read
    # from a .env value, but .env stores this as a plain comma-separated
    # string (INGESTION_SKIP_EXTENSIONS=.min.js,.min.css,...), which isn't
    # valid JSON — that decode happens before field_validators ever run, so
    # it can't be fixed with a validator. Instead the env-backed field is a
    # plain str, and `ingestion_skip_extensions` below exposes it as a list.
    ingestion_skip_extensions_csv: str = Field(
        default=".min.js,.min.css,.lock,.sum,.mod",
        # The field name was renamed to _csv to dodge the List[...]
        # JSON-decode issue above, so the env var is mapped explicitly.
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
    backend_host: str = Field(default="0.0.0.0")
    backend_port: int = Field(default=8000)
    backend_reload: bool = Field(default=True)
    rate_limit_per_minute: int = Field(
        default=30
    )
    # Comma-separated browser origins allowed to call the API. Plain string
    # (not List[str]) for the same .env JSON-decoding reason as
    # INGESTION_SKIP_EXTENSIONS above. "*" allows any origin.
    cors_origins_csv: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        validation_alias="CORS_ORIGINS",
    )
    # Only trust X-Forwarded-For for rate limiting when the API sits behind a
    # reverse proxy that overwrites it. Otherwise any client can send a fresh
    # fake IP on every request and never be rate-limited.
    trust_proxy_headers: bool = Field(default=False)

    # ── Logging ───────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="./logs/codesense.log")

    # ── Environment ───────────────────────────────────────────────
    environment: str = Field(default="development")

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
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.cors_origins_csv.split(",") if o.strip()]

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

    # Env vars match field names case-insensitively (GROQ_API_KEY →
    # groq_api_key); fields that need a different env name use
    # validation_alias. extra="ignore" lets .env hold keys used only by
    # docker-compose or other tools without failing validation.
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )


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
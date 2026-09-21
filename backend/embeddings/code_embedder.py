"""
code_embedder.py
─────────────────
PURPOSE
-------
Generates dense vector embeddings for code chunks using a sentence-embedding
model loaded via the `sentence-transformers` library.

WHY sentence-transformers INSTEAD OF HAND-ROLLED AutoModel + POOLING
----------------------------------------------------------------------
This module used to load `microsoft/codebert-base` directly via
transformers' AutoModel/AutoTokenizer and manually mean-pool + L2-normalize
the last hidden state. That was a real, measured problem for two reasons:

  1. CodeBERT is NOT contrastively trained (no "pull similar pairs together,
     push dissimilar pairs apart" objective) — it's a masked-language-model /
     replaced-token-detection encoder. Mean-pooling its raw hidden states for
     cosine similarity produces a well-documented "anisotropic" embedding
     space: nearly everything ends up in a narrow high-similarity band
     regardless of actual relevance. Measured on this project's own indexed
     repos: 20 candidates for one query all scored between 0.94–0.965 cosine
     similarity — including completely unrelated code — making retrieval
     recall close to random.
  2. Hand-rolling pooling hard-codes an assumption (mean pooling, 768-dim)
     that is specific to CodeBERT. Swapping to any other model means
     re-deriving its correct pooling strategy (CLS-token vs. mean-pooling)
     and dimension by hand — get it wrong and embeddings are silently
     corrupted with no error. The current default, `BAAI/bge-small-en-v1.5`,
     actually uses CLS-token pooling at 384 dimensions — different on both
     counts from CodeBERT's mean-pooling at 768.

`sentence_transformers.SentenceTransformer` reads a model's own pooling
config (its `1_Pooling/config.json`) and applies the pooling strategy that
model was actually trained with, automatically, for whatever model is
configured — eliminating this entire bug class for any future model change.

WHY BAAI/bge-small-en-v1.5 AS THE DEFAULT
-------------------------------------------
Measured (not assumed) on this project's real indexed corpora (Go, JS/React,
Java — 687 chunks across 3 repos) against a 16-query hand-labeled eval set:
Recall@5 improved from near-zero (CodeBERT's flat similarity band meant the
correct chunk rarely made the top 20 candidates at all) to 56%. bge-small
is a small (384-dim, ~130MB), fast, strongly contrastively-trained
general-purpose retrieval model — not code-specific, but empirically it
out-performed a CodeSearchNet-tuned code model on this project's own data
(see git history / conversation notes from the evaluation session).
Re-run `evaluation/run_eval.py` against a candidate before changing this
default again — do not swap based on spot checks (see that mistake in the
same evaluation session's history).

USED BY
-------
- indexing/qdrant_client.py        receives embeddings to upsert as vectors
- retrieval/semantic_retriever.py  calls embed_query() at query time
- tasks/celery_worker.py           calls embed_batch() during ingestion
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from config import settings

logger = logging.getLogger(__name__)

# ── Model Constants ────────────────────────────────────────────────────────────

# The HuggingFace model identifier for the default embedder. Overridable via
# settings.codebert_model_name / CODEBERT_MODEL_NAME env var (name kept for
# .env backward-compatibility even though the default is no longer CodeBERT).
DEFAULT_MODEL_ID = settings.codebert_model_name

# Maximum input length in tokens. Chunks longer than this are truncated.
# The chunking strategy in tree_sitter_parser.py is designed to keep
# functions under this limit, but we enforce it here as a safety net.
MAX_TOKEN_LENGTH = 512

# Default batch size for inference.  Higher = faster on GPU, but risks OOM.
# 16 is a safe default for CPU/MPS with 512-token inputs.
DEFAULT_BATCH_SIZE = 16


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EmbeddingResult:
    """
    The output of embedding a single code chunk.

    Attributes
    ----------
    chunk_id : str
        Unique identifier for this chunk (matches the Qdrant point ID).
        Format: "<repo>/<file_path>:<start_line>-<end_line>"
    embedding : list[float]
        The embedding vector as a plain Python list (dimension depends on
        the loaded model — see CodeEmbedder.get_embedding_dim()).
        Stored as list (not np.ndarray) because Qdrant's client expects
        a list and JSON serialization is simpler.
    model_id : str
        The model that produced this embedding.  Stored so that if the
        model is ever changed, old embeddings can be identified and
        invalidated in the cache.
    token_count : int
        Number of tokens in the input after tokenization.  Useful for
        monitoring truncation rates and estimating API costs.
    truncated : bool
        True if the input exceeded MAX_TOKEN_LENGTH and was truncated.
        Tracked so we can log truncation rates and alert if chunking
        produces too many oversized inputs.
    inference_time_ms : float
        Wall-clock time for this embedding in milliseconds.
        Used by evaluation/latency_benchmark.py to track p50/p90/p99.
    """
    chunk_id: str
    embedding: list[float]
    model_id: str
    token_count: int
    truncated: bool
    inference_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a plain dictionary for cache storage.

        The embedding vector is stored as a list so it can be written
        to JSON (numpy arrays are not JSON-serializable by default).
        """
        return {
            "chunk_id": self.chunk_id,
            "embedding": self.embedding,
            "model_id": self.model_id,
            "token_count": self.token_count,
            "truncated": self.truncated,
            "inference_time_ms": self.inference_time_ms,
        }


@dataclass
class BatchEmbeddingResult:
    """
    Aggregated result of embedding a batch of code chunks.

    Attributes
    ----------
    results : list[EmbeddingResult]
        Individual embedding results, one per input chunk.
    total_time_ms : float
        Total wall-clock time for the entire batch.
    truncation_count : int
        Number of chunks that were truncated during tokenization.
    mean_token_count : float
        Average token count across the batch — useful for monitoring
        whether chunk sizes are staying within the model's limits.
    """
    results: list[EmbeddingResult]
    total_time_ms: float
    truncation_count: int
    mean_token_count: float

    @property
    def embeddings_only(self) -> list[list[float]]:
        """Return just the embedding vectors, in input order."""
        return [r.embedding for r in self.results]


# ─────────────────────────────────────────────────────────────────────────────
# Core Embedder
# ─────────────────────────────────────────────────────────────────────────────

class CodeEmbedder:
    """
    Wraps a sentence-transformers embedding model to produce semantic
    embeddings for code chunks.

    The class handles:
      • Lazy model loading (model is loaded on first use, not at import time)
      • Device selection (CUDA → MPS → CPU, in that priority order — handled
        by sentence-transformers itself)
      • Tokenization, pooling, and L2 normalization — all per the loaded
        model's own config, not hard-coded (see module docstring for why
        that matters)
      • Batched inference for throughput

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier. Defaults to settings.codebert_model_name
        (BAAI/bge-small-en-v1.5 by default — see module docstring for why).
    device : str | None
        PyTorch device string ("cuda", "mps", "cpu").  If None, the
        best available device is selected automatically.
    batch_size : int
        Number of chunks to embed in a single forward pass.
    cache_dir : str | Path | None
        Directory where HuggingFace downloads and caches model weights.
        Defaults to the HuggingFace default (~/.cache/huggingface/).
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.model_id = model_id
        self.batch_size = batch_size
        self.cache_dir = str(cache_dir) if cache_dir else None
        self.device = device  # None lets SentenceTransformer auto-select CUDA/MPS/CPU

        # Lazy-loaded — set to None until _load_model() is called.
        self._model: SentenceTransformer | None = None

    # ── Model Loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """
        Download and load the embedding model via sentence-transformers.

        Called lazily on the first embed() call.  Lazy loading is used
        because the FastAPI app imports this module at startup, but model
        loading should not block the app from starting up — it should only
        happen when the first ingestion job runs.

        SentenceTransformer reads the model's own pooling config and applies
        it automatically (see module docstring for why this matters) — no
        manual mean-pooling / dimension bookkeeping needed here.

        Side effects:
          - Sets self._model.
          - Moves the model to the selected device (auto if self.device is None).
          - Sets the model to eval mode (disables dropout).
        """
        if self._model is not None:
            return  # Already loaded — idempotent.

        logger.info("Loading embedding model '%s'...", self.model_id)
        load_start = time.time()

        self._model = SentenceTransformer(
            self.model_id,
            device=self.device,
            cache_folder=self.cache_dir,
        )
        self._model.eval()
        self.device = str(self._model.device)  # record what was actually selected

        load_time = time.time() - load_start
        logger.info("Embedding model loaded in %.1fs on device '%s'", load_time, self.device)

    # ── Single Chunk Embedding ─────────────────────────────────────────────────

    def embed(self, code: str, chunk_id: str) -> EmbeddingResult:
        """
        Embed a single code chunk and return its vector.

        Prefer embed_batch() for multiple chunks — this method calls
        embed_batch() internally and is provided for convenience when
        embedding one chunk at a time (e.g. at query time).

        Parameters
        ----------
        code : str
            Raw source code text of the chunk to embed.
        chunk_id : str
            Unique identifier for this chunk.

        Returns
        -------
        EmbeddingResult
            The embedding vector and associated metadata.
        """
        results = self.embed_batch([(code, chunk_id)])
        return results.results[0]

    # ── Batch Embedding ───────────────────────────────────────────────────────

    def embed_batch(
        self,
        chunks: list[tuple[str, str]],
    ) -> BatchEmbeddingResult:
        """
        Embed a batch of code chunks in a single forward pass (or multiple
        passes if the batch exceeds self.batch_size).

        This is the primary method called by the ingestion pipeline.
        Batching is critical for GPU utilization — a single forward pass
        on 16 chunks is much faster than 16 separate forward passes.

        Parameters
        ----------
        chunks : list[tuple[str, str]]
            Each tuple is (code_text, chunk_id).
            code_text : the raw source code to embed.
            chunk_id  : unique identifier stored in the result.

        Returns
        -------
        BatchEmbeddingResult
            All embedding results plus aggregate batch statistics.
        """
        self._load_model()  # No-op if already loaded.

        batch_start = time.time()
        all_results: list[EmbeddingResult] = []

        # Process in sub-batches of self.batch_size to avoid OOM on GPU.
        for batch_start_idx in range(0, len(chunks), self.batch_size):
            sub_batch = chunks[batch_start_idx: batch_start_idx + self.batch_size]
            sub_results = self._embed_sub_batch(sub_batch)
            all_results.extend(sub_results)

        total_time_ms = (time.time() - batch_start) * 1000
        truncation_count = sum(1 for r in all_results if r.truncated)
        mean_tokens = (
            sum(r.token_count for r in all_results) / len(all_results)
            if all_results else 0.0
        )

        if truncation_count > 0:
            logger.warning(
                "%d/%d chunks were truncated to %d tokens. "
                "Consider adjusting chunk size in tree_sitter_parser.py.",
                truncation_count, len(chunks), MAX_TOKEN_LENGTH,
            )

        logger.info(
            "Embedded %d chunks in %.0fms (mean tokens=%.0f, truncated=%d)",
            len(all_results), total_time_ms, mean_tokens, truncation_count,
        )

        return BatchEmbeddingResult(
            results=all_results,
            total_time_ms=total_time_ms,
            truncation_count=truncation_count,
            mean_token_count=mean_tokens,
        )

    def _embed_sub_batch(
        self,
        sub_batch: list[tuple[str, str]],
    ) -> list[EmbeddingResult]:
        """
        Run a single encode() pass through the embedding model for one
        sub-batch, using the model's own tokenizer, pooling, and
        normalization — see the module docstring for why this replaced a
        hand-rolled mean-pooling implementation.

        Token-count / truncation stats are computed with a lightweight
        separate tokenizer call purely for observability (the same numbers
        this module has always logged) — encode() itself doesn't expose them.

        Parameters
        ----------
        sub_batch : list[tuple[str, str]]
            (code_text, chunk_id) pairs for this sub-batch.

        Returns
        -------
        list[EmbeddingResult]
            One result per input, in the same order.
        """
        texts = [code for code, _ in sub_batch]
        chunk_ids = [cid for _, cid in sub_batch]

        inference_start = time.time()

        # Token-count / truncation stats only — not used for the embedding
        # itself, which encode() tokenizes and truncates on its own.
        tokenizer = self._model.tokenizer
        encoded = tokenizer(texts, truncation=False, return_attention_mask=True)
        token_counts = [len(ids) for ids in encoded["input_ids"]]
        truncated_flags = [tc >= MAX_TOKEN_LENGTH for tc in token_counts]

        # normalize_embeddings=True gives unit-length vectors so Qdrant's
        # cosine similarity (dot product on normalized vectors) is correct.
        embeddings_np: np.ndarray = self._model.encode(
            texts,
            batch_size=len(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        inference_time_ms = (time.time() - inference_start) * 1000
        per_chunk_time = inference_time_ms / len(sub_batch)

        results = []
        for i, (chunk_id, token_count, truncated) in enumerate(
            zip(chunk_ids, token_counts, truncated_flags)
        ):
            results.append(EmbeddingResult(
                chunk_id=chunk_id,
                embedding=embeddings_np[i].tolist(),
                model_id=self.model_id,
                token_count=int(token_count),
                truncated=truncated,
                inference_time_ms=per_chunk_time,
            ))

        return results

    # ── Query Embedding ───────────────────────────────────────────────────────

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a natural language or code query for similarity search.

        This method is used at retrieval time (not ingestion time) to
        convert a user's query into the same vector space as the indexed
        code chunks.

        The query is prefixed with BAAI/bge's documented query instruction
        ("Represent this sentence for searching relevant passages: ") — bge
        models are trained asymmetrically: this prefix is applied to QUERY
        text only, never to the indexed passages/chunks themselves, and
        measurably improves retrieval for exactly this asymmetric-search use
        case. NOTE: this prefix is specific to the bge model family — if
        codebert_model_name is changed to a different model, check whether
        that model has its own required query instruction (most
        sentence-transformers model cards document this under "Usage").

        Parameters
        ----------
        query : str
            The user's query, e.g. "function that validates JWT tokens".

        Returns
        -------
        list[float]
            Normalized embedding vector (dimension depends on the loaded
            model — see get_embedding_dim()).
        """
        prefixed_query = f"Represent this sentence for searching relevant passages: {query.strip()}"
        result = self.embed(prefixed_query, chunk_id="__query__")
        return result.embedding

    # ── Utility ───────────────────────────────────────────────────────────────

    def warmup(self) -> None:
        """
        Run a dummy forward pass to warm up the model before real inference.

        On first inference, CUDA kernels must be compiled (JIT) which adds
        2–5 seconds of latency.  By running a warmup pass at startup, the
        first real query gets a pre-warmed GPU.

        Called by tasks/celery_worker.py after the worker starts.
        """
        logger.info("Running embedding model warmup pass...")
        self.embed("def warmup(): pass", chunk_id="__warmup__")
        logger.info("Embedding model warmup complete")

    def get_embedding_dim(self) -> int:
        """
        Return the output embedding dimensionality.

        Used by indexing/qdrant_client.py when creating a Qdrant collection
        so the vector dimension is always in sync with the actual model output.

        Returns
        -------
        int
            Dimensionality of the embedding vectors for the loaded model
            (e.g. 384 for bge-small-en-v1.5, 768 for codebert-base) — read
            from the model itself, not hard-coded, so this stays correct
            across model swaps without a matching code change.
        """
        self._load_model()
        return self._model.get_sentence_embedding_dimension()

    def unload_model(self) -> None:
        """
        Release the model from GPU/CPU memory.

        Call this when switching to a different embedding model to free
        resources. The model will be re-loaded lazily on the next embed()
        call.
        """
        if self._model is not None:
            import torch
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Embedding model unloaded from memory")

    def __repr__(self) -> str:
        loaded = self._model is not None
        return (
            f"CodeEmbedder(model='{self.model_id}', "
            f"device='{self.device}', "
            f"loaded={loaded}, "
            f"batch_size={self.batch_size})"
        )
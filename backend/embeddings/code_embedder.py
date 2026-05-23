"""
code_embedder.py
─────────────────
PURPOSE
-------
Generates dense vector embeddings for code chunks using CodeBERT
(microsoft/codebert-base), a transformer model pre-trained specifically
on code and natural language pairs across six programming languages.

WHY THIS FILE EXISTS
--------------------
Generic text embedding models (e.g. text-embedding-ada-002 trained on
web text) perform poorly on source code because:

  1. Code tokens like `self`, `->`, `[]`, `async/await` are out-of-distribution
     for models trained on prose.
  2. Variable names, function signatures, and type annotations carry semantic
     meaning that generic models underweight.
  3. Code has structural patterns (indentation, block scope, call chains)
     that text models are not trained to encode.

CodeBERT was trained on GitHub code + docstrings using both masked language
modeling AND replaced token detection on code.  It produces embeddings where
semantically similar code (e.g. two different implementations of binary search)
cluster together — which is exactly what we need for retrieval.

WHY CODEBERT OVER OPENAI text-embedding-3-small FOR CODE?
----------------------------------------------------------
CodeBERT is the primary embedder; openai_embedder.py is the fallback for:
  • Queries that mix natural language with code (docstring-heavy chunks)
  • Languages CodeBERT handles poorly (TypeScript edge cases)
  • When a GPU is not available and CodeBERT inference is too slow on CPU

This distinction is a deliberate architectural decision that you should
be able to defend in interviews — it shows you chose tools based on the
actual data distribution, not just convenience.

USED BY
-------
- indexing/qdrant_client.py        receives embeddings to upsert as vectors
- embeddings/embedding_cache.py    caches results to avoid redundant inference
- notebooks/hybrid_retrieval_experiment.ipynb  compared against OpenAI embedder
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# ── Model Constants ────────────────────────────────────────────────────────────

# The HuggingFace model identifier for CodeBERT.
# We pin the exact model name (not a version tag) so that any future
# HuggingFace Hub updates do not silently change embedding dimensions.
CODEBERT_MODEL_ID = "microsoft/codebert-base"

# CodeBERT produces 768-dimensional embeddings (same as BERT-base).
# This constant is used by indexing/qdrant_client.py to configure
# the Qdrant collection's vector size at creation time.
EMBEDDING_DIM = 768

# CodeBERT's maximum input length in tokens (BERT architecture limit).
# Chunks longer than this are truncated.  The chunking strategy in
# tree_sitter_parser.py is designed to keep functions under this limit,
# but we enforce it here as a safety net.
MAX_TOKEN_LENGTH = 512

# Default batch size for inference.  Higher = faster on GPU, but risks OOM.
# 16 is a safe default for a 16GB GPU with 512-token inputs.
# Reduce to 4–8 if you encounter CUDA out-of-memory errors.
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
        The 768-dimensional embedding vector as a plain Python list.
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
    Wraps the CodeBERT model to produce semantic embeddings for code chunks.

    The class handles:
      • Lazy model loading (model is loaded on first use, not at import time)
      • Device selection (CUDA → MPS → CPU, in that priority order)
      • Tokenization with truncation and padding
      • Mean-pooling of token embeddings → single chunk embedding
      • L2 normalization of output vectors (required for cosine similarity
        in Qdrant, which uses dot product on normalized vectors)
      • Batched inference for throughput

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier.  Defaults to CODEBERT_MODEL_ID.
        Override for experiments (e.g. to try graphcodebert-base).
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
        model_id: str = CODEBERT_MODEL_ID,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.model_id = model_id
        self.batch_size = batch_size
        self.cache_dir = str(cache_dir) if cache_dir else None

        # Device selection: prefer CUDA, fall back to Apple MPS, then CPU.
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("CodeEmbedder: using CUDA (GPU)")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("CodeEmbedder: using MPS (Apple Silicon)")
        else:
            self.device = torch.device("cpu")
            logger.warning(
                "CodeEmbedder: no GPU found, using CPU. "
                "Embedding will be slow for large repos. "
                "Consider using openai_embedder.py as fallback."
            )

        # Lazy-loaded — set to None until _load_model() is called.
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._model: PreTrainedModel | None = None

    # ── Model Loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """
        Download and load the CodeBERT tokenizer and model weights.

        Called lazily on the first embed() call.  Lazy loading is used
        because the FastAPI app imports this module at startup, but model
        loading should not block the app from starting up — it should only
        happen when the first ingestion job runs.

        Model weights (~500MB) are cached locally by HuggingFace's
        transformers library after the first download.

        Side effects:
          - Sets self._tokenizer and self._model.
          - Moves the model to self.device.
          - Sets the model to eval mode (disables dropout).
        """
        if self._model is not None:
            return  # Already loaded — idempotent.

        logger.info("Loading CodeBERT model '%s'...", self.model_id)
        load_start = time.time()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
        )
        self._model = AutoModel.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
        )

        # Move to device and set to eval mode.
        # eval() disables dropout layers — critical because we want
        # deterministic embeddings (same input → same output every time).
        self._model = self._model.to(self.device)
        self._model.eval()

        load_time = time.time() - load_start
        logger.info("CodeBERT loaded in %.1fs on device '%s'", load_time, self.device)

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
        Run a single forward pass through CodeBERT for one sub-batch.

        Steps:
          1. Tokenize all texts in the sub-batch with padding and truncation.
          2. Run the tokenized inputs through CodeBERT.
          3. Mean-pool the last hidden states across the token dimension.
          4. L2-normalize the pooled vectors.
          5. Wrap results in EmbeddingResult objects.

        WHY MEAN POOLING INSTEAD OF [CLS] TOKEN?
        -----------------------------------------
        CodeBERT's [CLS] token embedding was trained for classification tasks
        (is this code a docstring match?), not for semantic similarity.
        Mean pooling over all token embeddings produces better similarity
        properties for retrieval — this is empirically validated in the
        sentence-transformers literature and in our own Day 8 experiment.

        WHY L2 NORMALIZATION?
        ----------------------
        Qdrant's cosine similarity is computed as dot product after
        normalizing stored vectors at index time.  By normalizing here at
        embedding time, we ensure that:
          1. All vectors have unit length → dot product = cosine similarity.
          2. Embeddings are comparable across different batch sizes
             (batch norm artifacts don't affect the magnitude).

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

        # ── Step 1: Tokenize ────────────────────────────────────────────────
        encoded = self._tokenizer(
            texts,
            padding=True,           # Pad shorter sequences to the longest in batch.
            truncation=True,        # Truncate sequences longer than MAX_TOKEN_LENGTH.
            max_length=MAX_TOKEN_LENGTH,
            return_tensors="pt",    # Return PyTorch tensors.
            return_attention_mask=True,
        )

        # Detect which inputs were truncated by checking if any sequence hit
        # the max length (a heuristic — not 100% accurate but good enough).
        token_counts = encoded["attention_mask"].sum(dim=1).tolist()
        truncated_flags = [int(tc) >= MAX_TOKEN_LENGTH for tc in token_counts]

        # Move tensors to the target device.
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # ── Step 2: Forward Pass ────────────────────────────────────────────
        with torch.no_grad():
            # no_grad() disables gradient computation — we are doing inference,
            # not training.  This halves memory usage and speeds up the pass.
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # last_hidden_state: shape (batch_size, seq_len, hidden_dim=768)
        last_hidden_state = outputs.last_hidden_state

        # ── Step 3: Mean Pooling ────────────────────────────────────────────
        # Expand the attention mask to match the hidden state dimensions so
        # we can zero-out padding token embeddings before averaging.
        # mask shape: (batch_size, seq_len) → (batch_size, seq_len, hidden_dim)
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

        # Sum token embeddings, ignoring padding positions.
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)

        # Count non-padding tokens per sequence for the average denominator.
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

        # Mean-pooled embedding: shape (batch_size, 768)
        mean_pooled = sum_embeddings / sum_mask

        # ── Step 4: L2 Normalization ────────────────────────────────────────
        # Normalize each embedding vector to unit length.
        # After normalization: ||v|| = 1 for every vector.
        norms = mean_pooled.norm(p=2, dim=1, keepdim=True)
        normalized = mean_pooled / torch.clamp(norms, min=1e-9)

        # ── Step 5: Convert to Python Lists ────────────────────────────────
        # Move back to CPU and convert to numpy for list conversion.
        # numpy() requires CPU tensors — .cpu() is a no-op if already on CPU.
        embeddings_np: np.ndarray = normalized.cpu().numpy()

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

        The query is prefixed with a special token pattern that CodeBERT
        was trained to expect for code-search queries.  This prefix was
        established in the CodeBERT paper (Feng et al., 2020) and improves
        retrieval accuracy for natural language → code queries.

        Parameters
        ----------
        query : str
            The user's query, e.g. "function that validates JWT tokens".

        Returns
        -------
        list[float]
            768-dimensional normalized embedding vector.
        """
        # The CodeBERT paper uses this prefix for NL→code retrieval tasks.
        # It signals to the model that this is a query, not a code snippet.
        prefixed_query = f"<query> {query.strip()}"
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
        logger.info("Running CodeBERT warmup pass...")
        self.embed("def warmup(): pass", chunk_id="__warmup__")
        logger.info("CodeBERT warmup complete")

    def get_embedding_dim(self) -> int:
        """
        Return the output embedding dimensionality.

        Used by indexing/qdrant_client.py when creating a Qdrant collection
        so the vector dimension is always in sync with the actual model output.

        Returns
        -------
        int
            Dimensionality of the embedding vectors (768 for codebert-base).
        """
        return EMBEDDING_DIM

    def unload_model(self) -> None:
        """
        Release the model from GPU/CPU memory.

        Call this when switching to a different embedding model
        (e.g. falling back to openai_embedder.py) to free resources.
        The model will be re-loaded lazily on the next embed() call.
        """
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("CodeBERT model unloaded from memory")

    def __repr__(self) -> str:
        loaded = self._model is not None
        return (
            f"CodeEmbedder(model='{self.model_id}', "
            f"device='{self.device}', "
            f"loaded={loaded}, "
            f"batch_size={self.batch_size})"
        )
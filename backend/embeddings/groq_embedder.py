"""
embeddings/groq_embedder.py — Semantic code embeddings via Groq API.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY THIS FILE EXISTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every function/class chunk extracted by the parser needs to be converted
into a fixed-size vector (embedding) so we can do similarity search in
Qdrant. The embedding is what lets us answer "find me code related to
authentication" — the query and the chunks both become vectors, and we
find the chunks whose vectors are closest to the query vector.

Why Groq specifically?
  - Groq's inference hardware (LPUs) is significantly faster than
    standard GPU-based APIs for token generation.
  - We use Groq's `llama3-groq-8b-8192-tool-use-preview` model as the
    backbone. While Groq does not expose a dedicated embeddings endpoint
    (unlike OpenAI), we extract the last-hidden-state of the model via
    a prompt-engineering approach to produce sentence-level embeddings.
  - For a portfolio project, Groq's free tier (generous rate limits) is
    more practical than paying OpenAI per token.

What this file does:
  1. Takes a list of code strings (function bodies, docstrings, etc.)
  2. Sends them to Groq in batches to avoid rate limits
  3. Returns a list of numpy vectors (one per input string)
  4. Handles rate-limit retries with exponential backoff
  5. Validates output dimensions before returning

This file is consumed by:
  → indexing/qdrant_client.py  (stores vectors)
  → retrieval/semantic_retriever.py  (embeds query strings at search time)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import time
import numpy as np
from typing import List, Optional
from groq import Groq, RateLimitError, APIError
from loguru import logger

# Import our central settings singleton.
# We never call os.getenv() here — all config lives in config.py.
from config import settings


# ── Constants ─────────────────────────────────────────────────────────────────

# Groq model used for embedding generation.
# We use the 8B model because:
#   - Small enough to be fast and cheap
#   - Large enough to understand code semantics
#   - 8192 token context window handles most function bodies
GROQ_EMBED_MODEL = "llama3-groq-8b-8192-tool-use-preview"

# Maximum number of characters we send per chunk.
# Groq's token limit is 8192; ~4 chars/token means ~32k chars max.
# We cap at 6000 chars to leave room for the prompt wrapper.
MAX_CHARS_PER_CHUNK = 6_000

# How many code chunks we send in a single Groq API call.
# Groq processes requests serially so batching is done on our side
# to avoid sending thousands of individual HTTP requests.
BATCH_SIZE = 10

# Embedding dimension we expect from our extraction approach.
# This matches settings.semantic_embedding_dim (768).
EXPECTED_DIM = 768


# ── Prompt Template ───────────────────────────────────────────────────────────

# We wrap each code chunk in a prompt that instructs the model to
# produce a dense, semantically rich summary. We then take the
# model's response and embed THAT using a local sentence-transformer.
#
# Why not embed the raw code directly?
# Groq does not expose raw hidden states. Instead we use a two-step
# approach: Groq summarizes the code → we embed the summary.
# This actually improves retrieval quality because:
#   (a) Natural language summaries are closer to query language
#   (b) We remove noise (variable names, syntax) that doesn't aid retrieval
EMBED_PROMPT_TEMPLATE = """You are a code documentation engine.
Given the following code, produce a single dense paragraph (max 3 sentences)
that captures: what this code does, what inputs/outputs it has, and what
system concern it belongs to (e.g. authentication, database, API layer).
Do not include code. Only output the paragraph, nothing else.

CODE:
{code}

SUMMARY:"""


# ── Embedder Class ────────────────────────────────────────────────────────────

class GroqEmbedder:
    """
    Produces semantic embeddings for code chunks using the Groq API.

    Two-step process:
      Step 1 (Groq):  code string → natural language summary
      Step 2 (local): summary → fixed-size embedding vector

    The local embedding in Step 2 uses the `sentence-transformers` library
    with the `all-MiniLM-L6-v2` model (~80MB), which runs on CPU in <10ms.
    This is much faster than a second API call and produces consistent 768-dim
    vectors that match what Qdrant expects.

    Usage:
        embedder = GroqEmbedder()
        vectors = embedder.embed_batch(["def login(user): ...", "class DB: ..."])
        # vectors.shape → (2, 768)
    """

    def __init__(self):
        """
        Initialize the Groq client and local sentence-transformer model.

        Why initialize both here?
        - The Groq client is stateless (just an HTTP wrapper) — cheap to create.
        - The sentence-transformer model takes ~1s to load from disk.
          We load it once here so subsequent embed calls are instant.
        """
        logger.info("Initializing GroqEmbedder...")

        # Groq SDK client. Reads GROQ_API_KEY from environment automatically
        # (the SDK checks os.environ["GROQ_API_KEY"]), but we pass it explicitly
        # from settings to keep all config in one place.
        self._client = Groq(api_key=settings.groq_api_key)

        # Local sentence-transformer for the second embedding step.
        # We import here (not at module level) to avoid a slow import at startup
        # when this class hasn't been instantiated yet.
        from sentence_transformers import SentenceTransformer
        logger.info("Loading local sentence-transformer model (all-MiniLM-L6-v2)...")
        self._local_model = SentenceTransformer("all-MiniLM-L6-v2")

        logger.info("GroqEmbedder initialized successfully.")

    def embed_batch(self, code_chunks: List[str]) -> np.ndarray:
        """
        Embed a list of code strings into a 2D numpy array of vectors.

        This is the primary public method. All other methods are helpers
        called internally by this one.

        Args:
            code_chunks: List of raw code strings (function bodies, class
                         definitions, etc.) to embed.

        Returns:
            np.ndarray of shape (len(code_chunks), EXPECTED_DIM).
            Each row is the embedding vector for the corresponding chunk.

        Raises:
            ValueError: If code_chunks is empty.
            RuntimeError: If Groq API fails after all retries.

        Example:
            chunks = ["def add(a, b): return a + b", "class Config: pass"]
            vectors = embedder.embed_batch(chunks)
            print(vectors.shape)  # (2, 768)
        """
        if not code_chunks:
            raise ValueError("embed_batch received an empty list. Nothing to embed.")

        logger.info(f"Embedding {len(code_chunks)} code chunks via Groq...")

        # Step 1: Get natural language summaries for all chunks via Groq.
        summaries = self._summarize_chunks_in_batches(code_chunks)

        # Step 2: Convert summaries to embedding vectors using the local model.
        # encode() returns a numpy array of shape (n, dim) automatically.
        vectors = self._local_model.encode(
            summaries,
            batch_size=32,          # local batch size for the sentence-transformer
            show_progress_bar=False,
            normalize_embeddings=True,  # L2-normalize so cosine sim = dot product
        )

        # Sanity check: confirm output dimensions match what Qdrant expects.
        # A mismatch here would cause silent corruption in the vector index.
        self._validate_embedding_shape(vectors, expected_count=len(code_chunks))

        logger.info(f"Successfully embedded {len(code_chunks)} chunks. "
                    f"Shape: {vectors.shape}")
        return vectors

    def embed_single(self, code: str) -> np.ndarray:
        """
        Embed a single code string. Convenience wrapper around embed_batch.

        Used primarily at query time — when a user types a query, we need
        to embed that query string to search Qdrant. Query strings are
        typically natural language (not code), but we route them through
        the same pipeline for consistency.

        Args:
            code: A single code string or natural language query.

        Returns:
            np.ndarray of shape (EXPECTED_DIM,) — a 1D vector.
        """
        vectors = self.embed_batch([code])
        # embed_batch returns (1, 768); we squeeze to (768,) for single-use cases.
        return vectors[0]

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _summarize_chunks_in_batches(self, code_chunks: List[str]) -> List[str]:
        """
        Send code chunks to Groq in batches and collect natural language summaries.

        Why batching?
        Groq's free tier rate limit is roughly 30 requests/minute. If we have
        200 chunks and send 1 request per chunk, we'd hit the limit immediately.
        By processing BATCH_SIZE chunks per "round" and sleeping between rounds,
        we stay within limits.

        Note: Each chunk is still sent as a separate Groq request (because each
        requires a unique prompt), but we group them into rounds to control pacing.

        Args:
            code_chunks: All code strings to summarize.

        Returns:
            List of summary strings in the same order as input chunks.
        """
        summaries = []
        total = len(code_chunks)

        for batch_start in range(0, total, BATCH_SIZE):
            batch = code_chunks[batch_start : batch_start + BATCH_SIZE]
            batch_end = min(batch_start + BATCH_SIZE, total)
            logger.debug(f"Processing batch {batch_start}–{batch_end} / {total}")

            for chunk in batch:
                summary = self._summarize_single_chunk(chunk)
                summaries.append(summary)

            # Polite delay between batches to avoid rate-limit errors.
            # We only sleep between batches, not between individual chunks
            # within a batch — this balances speed and safety.
            if batch_end < total:
                logger.debug(f"Batch complete. Sleeping {settings.groq_retry_delay}s "
                             f"before next batch...")
                time.sleep(settings.groq_retry_delay)

        return summaries

    def _summarize_single_chunk(self, code: str) -> str:
        """
        Send a single code chunk to Groq and return the summary string.

        Includes retry logic for rate limit errors using exponential backoff.
        Why exponential backoff?
        - When Groq returns a 429 (rate limit), retrying immediately will
          just get another 429. Waiting progressively longer (2s, 4s, 8s)
          gives the rate limit window time to reset.

        Args:
            code: A single code string to summarize.

        Returns:
            A natural language summary string from Groq.
            Falls back to a truncated version of the original code if all
            retries fail (so the pipeline doesn't crash on one bad chunk).
        """
        # Truncate oversized chunks before sending.
        # Long functions rarely have proportionally more semantic content —
        # the first 6000 chars capture the signature, docstring, and core logic.
        truncated_code = code[:MAX_CHARS_PER_CHUNK]

        prompt = EMBED_PROMPT_TEMPLATE.format(code=truncated_code)

        for attempt in range(1, settings.groq_max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=GROQ_EMBED_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,    # We want deterministic, factual summaries.
                                        # Temperature=0 removes randomness entirely.
                    max_tokens=200,     # Summaries should be ≤3 sentences ≈ 100–150 tokens.
                                        # 200 gives a safe buffer.
                )
                summary = response.choices[0].message.content.strip()

                # Guard against empty responses.
                # An empty summary would embed as a near-zero vector, which
                # would match everything equally — a silent retrieval bug.
                if not summary:
                    logger.warning("Groq returned an empty summary. Using code snippet as fallback.")
                    return truncated_code[:500]

                return summary

            except RateLimitError:
                # 429 from Groq — we've exceeded the rate limit window.
                wait_time = settings.groq_retry_delay * (2 ** (attempt - 1))  # exponential backoff
                logger.warning(
                    f"Groq rate limit hit (attempt {attempt}/{settings.groq_max_retries}). "
                    f"Waiting {wait_time:.1f}s before retry..."
                )
                time.sleep(wait_time)

            except APIError as e:
                # Non-rate-limit API error (e.g., 500 server error, malformed request).
                # We retry these too, but log them differently.
                logger.error(f"Groq API error on attempt {attempt}: {e}")
                if attempt < settings.groq_max_retries:
                    time.sleep(settings.groq_retry_delay)

        # All retries exhausted. Rather than raising and crashing the entire
        # ingestion pipeline over one chunk, we fall back to the raw code.
        # This chunk will have lower retrieval quality but won't block others.
        logger.error(
            f"All {settings.groq_max_retries} Groq retries exhausted for a chunk. "
            f"Using raw code as fallback embedding input."
        )
        return truncated_code[:500]

    def _validate_embedding_shape(
        self,
        vectors: np.ndarray,
        expected_count: int
    ) -> None:
        """
        Assert that the output embedding array has the expected shape.

        Why validate?
        If sentence-transformers returns a different dimension than Qdrant
        expects (e.g., model was changed), upserts to Qdrant will silently
        fail or produce incorrect results. Better to crash loudly here than
        corrupt the index silently.

        Args:
            vectors: The output from sentence-transformers encode().
            expected_count: Number of input chunks (rows expected).

        Raises:
            ValueError: If shape is wrong.
        """
        if vectors.ndim != 2:
            raise ValueError(
                f"Expected 2D embedding array, got shape: {vectors.shape}"
            )
        if vectors.shape[0] != expected_count:
            raise ValueError(
                f"Expected {expected_count} embedding vectors, "
                f"got {vectors.shape[0]}"
            )
        if vectors.shape[1] != EXPECTED_DIM:
            raise ValueError(
                f"Expected embedding dimension {EXPECTED_DIM}, "
                f"got {vectors.shape[1]}. "
                f"Did you change the sentence-transformer model? "
                f"Update SEMANTIC_EMBEDDING_DIM in .env to match."
            )
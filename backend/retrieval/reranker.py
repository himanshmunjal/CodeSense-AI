"""
retrieval/reranker.py
─────────────────────────────────────────────────────────────────────────────
WHY THIS FILE EXISTS
─────────────────────────────────────────────────────────────────────────────
Vector similarity search (bi-encoder models like CodeBERT) is fast and scales
to millions of documents, but it has a known quality ceiling: it encodes the
query and each document independently, then compares their embeddings. It
never "sees" the query and document together at the same time.

A cross-encoder fixes this. It takes the query AND a candidate document as a
single input, letting the model attend across both simultaneously. This
produces much higher-quality relevance scores — but it is far too slow to run
over an entire corpus (O(N) full forward passes vs O(1) ANN lookup).

The standard industry solution: two-stage retrieval.
  Stage 1 — Fast recall:  bi-encoder fetches top-K candidates (K=20 here)
  Stage 2 — Precise rank: cross-encoder re-scores and re-ranks those K results

This is exactly what Google's production ranking pipelines do, and it is what
this file implements. The hybrid_retriever feeds 20 candidates in; the re-
ranker outputs the top N (default 5) in the correct relevance order.

WHY THIS MATTERS IN INTERVIEWS
─────────────────────────────────────────────────────────────────────────────
This is a signal of production search maturity. Most portfolio RAG projects
pass vector search results directly to the LLM. Explaining that you added a
cross-encoder re-ranking stage — and why — demonstrates you understand the
bi-encoder/cross-encoder tradeoff and the recall-precision pipeline pattern.

POSITION IN PIPELINE
─────────────────────────────────────────────────────────────────────────────
hybrid_retriever.py → [this file] → generation/prompt_builder.py
"""

from dataclasses import dataclass
from typing import Optional

import torch
from sentence_transformers import CrossEncoder
from loguru import logger

from retrieval.hybrid_retriever import HybridResult
from config import get_settings

settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RankedResult:
    """
    A single result after cross-encoder re-ranking.

    This is the final retrieval output before generation. It extends
    HybridResult with the cross-encoder score and rank position.

    Fields
    ──────
    hybrid_result : HybridResult
        The original result from the hybrid retriever, carried through
        unchanged. The generator needs the code, file_path, function_name,
        and line numbers for citation building.

    rerank_score : float
        Raw logit score from the cross-encoder. Higher = more relevant.
        Note: this is NOT bounded to [0, 1] — cross-encoder outputs raw
        logits. Do not compare this to hybrid_score or the confidence
        threshold.

    rank : int
        Final position in the ranked list (1-indexed).
        Rank 1 = most relevant to the query.
    """
    hybrid_result: HybridResult
    rerank_score: float
    rank: int

    # ── Convenience pass-throughs ──────────────────────────────────────
    # These let callers use ranked_result.file_path instead of
    # ranked_result.hybrid_result.file_path — cleaner call sites.

    @property
    def file_path(self) -> str:
        return self.hybrid_result.file_path

    @property
    def function_name(self) -> str:
        return self.hybrid_result.function_name

    @property
    def code(self) -> str:
        return self.hybrid_result.code

    @property
    def docstring(self) -> str:
        return self.hybrid_result.docstring

    @property
    def start_line(self) -> int:
        return self.hybrid_result.start_line

    @property
    def end_line(self) -> int:
        return self.hybrid_result.end_line

    @property
    def language(self) -> str:
        return self.hybrid_result.language

    @property
    def hybrid_score(self) -> float:
        return self.hybrid_result.hybrid_score


# ─────────────────────────────────────────────────────────────────────────────
# Main Class
# ─────────────────────────────────────────────────────────────────────────────

class Reranker:
    """
    Cross-encoder re-ranker for retrieved code chunks.

    Model
    ─────
    Uses `cross-encoder/ms-marco-MiniLM-L-6-v2`, a lightweight (22M param)
    cross-encoder trained on the MS MARCO passage ranking dataset. It was
    trained for query-passage relevance on natural language, but transfers
    well to code because:
      a) User queries are natural language ("where is auth handled?")
      b) Function docstrings included with the code provide NL context
      c) MiniLM-L-6 is fast enough that re-ranking 20 results takes ~100ms
         on CPU (well within the 3s p90 latency budget)

    For a code-specific cross-encoder, one option is to fine-tune this model
    on code-NL pairs from CodeSearchNet — but that is out of scope for v1.

    Device Selection
    ────────────────
    Automatically uses CUDA if available, MPS (Apple Silicon) if available,
    otherwise falls back to CPU. For a 22M parameter model, CPU inference
    is fast enough that GPU is a nice-to-have, not a requirement.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Load the cross-encoder model.

        Model loading happens once at startup (when the FastAPI app boots),
        not on every request. This is why Reranker is instantiated at the
        module level in api/main.py and injected via dependency injection.

        Parameters
        ----------
        model_name : str, optional
            HuggingFace model identifier. Defaults to
            settings.reranker_model_name ("cross-encoder/ms-marco-MiniLM-L-6-v2").
            Override in tests to use a smaller/mock model.
        """
        self.model_name = model_name or settings.reranker_model_name

        # Determine the best available device.
        # The CrossEncoder class from sentence-transformers handles the
        # device assignment internally, but we log it here for observability
        # — knowing whether you're on GPU or CPU is important for latency
        # debugging.
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        logger.info(
            f"Loading cross-encoder '{self.model_name}' on device '{self.device}'"
        )

        # sentence-transformers CrossEncoder wraps the HuggingFace model and
        # provides a clean .predict(pairs) interface that handles tokenization,
        # batching, and device placement automatically.
        self.model = CrossEncoder(
            self.model_name,
            device=self.device,
            # max_length=512 is the model's context window.
            # Code + docstring combinations that exceed this are truncated from
            # the right — meaning the function signature and early lines are
            # always included, which is where the most semantically dense
            # information sits.
            max_length=512,
        )

        logger.info(f"Cross-encoder loaded successfully")

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        candidates: list[HybridResult],
        top_n: Optional[int] = None,
    ) -> list[RankedResult]:
        """
        Re-rank a list of hybrid retrieval candidates using the cross-encoder.

        The model scores each (query, document) pair jointly. "document" here
        is a concatenation of the function's docstring and code body — both
        are included because the docstring provides natural language context
        that helps the cross-encoder (which was trained on NL text) assess
        relevance to the query.

        Parameters
        ----------
        query : str
            The original user query. Passed verbatim to the cross-encoder.

        candidates : list[HybridResult]
            The top-K results from hybrid_retriever.retrieve(). Typically
            K=20 candidates are passed in.

        top_n : int, optional
            How many results to return after re-ranking. Defaults to
            settings.reranker_top_n (5). These 5 results are what gets
            passed to the prompt builder.

        Returns
        -------
        list[RankedResult]
            Top-N re-ranked results, sorted by rerank_score descending.
            The list is at most top_n items; it may be shorter if fewer
            candidates were passed in.
        """
        top_n = top_n or settings.reranker_top_n

        if not candidates:
            logger.warning("Reranker called with empty candidate list")
            return []

        logger.debug(
            f"Re-ranking {len(candidates)} candidates | "
            f"query='{query[:60]}' | top_n={top_n}"
        )

        # ── Build (query, document) pairs ─────────────────────────────────
        # The cross-encoder expects a list of [query, document] pairs.
        # We format the document as: docstring (if present) + code body.
        # Including the docstring helps the model understand what the function
        # does before seeing implementation details.
        pairs = []
        for candidate in candidates:
            document = self._format_document(candidate)
            pairs.append([query, document])

        # ── Score all pairs in a single batched forward pass ──────────────
        # CrossEncoder.predict() handles batching internally. Scoring all 20
        # pairs at once is faster than scoring them one by one because it
        # maximizes GPU/CPU utilization per forward pass.
        scores = self.model.predict(
            pairs,
            batch_size=min(len(pairs), 16),  # 16 is a safe default batch size
            show_progress_bar=False,          # No progress bar in production
            convert_to_numpy=True,            # Returns np.ndarray for easy sorting
        )

        # ── Sort by score descending and take top-N ───────────────────────
        # zip() pairs each candidate with its score, then sort by score.
        scored_candidates = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # ── Build RankedResult objects with rank positions ────────────────
        ranked = []
        for rank_idx, (candidate, score) in enumerate(scored_candidates[:top_n]):
            ranked.append(RankedResult(
                hybrid_result=candidate,
                rerank_score=float(score),
                rank=rank_idx + 1,  # 1-indexed for human readability
            ))

        logger.info(
            f"Re-ranking complete | "
            f"returning {len(ranked)}/{len(candidates)} | "
            f"top score={ranked[0].rerank_score:.3f} | "
            f"bottom score={ranked[-1].rerank_score:.3f}"
        )

        return ranked

    # ─────────────────────────────────────────────────────────────────────
    # Private Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _format_document(self, candidate: HybridResult) -> str:
        """
        Format a HybridResult into a single string for the cross-encoder.

        Format
        ──────
            [DOCSTRING]
            <docstring text>

            [CODE]
            <code text>

        The structured header tokens ([DOCSTRING], [CODE]) help the model
        understand the document structure. This is a lightweight form of
        prompt formatting for a discriminative model (the cross-encoder is
        not generative — it just outputs a relevance score).

        Why include the docstring separately?
        The docstring is natural language. The cross-encoder was trained on
        NL text. By surfacing the docstring prominently at the top, we give
        the model the best chance of matching query intent to function
        purpose — even for queries that use different vocabulary than the
        code itself uses.

        Parameters
        ----------
        candidate : HybridResult
            The chunk to format.

        Returns
        -------
        str
            Formatted document string. Truncation (if needed) is handled
            by the CrossEncoder model's tokenizer (max_length=512).
        """
        parts = []

        # Add docstring section if present
        if candidate.docstring and candidate.docstring.strip():
            parts.append(f"[DOCSTRING]\n{candidate.docstring.strip()}")

        # Add code section — always present (it's the core of the chunk)
        if candidate.code and candidate.code.strip():
            parts.append(f"[CODE]\n{candidate.code.strip()}")

        # Add file path context — helps the model understand the module
        # context (e.g. "auth/token.py" signals this is about authentication)
        parts.append(f"[FILE] {candidate.file_path}")

        return "\n\n".join(parts)

    def explain_reranking(
        self,
        query: str,
        before: list[HybridResult],
        after: list[RankedResult],
    ) -> str:
        """
        Generate a human-readable explanation of how re-ranking changed the
        result order.

        Used in the Day 8 experiment notebook and during debugging to
        understand cases where re-ranking moved a result significantly up or
        down relative to its hybrid score position.

        Example output
        ──────────────
            Re-ranking changed order for query: "where is auth handled?"
            Rank changes:
              auth/token.py::validate_token    hybrid_rank=3 → rerank_rank=1 (+2)
              auth/session.py::create_session  hybrid_rank=1 → rerank_rank=2 (-1)
              utils/helpers.py::parse_header   hybrid_rank=2 → rerank_rank=3 (-1)

        Parameters
        ----------
        query : str
            The query, included in the explanation header.

        before : list[HybridResult]
            Results in hybrid_score order (before re-ranking).

        after : list[RankedResult]
            Results in rerank_score order (after re-ranking).
        """
        lines = [f'Re-ranking order changes for query: "{query}"']

        # Build a lookup of chunk_id → hybrid rank position
        hybrid_rank_map = {
            r.chunk_id: idx + 1
            for idx, r in enumerate(before)
        }

        for ranked in after:
            chunk_id = ranked.hybrid_result.chunk_id
            hybrid_pos = hybrid_rank_map.get(chunk_id, "?")
            rerank_pos = ranked.rank
            delta = (
                int(hybrid_pos) - rerank_pos
                if isinstance(hybrid_pos, int) else "?"
            )
            direction = f"+{delta}" if isinstance(delta, int) and delta > 0 else str(delta)
            lines.append(
                f"  {ranked.file_path}::{ranked.function_name:<30} "
                f"hybrid_rank={hybrid_pos} → rerank_rank={rerank_pos} ({direction})"
            )

        return "\n".join(lines)
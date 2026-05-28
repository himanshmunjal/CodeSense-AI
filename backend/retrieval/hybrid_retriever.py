"""
retrieval/hybrid_retriever.py
─────────────────────────────────────────────────────────────────────────────
WHY THIS FILE EXISTS
─────────────────────────────────────────────────────────────────────────────
Semantic search alone (finding code by meaning) and structural search alone
(traversing the call graph) each have blind spots:

  - Semantic-only: great at finding "payment-related code" but cannot answer
    "what calls this function?" because vector similarity has no concept of
    code relationships.

  - Structural-only: great at traversing call chains but cannot find code by
    intent — it requires you to already know the node name.

This file fuses both signals into a single ranked result list using a
weighted combination of their scores. It is the central retrieval brain of
CodeSense. Every user query ultimately passes through here (unless the query
classifier routes it to a pure-graph traversal, which is handled separately
in graph_retriever.py).

The design mirrors what high-quality production search systems do — BM25 +
dense vector fusion in Elasticsearch, for example. We apply the same idea to
code: structural centrality + semantic similarity = better retrieval.

POSITION IN PIPELINE
─────────────────────────────────────────────────────────────────────────────
Query → query_classifier.py → [this file] → reranker.py → generator.py
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

# Local imports — each handles one half of the hybrid signal
from retrieval.semantic_retriever import SemanticRetriever, SemanticResult
from retrieval.graph_retriever import GraphRetriever, GraphResult
from config import get_settings

settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HybridResult:
    """
    A single retrieved code chunk after hybrid scoring.

    This is the output unit of the hybrid retriever and the input unit for
    the re-ranker. It carries both the chunk content and all metadata needed
    for citation generation (file path, line numbers, etc.).

    Fields
    ------
    chunk_id : str
        Unique identifier for this chunk in Qdrant. Used for deduplication
        when the same chunk surfaces in both semantic and structural results.

    file_path : str
        Relative path within the repository, e.g. "src/auth/token.py".
        Displayed in citations and used by the frontend source viewer.

    function_name : str
        Name of the function or class this chunk represents.

    start_line / end_line : int
        Line range for the source viewer to highlight.

    language : str
        Programming language, e.g. "python". Used for syntax highlighting.

    code : str
        The actual source code text. This is what gets fed to the LLM.

    docstring : str
        Docstring or inline comment block, if present. Included in the
        prompt as additional context separate from the code body.

    semantic_score : float
        Cosine similarity score from the vector search, range [0, 1].
        Higher = more semantically similar to the query.

    structural_score : float
        Graph-based relevance score from the call graph analysis.
        Higher = more structurally central / more relevant to changed nodes.
        Defaults to 0.0 if the chunk has no graph presence (e.g. top-level
        scripts with no call relationships).

    hybrid_score : float
        The fused final score used for ranking.
        Formula: hybrid = w_sem * semantic + w_str * structural
        Weights come from settings (default: 0.7 semantic, 0.3 structural).

    graph_distance : Optional[int]
        For relational queries, how many hops away this chunk is from the
        query node in the call graph. None for pure semantic results.
    """
    chunk_id: str
    file_path: str
    function_name: str
    start_line: int
    end_line: int
    language: str
    code: str
    docstring: str = ""
    semantic_score: float = 0.0
    structural_score: float = 0.0
    hybrid_score: float = 0.0
    graph_distance: Optional[int] = None
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Main Class
# ─────────────────────────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Fuses semantic and structural retrieval into a single ranked result list.

    Architecture
    ────────────
    1. Both retrieval paths run concurrently (asyncio.gather).
    2. Results are merged into a shared dict keyed by chunk_id.
       If a chunk appears in both, its scores are combined.
       If it appears in only one, the missing score defaults to 0.0.
    3. Each chunk's final score is computed:
           hybrid_score = w_sem * semantic_score + w_str * structural_score
    4. The result list is sorted by hybrid_score descending.
    5. Low-confidence results (below settings.retrieval_confidence_threshold)
       are filtered out before returning.

    The caller (query_classifier or the API route) decides how many results
    to request. This class does not impose a top-K limit — that is the
    re-ranker's job.
    """

    def __init__(
        self,
        semantic_retriever: SemanticRetriever,
        graph_retriever: GraphRetriever,
        semantic_weight: Optional[float] = None,
        structural_weight: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        semantic_retriever : SemanticRetriever
            Handles vector similarity search against Qdrant. Injected so
            it can be mocked in tests without a live Qdrant instance.

        graph_retriever : GraphRetriever
            Handles call graph traversal using networkx. Injected for the
            same reason.

        semantic_weight : float, optional
            Override the global semantic weight from settings. Useful in
            experiments where you want to test different fusion ratios
            without changing the .env file.

        structural_weight : float, optional
            Override the global structural weight from settings.

        Note: semantic_weight + structural_weight must sum to 1.0. If they
        don't, a ValueError is raised at init time rather than silently
        producing wrong scores at query time.
        """
        self.semantic_retriever = semantic_retriever
        self.graph_retriever = graph_retriever

        # Use provided weights or fall back to global config
        self.semantic_weight = semantic_weight or settings.hybrid_semantic_weight
        self.structural_weight = structural_weight or settings.hybrid_structural_weight

        # Validate weights sum to 1.0 at construction time.
        # Catching this early prevents subtle ranking bugs that are hard to
        # trace back to a misconfigured weight.
        total = self.semantic_weight + self.structural_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Hybrid weights must sum to 1.0, got {total:.4f}. "
                f"(semantic={self.semantic_weight}, structural={self.structural_weight})"
            )

        logger.info(
            f"HybridRetriever initialized | "
            f"semantic_weight={self.semantic_weight}, "
            f"structural_weight={self.structural_weight}"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        repo_name: str,
        top_k: int = None,
        filter_language: Optional[str] = None,
        filter_file_prefix: Optional[str] = None,
    ) -> list[HybridResult]:
        """
        Main entry point. Retrieve and fuse results for a user query.

        Parameters
        ----------
        query : str
            The user's natural language question, e.g.
            "Where is authentication handled?"

        repo_name : str
            The Qdrant collection to search against, formatted as
            "{owner}_{repo}", e.g. "tiangolo_fastapi". This scopes
            all retrieval to a single repository.

        top_k : int, optional
            How many raw candidates to fetch from each retriever before
            fusion. Defaults to settings.retrieval_top_k (20).
            The final returned list may be shorter after confidence
            filtering.

        filter_language : str, optional
            If provided, restrict semantic search to chunks from this
            language, e.g. "python". Passed as a Qdrant payload filter.
            Useful for polyglot repos where the user is clearly asking
            about one language.

        filter_file_prefix : str, optional
            If provided, restrict semantic search to files whose path
            starts with this prefix, e.g. "src/auth/". Useful for
            analytical queries scoped to a subsystem.

        Returns
        -------
        list[HybridResult]
            Fused, scored, and sorted results. Already filtered by
            confidence threshold. The re-ranker will do the final top-N
            selection.
        """
        top_k = top_k or settings.retrieval_top_k

        logger.debug(
            f"HybridRetriever.retrieve | query='{query[:60]}...' | "
            f"repo={repo_name} | top_k={top_k}"
        )

        # ── Step 1: Run both retrievers concurrently ──────────────────────
        # asyncio.gather fires both coroutines at the same time.
        # Semantic search hits Qdrant over the network; graph traversal is
        # in-memory but can be CPU-heavy for large graphs. Running them
        # concurrently shaves ~30-50% off total retrieval latency.
        semantic_results, structural_results = await asyncio.gather(
            self.semantic_retriever.search(
                query=query,
                repo_name=repo_name,
                top_k=top_k,
                filter_language=filter_language,
                filter_file_prefix=filter_file_prefix,
            ),
            self.graph_retriever.search(
                query=query,
                repo_name=repo_name,
                top_k=top_k,
            ),
            return_exceptions=True,  # Don't crash if one retriever fails
        )

        # Handle partial failures gracefully.
        # If semantic search fails, we still try to return structural results
        # and vice versa. Degraded results are far better than a 500 error.
        if isinstance(semantic_results, Exception):
            logger.error(f"Semantic retriever failed: {semantic_results}")
            semantic_results = []

        if isinstance(structural_results, Exception):
            logger.error(f"Graph retriever failed: {structural_results}")
            structural_results = []

        # ── Step 2: Merge results by chunk_id ────────────────────────────
        merged = self._merge_results(semantic_results, structural_results)

        # ── Step 3: Compute hybrid scores ────────────────────────────────
        scored = self._compute_hybrid_scores(merged)

        # ── Step 4: Filter by confidence threshold ───────────────────────
        # This is a hard gate. If the best result is below 0.65, the query
        # has no reliable answer in this codebase and should not proceed to
        # generation. The generator checks this again, but filtering here
        # avoids wasting tokens on a prompt build.
        filtered = [
            r for r in scored
            if r.hybrid_score >= settings.retrieval_confidence_threshold
        ]

        if not filtered:
            logger.warning(
                f"No results above confidence threshold "
                f"({settings.retrieval_confidence_threshold}) for query: '{query[:60]}'"
            )
            return []

        # ── Step 5: Sort by hybrid_score descending ───────────────────────
        filtered.sort(key=lambda r: r.hybrid_score, reverse=True)

        logger.info(
            f"HybridRetriever returned {len(filtered)} results | "
            f"top score={filtered[0].hybrid_score:.3f}"
        )
        return filtered

    # ─────────────────────────────────────────────────────────────────────
    # Private Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _merge_results(
        self,
        semantic_results: list[SemanticResult],
        structural_results: list[GraphResult],
    ) -> dict[str, HybridResult]:
        """
        Combine semantic and structural results into a single dict.

        Both result types may contain the same chunk (a function that is both
        semantically similar to the query AND structurally central in the call
        graph). Merging by chunk_id ensures it appears once in the final list
        with both scores populated, rather than appearing twice with one score
        each.

        For chunks that appear in only one result set:
          - Semantic-only: structural_score = 0.0
          - Structural-only: semantic_score = 0.0

        Returns
        -------
        dict[str, HybridResult]
            Keys are chunk_ids. Values are HybridResult objects with both
            scores populated (or 0.0 where not available).
        """
        merged: dict[str, HybridResult] = {}

        # Process semantic results first
        for sem in semantic_results:
            merged[sem.chunk_id] = HybridResult(
                chunk_id=sem.chunk_id,
                file_path=sem.file_path,
                function_name=sem.function_name,
                start_line=sem.start_line,
                end_line=sem.end_line,
                language=sem.language,
                code=sem.code,
                docstring=sem.docstring,
                semantic_score=sem.score,
                structural_score=0.0,   # Will be filled in below if available
                metadata=sem.metadata,
            )

        # Overlay structural results onto the merged dict.
        # If a chunk already exists (from semantic), add its structural score.
        # If it's new (only in structural), create a new entry with
        # semantic_score=0.0.
        for struct in structural_results:
            if struct.chunk_id in merged:
                # Chunk found in both — augment existing entry
                merged[struct.chunk_id].structural_score = struct.score
                merged[struct.chunk_id].graph_distance = struct.graph_distance
            else:
                # Structural-only chunk — create entry with no semantic score
                merged[struct.chunk_id] = HybridResult(
                    chunk_id=struct.chunk_id,
                    file_path=struct.file_path,
                    function_name=struct.function_name,
                    start_line=struct.start_line,
                    end_line=struct.end_line,
                    language=struct.language,
                    code=struct.code,
                    docstring=struct.docstring,
                    semantic_score=0.0,
                    structural_score=struct.score,
                    graph_distance=struct.graph_distance,
                    metadata=struct.metadata,
                )

        logger.debug(
            f"Merged {len(semantic_results)} semantic + "
            f"{len(structural_results)} structural → {len(merged)} unique chunks"
        )
        return merged

    def _compute_hybrid_scores(
        self,
        merged: dict[str, HybridResult],
    ) -> list[HybridResult]:
        """
        Apply weighted fusion to compute a final hybrid_score for each chunk.

        Formula
        ───────
            hybrid_score = (w_sem × semantic_score) + (w_str × structural_score)

        Both input scores are already in [0, 1] from their respective
        retrievers. The weighted sum is therefore also in [0, 1], making it
        directly comparable to the confidence threshold.

        Why this formula instead of something fancier (e.g. RRF)?
        Reciprocal Rank Fusion is rank-based and doesn't use the raw scores —
        it works well when scores from different systems are on incomparable
        scales. Here, both scores are cosine similarities normalized to [0, 1],
        so a weighted linear combination is appropriate and interpretable.

        Parameters
        ----------
        merged : dict[str, HybridResult]
            Output of _merge_results.

        Returns
        -------
        list[HybridResult]
            Same chunks with hybrid_score populated.
        """
        results = list(merged.values())
        for result in results:
            result.hybrid_score = (
                self.semantic_weight * result.semantic_score
                + self.structural_weight * result.structural_score
            )
        return results

    def score_breakdown(self, result: HybridResult) -> str:
        """
        Human-readable score breakdown for a single result.

        Used in logging, debugging, and the Day 8 experiment notebook to
        understand which signal contributed more to each result's ranking.

        Example output:
            "auth/token.py::validate_token | hybrid=0.821 (sem=0.910×0.7 + str=0.543×0.3)"
        """
        sem_contribution = self.semantic_weight * result.semantic_score
        str_contribution = self.structural_weight * result.structural_score
        return (
            f"{result.file_path}::{result.function_name} | "
            f"hybrid={result.hybrid_score:.3f} "
            f"(sem={result.semantic_score:.3f}×{self.semantic_weight} + "
            f"str={result.structural_score:.3f}×{self.structural_weight}) | "
            f"sem_contrib={sem_contribution:.3f}, str_contrib={str_contribution:.3f}"
        )
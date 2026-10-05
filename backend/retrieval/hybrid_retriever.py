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
CodeSense, and it is used specifically for ANALYTICAL queries (see
api/routes/query.py's dispatch logic) — LOOKUP/SUMMARIZATION go through
SemanticRetriever alone, and RELATIONAL goes through GraphRetriever alone,
because those two query types ask a question the *other* signal cannot
answer at all ("find code like X" has no graph meaning; "what calls X" has
no vector-similarity meaning). ANALYTICAL queries ("find inconsistent error
handling", "which functions are riskiest to touch") are exactly the case
where blending "semantically relevant" with "structurally central" produces
a better-ordered list than either signal alone.

The design mirrors what high-quality production search systems do — BM25 +
dense vector fusion in Elasticsearch, for example. We apply the same idea to
code: structural centrality + semantic similarity = better retrieval.

HOW THE STRUCTURAL SIGNAL IS COMPUTED
─────────────────────────────────────────────────────────────────────────────
GraphRetriever (retrieval/graph_retriever.py) has no generic "search by
free-text query" method — it only supports targeted traversals from a
*known* function name (get_callers, get_callees, get_impact, get_path,
get_file_nodes). There is no way to ask it "which nodes are relevant to
this natural-language query" the way SemanticRetriever can.

So instead of running two independent searches and merging by chunk_id (the
original — and unworkable — design of this file), we run semantic search
first to get a candidate set, and then use the graph purely as a per-
candidate *signal*: for each semantically-retrieved chunk, we ask "how many
functions directly call this one?" via GraphRetriever.get_callers(...,
max_depth=1). A function with many direct callers is structurally central —
changing it or misunderstanding it has outsized impact — so it is boosted.
A leaf function with zero callers gets structural_score=0.0 and is ranked
on semantic similarity alone. This is the "structural centrality" the
original docstring above referred to.

POSITION IN PIPELINE
─────────────────────────────────────────────────────────────────────────────
Query → query_classifier.py → [this file, ANALYTICAL only] → reranker.py → generator.py
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

# Local imports — each handles one half of the hybrid signal
from retrieval.semantic_retriever import SemanticRetriever, SemanticSearchResult
from retrieval.graph_retriever import GraphRetriever
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
    Fuses semantic similarity with graph-derived structural centrality into
    a single ranked result list.

    Architecture
    ────────────
    1. Run semantic search once to get a candidate set (SemanticRetriever).
    2. For each candidate chunk, ask the graph "how many functions directly
       call this one?" (GraphRetriever.get_callers(max_depth=1)). This is
       the structural_score, normalized to [0, 1].
    3. Each chunk's final score is computed:
           hybrid_score = w_sem * semantic_score + w_str * structural_score
    4. The result list is sorted by hybrid_score descending.
    5. Low-confidence results (below settings.retrieval_confidence_threshold)
       are filtered out before returning.

    The caller (api/routes/query.py) decides how many results to request.
    This class does not impose a top-K limit — that is the re-ranker's job.
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

        # ── Step 1: Semantic search gives us the candidate set ────────────
        # There is no generic "search by free text" on GraphRetriever (it
        # only supports targeted traversals from a known function name), so
        # semantic search is the only way to turn a natural-language query
        # into a set of candidate chunks. SemanticRetriever.retrieve() is
        # synchronous (it wraps a blocking Qdrant HTTP call) — run it in a
        # thread so we don't block the event loop.
        filters: dict = {}
        if filter_language:
            filters["language"] = filter_language
        if filter_file_prefix:
            filters["file_path_prefix"] = filter_file_prefix

        try:
            semantic_result: SemanticSearchResult = await asyncio.to_thread(
                self.semantic_retriever.retrieve,
                query=query,
                collection_name=repo_name,
                top_k=top_k,
                filters=filters or None,
            )
        except Exception as e:
            logger.error(f"Semantic retriever failed: {e}")
            return []

        if semantic_result.is_empty:
            logger.warning(f"No semantic candidates for query: '{query[:60]}'")
            return []

        # ── Step 2: Load the call graph for structural augmentation ───────
        # Best-effort: if the repo has no graph (e.g. ingestion hasn't built
        # one yet, or the language isn't graph-supported), we still return
        # semantic-only results with structural_score=0.0 rather than failing
        # the whole query.
        graph_loaded = await asyncio.to_thread(
            self.graph_retriever.load_graph, repo_name
        )

        # ── Step 3: Score each candidate with semantic + structural signal ─
        results = await asyncio.gather(*(
            self._score_chunk(chunk, repo_name, graph_loaded)
            for chunk in semantic_result.chunks
        ))

        # ── Step 4: Filter by confidence threshold ───────────────────────
        # This is a hard gate. If the best result is below 0.65, the query
        # has no reliable answer in this codebase and should not proceed to
        # generation. The generator checks this again, but filtering here
        # avoids wasting tokens on a prompt build.
        filtered = [
            r for r in results
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

    async def _score_chunk(
        self,
        chunk,  # retrieval.semantic_retriever.RetrievedChunk
        repo_name: str,
        graph_loaded: bool,
    ) -> HybridResult:
        """
        Build a HybridResult for one semantically-retrieved chunk, augmented
        with a structural centrality score from the call graph.

        structural_score is the normalized count of direct callers of this
        chunk's function (GraphRetriever.get_callers(max_depth=1)) — a proxy
        for "how central is this function in the codebase." A function with
        10+ direct callers is treated as fully central (score=1.0); fewer
        callers scale down linearly; a leaf function with no callers (or one
        that isn't in the graph at all, e.g. a class attribute) gets 0.0 and
        is ranked on semantic similarity alone.
        """
        structural_score = 0.0
        graph_distance: Optional[int] = None

        if graph_loaded:
            try:
                caller_result = await asyncio.to_thread(
                    self.graph_retriever.get_callers,
                    function_query=chunk.function_name,
                    collection_name=repo_name,
                    max_depth=1,
                )
                num_direct_callers = len(caller_result.nodes)
                structural_score = min(num_direct_callers / 10.0, 1.0)
                if num_direct_callers > 0:
                    graph_distance = 1
            except Exception as e:
                logger.debug(
                    f"Structural lookup failed for '{chunk.function_name}': {e}"
                )

        hybrid_score = (
            self.semantic_weight * chunk.similarity_score
            + self.structural_weight * structural_score
        )

        return HybridResult(
            chunk_id=chunk.chunk_id,
            file_path=chunk.file_path,
            function_name=chunk.function_name,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            language=chunk.language,
            code=chunk.code_snippet,
            docstring=chunk.docstring,
            semantic_score=chunk.similarity_score,
            structural_score=structural_score,
            hybrid_score=hybrid_score,
            graph_distance=graph_distance,
            metadata={"complexity": chunk.complexity},
        )

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
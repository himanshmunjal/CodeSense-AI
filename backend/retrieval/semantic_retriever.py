"""
retrieval/semantic_retriever.py
════════════════════════════════

WHY THIS FILE EXISTS
────────────────────
After the query is classified (query_classifier.py), we need to actually fetch
relevant code chunks from our vector database. For LOOKUP and SUMMARIZATION
queries — and as the first stage of ANALYTICAL queries — the right tool is
Approximate Nearest Neighbor (ANN) search over the semantic embeddings we
built during ingestion.

This file is responsible for exactly one thing: given a query string, embed
it the same way we embedded code chunks during ingestion, and find the most
semantically similar chunks in Qdrant.

WHY "SEMANTIC" SPECIFICALLY
────────────────────────────
"Semantic" retrieval means we are comparing meaning, not keywords. The query
"where is login handled?" will match a function called `verify_credentials()`
because their embeddings are close in vector space — even though none of the
words overlap.

This is the core advantage over grep or text search:
    - Grep for "login" → misses `authenticate()`, `verify_token()`, etc.
    - Semantic search for "login" → retrieves all of them, ranked by relevance.

The embedding model used here (CodeBERT) was pre-trained on code + docstring
pairs, so it understands that variable names, function signatures, and code
structure carry semantic meaning differently from prose.

HOW EMBEDDING SYMMETRY WORKS
──────────────────────────────
During ingestion: function bodies are embedded with CodeBERT.
During retrieval: the user's query is embedded with the SAME model.

This symmetry is critical. If you embed code with CodeBERT but embed queries
with a generic text model (like ada-002), you are comparing apples to oranges
in the vector space — similarity scores become unreliable.

Both the ingestion pipeline and this file use the same embedding logic
(from embeddings/code_embedder.py). If you ever change the embedding model,
you MUST re-index the entire collection.

METADATA FILTERING
──────────────────
Qdrant supports filtering the search space by metadata payload *before* running
ANN. This means we can answer "find the authentication function, but only in
Python files" by first filtering to language=python, then running ANN over
that subset.

The `filters` parameter in `retrieve()` accepts a dict that gets translated into
Qdrant's filter syntax. The query_classifier may populate `suggested_filters`
from the query text — these are passed through here.

Supported filter keys (defined in indexing/chunk_schema.py):
    language          — "python" | "javascript" | "typescript" | "java"
    file_path_prefix  — partial path string (e.g., "src/auth")
    function_name     — exact function name match
    min_complexity    — minimum cyclomatic complexity (int)
    max_complexity    — maximum cyclomatic complexity (int)

CONFIDENCE THRESHOLD
─────────────────────
Every retrieved result has a cosine similarity score (0.0 to 1.0). We apply a
minimum threshold (from settings.retrieval_confidence_threshold, default 0.65)
and discard results below it.

This threshold is what prevents hallucination: if no result scores above 0.65,
the generator is told the context is insufficient and responds with "I couldn't
find relevant code for this query" rather than making something up.

DEPENDENCIES
────────────
    qdrant-client       — Qdrant Python SDK for ANN search and filtering
    embeddings/code_embedder.py — CodeBERT embedding of the query string
    config              — Qdrant connection settings, threshold, top_k
    loguru              — Structured logging
"""

from dataclasses import dataclass, field
from typing import Optional

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

from config import settings

# The code embedder is imported here (not instantiated) so this module stays
# testable — tests can mock embed_query without importing torch.
from embeddings.code_embedder import CodeEmbedder


# ─────────────────────────────────────────────────────────────────────────────
# Result Schema
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """
    A single code chunk returned by semantic search.

    WHY A DATACLASS:
    Using a dataclass (rather than a plain dict) gives us:
        - Named attribute access (chunk.file_path, not chunk["file_path"])
        - Type hints that IDEs can use for autocomplete
        - Automatic __repr__ for clean debug output
        - Easy to convert to dict with dataclasses.asdict() for JSON serialization

    These objects flow downstream to:
        - reranker.py (which re-scores them with a cross-encoder)
        - generator.py (which formats them into the LLM prompt)
        - The API response (serialized as source citations)

    Fields
    ──────
    chunk_id        : Qdrant point ID. Used to fetch the full chunk body if
                      needed after re-ranking (we don't store the body in the
                      vector payload to keep memory usage low in Qdrant).
    file_path       : Relative path from the repo root. Shown to the user as
                      a citation (e.g., "src/auth/jwt.py").
    function_name   : Name of the function or class this chunk represents.
    start_line      : First line of the chunk in the original file.
    end_line        : Last line of the chunk in the original file.
    language        : Programming language of this chunk.
    code_snippet    : The actual source code text. Populated from the payload
                      stored in Qdrant during indexing.
    docstring       : Docstring of the function, if present. Included in the
                      LLM prompt to give context without sending full bodies.
    similarity_score: Cosine similarity of the chunk's embedding to the query
                      embedding (0.0 to 1.0). Higher = more relevant.
    complexity      : Cyclomatic complexity score. Useful metadata for the user
                      (high complexity functions are harder to change safely).
    """

    chunk_id: str
    file_path: str
    function_name: str
    start_line: int
    end_line: int
    language: str
    code_snippet: str
    docstring: str
    similarity_score: float
    complexity: int = 0
    embedding: Optional[list[float]] = None
    """The "semantic" named vector, populated only when retrieve() is called
    with include_vectors=True. None otherwise — see features/inconsistency_detector.py
    for the one consumer that needs this."""

    @property
    def citation(self) -> str:
        """
        Human-readable citation string for use in LLM prompts and API responses.

        Format: "src/auth/jwt.py::validate_token [lines 42–67]"

        The generator.py uses this format in the system prompt so the LLM
        can include precise citations in its answer. The frontend parses this
        format to highlight the correct lines in the source viewer.
        """
        return f"{self.file_path}::{self.function_name} [lines {self.start_line}–{self.end_line}]"

    @property
    def prompt_block(self) -> str:
        """
        Formatted block for inclusion in the LLM generation prompt.

        WHY FORMAT IT HERE:
        Centralizing the prompt format in the data object means prompt_builder.py
        can just call chunk.prompt_block for each result, rather than duplicating
        formatting logic. If the format needs to change, it changes in one place.

        The format includes the citation header so the LLM "sees" the source
        location alongside the code — this dramatically improves citation accuracy
        in the generated response.
        """
        lines = [
            f"### Source: {self.citation}",
            f"Language: {self.language}",
        ]
        if self.docstring:
            lines.append(f"Docstring: {self.docstring}")
        lines.append(f"```{self.language}")
        lines.append(self.code_snippet)
        lines.append("```")
        return "\n".join(lines)


@dataclass
class SemanticSearchResult:
    """
    Container for the full output of a semantic retrieval operation.

    Wrapping the list of chunks in a result object (rather than returning
    a bare list) lets us carry metadata about the search operation alongside
    the results. This metadata is used for:
        - Logging and monitoring (how many results passed the threshold?)
        - The API response (returning the max score helps the frontend show
          an overall "confidence" indicator for the query)
        - The evaluation pipeline (tracking p50/p90 latency per query type)

    Fields
    ──────
    chunks              : Retrieved chunks that passed the confidence threshold,
                          ordered by similarity_score descending.
    total_candidates    : How many results Qdrant returned before threshold
                          filtering. Useful for debugging low-recall cases.
    max_similarity      : Highest similarity score in the result set. Used by
                          generator.py to decide whether to answer or refuse.
    query_embedding_ms  : Time taken to embed the query (milliseconds).
    qdrant_search_ms    : Time taken for the Qdrant ANN search (milliseconds).
    filters_applied     : The filters that were active during this search, for
                          logging and debugging.
    """

    chunks: list[RetrievedChunk] = field(default_factory=list)
    total_candidates: int = 0
    max_similarity: float = 0.0
    query_embedding_ms: float = 0.0
    qdrant_search_ms: float = 0.0
    filters_applied: Optional[dict] = None

    @property
    def is_confident(self) -> bool:
        """
        Returns True if any result exceeds the confidence threshold.

        This is the key gate that prevents hallucination: generator.py checks
        this before calling the LLM. If False, the system returns a "not found"
        response instead of generating from insufficient context.
        """
        return self.max_similarity >= settings.retrieval_confidence_threshold

    @property
    def is_empty(self) -> bool:
        """Returns True if no chunks passed the confidence threshold."""
        return len(self.chunks) == 0


# ─────────────────────────────────────────────────────────────────────────────
# SemanticRetriever Class
# ─────────────────────────────────────────────────────────────────────────────

class SemanticRetriever:
    """
    Performs ANN (Approximate Nearest Neighbor) search over code embeddings
    stored in Qdrant to retrieve semantically relevant code chunks.

    LIFECYCLE:
        Instantiated once at application startup (module-level singleton below).
        The Qdrant client and CodeEmbedder are initialized once and reused for
        all subsequent retrieve() calls.

    SEARCH FLOW:
        1. Embed the user query using CodeBERT (same model used during indexing).
        2. Build Qdrant filter from the filters dict (if provided).
        3. Run ANN search: Qdrant returns top_k candidates with cosine similarity scores.
        4. Filter out candidates below the confidence threshold.
        5. Map Qdrant search results to RetrievedChunk dataclass instances.
        6. Return SemanticSearchResult with chunks + search metadata.

    USAGE EXAMPLE (called from hybrid_retriever.py):
        retriever = SemanticRetriever()

        result = retriever.retrieve(
            query="Where is JWT token validation handled?",
            collection_name="codesense_tiangolo_fastapi",
            top_k=20,
            filters={"language": "python"}
        )

        if result.is_confident:
            for chunk in result.chunks:
                print(chunk.citation, chunk.similarity_score)
        else:
            print("No confident results found.")
    """

    def __init__(self) -> None:
        """
        Initialize the Qdrant client and the query embedder.

        WHY INITIALIZE BOTH HERE:
        The Qdrant client opens a connection pool to the Qdrant server.
        The CodeEmbedder loads the CodeBERT model into memory (~500MB).
        Both are expensive one-time operations. Doing them at construction
        time (application startup) rather than on the first retrieve() call
        means the first user query has no cold-start penalty.
        """
        logger.info(
            f"Initializing SemanticRetriever | "
            f"qdrant={settings.qdrant_host}:{settings.qdrant_port}"
        )

        # Qdrant client — manages HTTP connection to the Qdrant server.
        # api_key is empty string for local Docker instances (no auth needed).
        self._qdrant = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key or None,
        )

        # CodeBERT embedder — used to embed the user's query.
        # The same model that was used during ingestion (embeddings/code_embedder.py).
        # Embedding symmetry is critical: query and code must be in the same vector space.
        self._embedder = CodeEmbedder()

        logger.info("SemanticRetriever ready.")

    def retrieve(
        self,
        query: str,
        collection_name: str,
        top_k: Optional[int] = None,
        filters: Optional[dict] = None,
        include_vectors: bool = False,
    ) -> SemanticSearchResult:
        """
        Retrieve semantically similar code chunks for a given query.

        This is the primary method called by hybrid_retriever.py for LOOKUP
        and SUMMARIZATION query types, and as the first stage for ANALYTICAL.

        Parameters
        ──────────
        query : str
            The user's natural language question (already classified).
            We embed this with CodeBERT and search for similar vectors.

        collection_name : str
            The Qdrant collection to search. Each ingested repository has its
            own collection (see config.qdrant_collection_name()). This must
            match the collection created during ingestion.

        top_k : int, optional
            Number of candidates to retrieve from Qdrant before threshold
            filtering. Defaults to settings.retrieval_top_k (20).
            The reranker.py then narrows this to settings.reranker_top_n (5).
            We fetch more than we need because the reranker may reorder results
            significantly — fetching only 5 would miss relevant chunks.

        filters : dict, optional
            Metadata filters to narrow the search space before ANN.
            Supported keys: language, file_path_prefix, function_name,
            min_complexity, max_complexity.
            Populated by query_classifier.py's suggested_filters field.

        include_vectors : bool, optional
            When True, the "semantic" named vector is fetched back alongside
            each result and attached to RetrievedChunk.embedding. Needed by
            features/inconsistency_detector.py for ANALYTICAL queries, which
            clusters chunks by embedding — normal LOOKUP/SUMMARIZATION queries
            don't need this and leave it False to save bandwidth.

        Returns
        ───────
        SemanticSearchResult
            Contains retrieved chunks (above confidence threshold) and
            metadata about the search operation. Check result.is_confident
            before passing to the generator.
        """
        import time

        top_k = top_k or settings.retrieval_top_k

        logger.info(
            f"SemanticRetriever.retrieve() | "
            f"collection={collection_name} | top_k={top_k} | filters={filters}"
        )

        # ── Step 1: Embed the query ───────────────────────────────────────────
        # We time this separately from the Qdrant search so we can track
        # where latency is coming from in the evaluation pipeline.

        t0 = time.perf_counter()
        query_vector = self._embedder.embed_query(query)
        embedding_ms = (time.perf_counter() - t0) * 1000

        logger.debug(f"Query embedded | dim={len(query_vector)} | time={embedding_ms:.1f}ms")

        # ── Step 2: Build Qdrant filter ───────────────────────────────────────
        qdrant_filter = self._build_filter(filters) if filters else None

        # ── Step 3: ANN search in Qdrant ─────────────────────────────────────
        try:
            t1 = time.perf_counter()
            search_results = self._qdrant.search(
                collection_name=collection_name,

                # The query vector — Qdrant finds the top_k most similar vectors
                # using cosine similarity (configured during collection creation
                # in indexing/qdrant_client.py).
                query_vector=("semantic", query_vector),

                # Named vector "semantic" — Qdrant supports multiple named vectors
                # per point (we store both "semantic" and "structural" embeddings).
                # Specifying "semantic" here ensures we search the right vector space.

                limit=top_k,
                query_filter=qdrant_filter,

                # with_payload=True fetches all metadata (file_path, function_name,
                # code_snippet, etc.) alongside the similarity score. Without this,
                # we would get only IDs and would need a second lookup to get metadata.
                with_payload=True,

                # with_vectors=False by default — we don't usually need the stored
                # vectors back, just the similarity scores and metadata. Set to
                # True (via include_vectors=) only for ANALYTICAL queries, which
                # need the raw embeddings for k-means clustering.
                with_vectors=["semantic"] if include_vectors else False,

                # score_threshold applies a minimum cosine similarity at the Qdrant
                # layer before results are returned. This is more efficient than
                # fetching all results and filtering in Python, because Qdrant can
                # stop the ANN search early once it runs out of results above the threshold.
                score_threshold=settings.retrieval_confidence_threshold,
            )
            qdrant_ms = (time.perf_counter() - t1) * 1000

        except UnexpectedResponse as e:
            # Collection does not exist, Qdrant is down, etc.
            logger.error(
                f"Qdrant search failed | collection={collection_name} | error={e}"
            )
            # Return an empty result rather than crashing the request.
            return SemanticSearchResult(filters_applied=filters)

        logger.info(
            f"Qdrant search complete | "
            f"candidates={len(search_results)} | time={qdrant_ms:.1f}ms"
        )

        # ── Step 4: Map Qdrant results to RetrievedChunk objects ──────────────
        chunks = self._map_results(search_results)

        # ── Step 5: Build and return the result container ─────────────────────
        max_similarity = max((c.similarity_score for c in chunks), default=0.0)

        result = SemanticSearchResult(
            chunks=chunks,
            total_candidates=len(search_results),
            max_similarity=max_similarity,
            query_embedding_ms=embedding_ms,
            qdrant_search_ms=qdrant_ms,
            filters_applied=filters,
        )

        logger.info(
            f"Semantic retrieval done | "
            f"returned={len(chunks)} chunks | "
            f"max_similarity={max_similarity:.3f} | "
            f"confident={result.is_confident}"
        )

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Private Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_filter(self, filters: dict) -> qdrant_models.Filter:
        """
        Translate a plain dict of filter keys into a Qdrant Filter object.

        WHY A SEPARATE METHOD:
        Qdrant's filter API uses a nested object model (Filter > must > FieldCondition).
        Isolating this translation in a private method keeps retrieve() readable
        and makes it easy to add new filter types without modifying the main logic.

        Supported filter keys and their Qdrant translations
        ────────────────────────────────────────────────────
        "language"          → exact match on payload.language field
        "file_path_prefix"  → prefix match on payload.file_path field
        "function_name"     → exact match on payload.function_name field
        "min_complexity"    → range filter on payload.complexity (gte)
        "max_complexity"    → range filter on payload.complexity (lte)

        Multiple filters are ANDed together (Qdrant's "must" clause).

        Parameters
        ──────────
        filters : dict
            Raw filter dict from the query classifier or the API request.

        Returns
        ───────
        qdrant_models.Filter
            A Qdrant filter object ready to pass to client.search().
        """
        conditions = []

        if "language" in filters:
            # Exact match — payload.language must equal the specified language string.
            conditions.append(
                qdrant_models.FieldCondition(
                    key="language",
                    match=qdrant_models.MatchValue(value=filters["language"])
                )
            )

        if "file_path_prefix" in filters:
            # Prefix match — payload.file_path must start with the given string.
            # Qdrant uses MatchText for substring/prefix matching.
            # This lets users say "in the auth module" → file_path_prefix="auth"
            # and it matches "src/auth/jwt.py", "auth/models.py", etc.
            conditions.append(
                qdrant_models.FieldCondition(
                    key="file_path",
                    match=qdrant_models.MatchText(text=filters["file_path_prefix"])
                )
            )

        if "function_name" in filters:
            # Exact function name match — used when the user names a specific function.
            # Payload key is "entity_name" (see CodeChunk.to_qdrant_payload()) —
            # "function_name" was never written to Qdrant, that name only exists
            # on the RetrievedChunk dataclass this module returns.
            conditions.append(
                qdrant_models.FieldCondition(
                    key="entity_name",
                    match=qdrant_models.MatchValue(value=filters["function_name"])
                )
            )

        if "min_complexity" in filters:
            # Complexity lower bound — useful for analytical queries like
            # "find highly complex functions" (min_complexity=10 or higher).
            conditions.append(
                qdrant_models.FieldCondition(
                    key="cyclomatic_complexity",
                    range=qdrant_models.Range(gte=filters["min_complexity"])
                )
            )

        if "max_complexity" in filters:
            # Complexity upper bound — useful for "find simple utility functions".
            conditions.append(
                qdrant_models.FieldCondition(
                    key="cyclomatic_complexity",
                    range=qdrant_models.Range(lte=filters["max_complexity"])
                )
            )

        # "must" = AND — all conditions must be satisfied.
        # "should" = OR — Qdrant supports this too, but we don't use it here.
        return qdrant_models.Filter(must=conditions)

    def _map_results(
        self, search_results: list
    ) -> list[RetrievedChunk]:
        """
        Convert Qdrant ScoredPoint objects into RetrievedChunk dataclass instances.

        WHY A SEPARATE MAPPING STEP:
        Qdrant's ScoredPoint objects are Qdrant-specific types. Converting them
        to our own RetrievedChunk dataclass decouples the rest of the codebase
        from Qdrant's API — if we ever swap Qdrant for another vector DB, only
        this method needs to change.

        The payload structure here must match what was written during indexing
        in indexing/metadata_builder.py. If you add a field to the chunk schema,
        add it here too.

        Parameters
        ──────────
        search_results : list[ScoredPoint]
            Raw results from qdrant_client.search().

        Returns
        ───────
        list[RetrievedChunk]
            Mapped and validated chunk objects, sorted by similarity_score
            descending (Qdrant returns them sorted, but we sort again to be safe).
        """
        chunks = []

        for point in search_results:
            payload = point.payload or {}

            # .get() with defaults prevents KeyError if a field is missing from
            # the payload (e.g., if the schema evolved after some chunks were indexed).
            # Keys here must match CodeChunk.to_qdrant_payload() in
            # indexing/chunk_schema.py, not this dataclass's own field names.
            chunk = RetrievedChunk(
                chunk_id=str(point.id),
                file_path=payload.get("file_path", "unknown"),
                function_name=payload.get("fully_qualified_name") or payload.get("entity_name", "unknown"),
                start_line=payload.get("start_line", 0),
                end_line=payload.get("end_line", 0),
                language=payload.get("language", "unknown"),
                code_snippet=payload.get("source_code", ""),
                docstring=payload.get("docstring", ""),
                similarity_score=float(point.score),
                complexity=payload.get("cyclomatic_complexity", 0),
                # point.vector is None unless retrieve() was called with
                # include_vectors=True, in which case it's a dict of named
                # vectors ({"semantic": [...]}) since this is a multi-vector
                # collection — see indexing/qdrant_client.py's VECTOR_SEMANTIC.
                embedding=(point.vector or {}).get("semantic") if point.vector else None,
            )
            chunks.append(chunk)

        # Sort descending by similarity score.
        # Qdrant guarantees this order, but we sort again defensively.
        chunks.sort(key=lambda c: c.similarity_score, reverse=True)

        return chunks

    def collection_exists(self, collection_name: str) -> bool:
        """
        Check whether a Qdrant collection exists for the given repo.

        Called by the /query route before attempting retrieval to give a
        clear error message if the repo hasn't been ingested yet, rather
        than a confusing "collection not found" error from Qdrant.

        Parameters
        ──────────
        collection_name : str
            The Qdrant collection name (from config.qdrant_collection_name()).

        Returns
        ───────
        bool
            True if the collection exists and is ready for search.
        """
        try:
            self._qdrant.get_collection(collection_name)
            return True
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

# Shared instance used by hybrid_retriever.py:
#   from retrieval.semantic_retriever import semantic_retriever
#
# The CodeBERT model is loaded once here at import time. This adds ~2–3 seconds
# to application startup but eliminates cold-start latency on the first query.

semantic_retriever = SemanticRetriever()
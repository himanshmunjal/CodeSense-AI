"""
indexing/qdrant_client.py — Qdrant vector database interface for CodeSense.

WHY THIS FILE EXISTS:
─────────────────────
Every chunk that gets parsed and embedded needs to be stored somewhere that
supports fast approximate nearest-neighbor (ANN) search, rich metadata
filtering, and multiple named vectors per record. That "somewhere" is Qdrant.

This file is the single point of contact between the rest of the codebase and
Qdrant. No other module should import qdrant-client directly — they all go
through the QdrantIndexer class defined here. This isolation means:
  1. If we ever swap Qdrant for another vector DB, only this file changes.
  2. All collection-naming, schema, and error-handling logic lives in one place.
  3. Unit tests can mock this class without touching Qdrant infrastructure.

WHAT THIS FILE DOES:
─────────────────────
  - Connects to a running Qdrant instance (local Docker or Qdrant Cloud).
  - Creates a per-repository collection with the correct vector configuration
    (two named vectors: "semantic" and "structural").
  - Upserts code chunks as Qdrant points with their embeddings and metadata.
  - Performs filtered ANN search: semantic search, structural search, or both
    combined (hybrid mode with weighted score fusion).
  - Deletes chunks when a file is re-ingested (stale data removal).
  - Provides health-check and collection-info utilities used by the API layer.

QDRANT CONCEPTS USED HERE:
───────────────────────────
  - Collection  : A named group of vectors, like a table in a relational DB.
                  One collection per repository (e.g. codesense_tiangolo_fastapi).
  - Point       : A single record in a collection. Consists of:
                    • id       — UUID, deterministically derived from file+function
                    • vectors  — dict of named vectors {"semantic": [...], "structural": [...]}
                    • payload  — arbitrary JSON metadata attached to the point
  - Named vectors: Qdrant allows multiple embedding vectors per point with
                  different dimensions. We use this to store both the 768-dim
                  CodeBERT semantic embedding and the 128-dim node2vec structural
                  embedding on the same chunk.
  - Payload filter: Server-side filtering on metadata fields BEFORE the ANN
                  search runs. E.g. "only search chunks where language=python".
                  This is far faster than post-filtering on the client side.
"""

import uuid
import sys
from pathlib import Path
from typing import Optional
from loguru import logger

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

# Allow running this file directly for manual testing (python -m indexing.qdrant_client)
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import settings
from indexing.chunk_schema import CodeChunk


# ─────────────────────────────────────────────────────────────────────────────
# Named vector keys
# These string constants are used everywhere vectors are read or written.
# Changing these would require re-creating all collections, so treat them
# as stable identifiers.
# ─────────────────────────────────────────────────────────────────────────────
VECTOR_SEMANTIC    = "semantic"     # 768-dim CodeBERT embedding of code + docstring
VECTOR_STRUCTURAL  = "structural"   # 128-dim node2vec embedding of call graph position


class QdrantIndexer:
    """
    Manages all interactions with the Qdrant vector database.

    One instance of this class is created at application startup and shared
    across all ingestion and retrieval operations via FastAPI dependency
    injection.

    Each repository gets its own Qdrant collection. The collection name is
    derived deterministically from the repo owner + name using the helper
    in config.py (e.g. "codesense_tiangolo_fastapi").

    Example usage:
        indexer = QdrantIndexer()
        indexer.ensure_collection("tiangolo", "fastapi")
        indexer.upsert_chunks("tiangolo", "fastapi", chunks)
        results = indexer.search_semantic("tiangolo", "fastapi", query_vector, top_k=20)
    """

    def __init__(self) -> None:
        """
        Initialises the Qdrant client using host/port from settings.

        Why separate host+port instead of a URL string?
        QdrantClient accepts both forms. Using host+port makes it easier to
        override individual components (e.g. change just the port in tests)
        without re-parsing a URL string.

        The api_key is only required for Qdrant Cloud. For the local Docker
        instance used in development, it is left empty and ignored by the client.
        """
        logger.info(
            f"Connecting to Qdrant at {settings.qdrant_host}:{settings.qdrant_port}"
        )
        self._client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            # api_key is None when empty string — Qdrant client treats None as
            # "no authentication required", which is correct for local Docker.
            api_key=settings.qdrant_api_key or None,
            # Timeout in seconds for all HTTP calls to Qdrant.
            # 30s is generous for local use; reduce to ~5s in production.
            timeout=30,
        )
        logger.info("Qdrant client initialised successfully.")

    # ─────────────────────────────────────────────────────────────────────
    # Collection Management
    # ─────────────────────────────────────────────────────────────────────

    def ensure_collection(self, owner: str, repo: str) -> str:
        """
        Creates a Qdrant collection for a repository if it does not exist.
        Returns the collection name regardless of whether it was just created
        or already existed.

        WHY THIS DESIGN (idempotent create):
        Called at the start of every ingestion job. Using "create if not exists"
        rather than "always create" means re-ingesting an existing repo does not
        wipe the collection — we delete only the specific stale chunks via
        delete_chunks_for_file() before upserting fresh ones.

        COLLECTION SCHEMA DECISIONS:
        - Two named vectors: "semantic" (settings.semantic_embedding_dim —
          384 for the default bge-small-en-v1.5 embedder) and "structural"
          (128-dim).
          Having two separate named vectors per point — rather than
          concatenating them into one long vector — lets us search against
          either embedding independently, which is needed for the semantic-only
          vs. structural-only experiments in the notebooks.
        - Distance metric: COSINE for both.
          Cosine similarity is standard for embedding-based retrieval because
          it measures the angle between vectors (directional similarity),
          ignoring magnitude. This is what we want for code: two functions
          that do similar things should have similar directions in embedding
          space even if one is 10 lines and the other is 100 lines.
        - on_disk_payload=True: stores metadata payload on disk instead of RAM.
          Payloads (file path, function name, etc.) can be large compared to
          vectors. Keeping them on disk reduces memory footprint with minimal
          latency impact since payloads are only fetched after the ANN search
          has already narrowed results to a small set.

        Args:
            owner: GitHub repository owner (e.g. "tiangolo").
            repo:  GitHub repository name  (e.g. "fastapi").

        Returns:
            The Qdrant collection name string.
        """
        collection_name = settings.qdrant_collection_name(owner, repo)

        # Check if the collection already exists to avoid redundant work.
        existing = [c.name for c in self._client.get_collections().collections]
        if collection_name in existing:
            logger.info(f"Collection '{collection_name}' already exists — skipping creation.")
            return collection_name

        logger.info(f"Creating new Qdrant collection: '{collection_name}'")
        self._client.create_collection(
            collection_name=collection_name,
            vectors_config={
                # Semantic vector: the configured embedding model (bge-small
                # by default) produces settings.semantic_embedding_dim-
                # dimensional embeddings capturing *meaning* — what a
                # function does. NOTE: this must be recreated (not just
                # re-upserted into) if you change to a model with a
                # different output dimension — Qdrant rejects mismatched
                # vector sizes on upsert rather than resizing in place.
                VECTOR_SEMANTIC: qmodels.VectorParams(
                    size=settings.semantic_embedding_dim,
                    distance=qmodels.Distance.COSINE,
                ),
                # Structural vector: node2vec produces 128-dimensional embeddings.
                # These capture the *position* of a function in the call graph —
                # which functions it calls and which call it.
                VECTOR_STRUCTURAL: qmodels.VectorParams(
                    size=settings.structural_embedding_dim,  # 128
                    distance=qmodels.Distance.COSINE,
                ),
            },
            # Store payload metadata on disk rather than in RAM.
            # At scale (10k+ functions per repo) this matters significantly.
            on_disk_payload=True,
        )

        # Create payload indexes on frequently-filtered fields.
        # WITHOUT these indexes, Qdrant scans all payloads on every filtered
        # search — O(n). WITH indexes, filtering is O(log n).
        # We index: file_path (for "re-index changed file" deletes),
        # language (for "Python only" queries), function_name (for lookup).
        for field, field_type in [
            ("file_path",     qmodels.PayloadSchemaType.KEYWORD),
            ("language",      qmodels.PayloadSchemaType.KEYWORD),
            ("function_name", qmodels.PayloadSchemaType.KEYWORD),
        ]:
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=field_type,
            )
            logger.debug(f"Created payload index on '{field}' in '{collection_name}'.")

        logger.info(f"Collection '{collection_name}' created with semantic + structural vectors.")
        return collection_name

    def collection_info(self, owner: str, repo: str) -> dict:
        """
        Returns summary statistics about a repository's collection.

        WHY THIS EXISTS:
        Used by the /ingest status endpoint to report back to the user:
        how many chunks are indexed, when the last update happened, etc.
        Also useful for debugging — if the count is 0 after ingestion,
        something went wrong in the upsert pipeline.

        Returns a plain dict so the API layer can serialise it directly
        into a JSON response without further processing.
        """
        collection_name = settings.qdrant_collection_name(owner, repo)
        try:
            info = self._client.get_collection(collection_name)
            return {
                "collection_name":  collection_name,
                "total_points":     info.points_count,
                "indexed_vectors":  info.indexed_vectors_count,
                "status":           info.status,
            }
        except UnexpectedResponse as e:
            # Collection does not exist yet — not an error, just not ingested.
            logger.warning(f"Collection '{collection_name}' not found: {e}")
            return {"collection_name": collection_name, "total_points": 0, "status": "not_found"}

    def delete_collection(self, owner: str, repo: str) -> None:
        """
        Completely removes a repository's collection from Qdrant.

        WHY THIS EXISTS:
        Used when a user wants to fully re-index a repo from scratch,
        or when a repo is deleted from CodeSense. More reliable than
        deleting all points individually when the goal is a clean slate.
        """
        collection_name = settings.qdrant_collection_name(owner, repo)
        self._client.delete_collection(collection_name)
        logger.info(f"Deleted collection '{collection_name}'.")

    # ─────────────────────────────────────────────────────────────────────
    # Upserting Chunks
    # ─────────────────────────────────────────────────────────────────────

    def upsert_chunks(
        self,
        owner:  str,
        repo:   str,
        chunks: list["CodeChunk"],
    ) -> None:
        """
        Upserts a list of CodeChunk objects into the repository's collection.

        WHY UPSERT INSTEAD OF INSERT:
        "Upsert" means insert-or-update. If a point with the same ID already
        exists, it is overwritten. This is correct behaviour for re-ingestion:
        when a file changes, we delete its old chunks (via delete_chunks_for_file)
        and upsert fresh ones. Using plain insert would raise errors if the
        IDs somehow collide.

        WHY BATCH IN GROUPS OF 100:
        Qdrant's HTTP API has a practical limit on request body size.
        A single 384-dim semantic vector is 384 * 4 bytes = ~1.5KB. 100 chunks with
        both semantic + structural vectors = ~(384+128) * 4 * 100 ≈ 200KB
        per batch, comfortably under HTTP limits. Batching also means a
        single network failure only loses one batch, not the entire ingestion.

        WHY DETERMINISTIC POINT IDs:
        Each point's UUID is derived from the chunk's chunk_id (repo + file_path
        + entity name + start line).
        Deterministic IDs mean upserting the same chunk twice is idempotent —
        the second upsert simply overwrites the first with identical data.
        This is safer than using random UUIDs, which would create duplicates
        on re-ingestion.

        Args:
            owner:  Repository owner.
            repo:   Repository name.
            chunks: List of CodeChunk objects (see indexing/chunk_schema.py).
                    Each chunk must have .semantic_vector and
                    .structural_vector already populated.
        """
        collection_name = settings.qdrant_collection_name(owner, repo)

        # Filter out any chunks that are missing embeddings.
        # This can happen if the embedding step failed for a particular chunk
        # (e.g. CodeBERT OOM on a very long function). We log a warning but
        # continue rather than aborting the entire ingestion.
        valid_chunks = []
        for chunk in chunks:
            if chunk.semantic_vector is None:
                logger.warning(
                    f"Skipping chunk '{chunk.entity_name}' in {chunk.file_path} "
                    f"— missing semantic embedding."
                )
                continue
            if chunk.structural_vector is None:
                logger.warning(
                    f"Skipping chunk '{chunk.entity_name}' in {chunk.file_path} "
                    f"— missing structural embedding."
                )
                continue
            valid_chunks.append(chunk)

        if not valid_chunks:
            logger.warning("No valid chunks to upsert — all were missing embeddings.")
            return

        # Batch the upsert into groups of 100.
        batch_size = 100
        total_batches = (len(valid_chunks) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            batch = valid_chunks[batch_idx * batch_size : (batch_idx + 1) * batch_size]

            points = [
                qmodels.PointStruct(
                    # Deterministic UUID from content identity.
                    # uuid.uuid5 generates a UUID from a namespace + string,
                    # always producing the same UUID for the same input.
                    # chunk_id (repo + file + name + start line) is used rather
                    # than file + name alone: two entities with the same name in
                    # one file (every Python class's __init__, Java overloads)
                    # would otherwise map to one point and overwrite each other.
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                    # Named vectors — each key must match the collection's
                    # vector config defined in ensure_collection().
                    vector={
                        VECTOR_SEMANTIC:   chunk.semantic_vector,
                        VECTOR_STRUCTURAL: chunk.structural_vector,
                    },
                    # Payload is the metadata that gets returned with search
                    # results and used for filtering. CodeChunk.to_qdrant_payload()
                    # is the single source of truth for payload shape — see
                    # indexing/chunk_schema.py for exactly what fields this includes
                    # (it deliberately excludes the vectors, which are stored
                    # separately above, and flattens enums/datetimes to primitives).
                    payload=chunk.to_qdrant_payload(),
                )
                for chunk in batch
            ]

            self._client.upsert(
                collection_name=collection_name,
                points=points,
                # wait=True means the call blocks until Qdrant has confirmed
                # the upsert is persisted. Slower but safer for ingestion
                # pipelines where we need to know the data is actually stored.
                wait=True,
            )
            logger.info(
                f"Upserted batch {batch_idx + 1}/{total_batches} "
                f"({len(batch)} chunks) into '{collection_name}'."
            )

        logger.info(
            f"Upsert complete: {len(valid_chunks)} chunks indexed into '{collection_name}'."
        )

    def delete_chunks_for_file(self, owner: str, repo: str, file_path: str) -> None:
        """
        Deletes all indexed chunks that belong to a specific file.

        WHY THIS EXISTS:
        The change-detection layer (ingestion/change_detector.py) identifies
        which files changed since the last ingestion using git diff. For each
        changed file, we call this method to remove its old chunks before
        upserting fresh ones. This keeps the index in sync with the actual
        codebase without requiring a full re-index.

        WHY A PAYLOAD FILTER DELETE:
        Qdrant supports deleting points by a filter on their payload fields.
        Since we indexed file_path as a keyword payload field (in ensure_collection),
        this delete is efficient — O(log n) — rather than a full collection scan.

        Args:
            owner:     Repository owner.
            repo:      Repository name.
            file_path: Relative path of the file inside the repo
                       (e.g. "fastapi/routing.py").
        """
        collection_name = settings.qdrant_collection_name(owner, repo)

        deleted = self._client.delete(
            collection_name=collection_name,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="file_path",
                            match=qmodels.MatchValue(value=file_path),
                        )
                    ]
                )
            ),
        )
        logger.info(
            f"Deleted stale chunks for '{file_path}' from '{collection_name}'. "
            f"Operation status: {deleted.status}"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Search Methods
    # ─────────────────────────────────────────────────────────────────────

    def search_semantic(
        self,
        owner:        str,
        repo:         str,
        query_vector: list[float],
        top_k:        int = 20,
        language:     Optional[str] = None,
        file_path:    Optional[str] = None,
    ) -> list[dict]:
        """
        Performs approximate nearest-neighbor search using the semantic vector.

        WHY SEMANTIC SEARCH:
        Used for "lookup" and "summarization" query types where the user is
        asking about what code *does* — e.g. "find the authentication function".
        CodeBERT embeddings capture meaning, so semantically similar functions
        will be close in the 768-dim vector space even if they use different
        variable names or coding styles.

        WHY OPTIONAL FILTERS:
        Payload filters are applied server-side in Qdrant BEFORE the ANN
        search. This is called "pre-filtering" and is much more efficient than
        fetching all results and filtering in Python. Use language= to restrict
        search to Python-only files, or file_path= to restrict to a specific
        file (useful for impact analysis).

        Args:
            owner:        Repository owner.
            repo:         Repository name.
            query_vector: 768-dimensional semantic embedding of the user's query.
                          Produced by the same CodeBERT model used for indexing.
            top_k:        Number of nearest neighbours to return. Default 20
                          because the re-ranker will later narrow this to 5.
            language:     Optional filter — only return chunks in this language.
            file_path:    Optional filter — only search within this file.

        Returns:
            List of dicts, each containing the chunk payload + similarity score.
            Sorted by score descending (most relevant first).
        """
        collection_name = settings.qdrant_collection_name(owner, repo)

        # Build the filter only if at least one filter argument was provided.
        # An empty Filter() object would cause a Qdrant API error.
        query_filter = self._build_filter(language=language, file_path=file_path)

        results = self._client.search(
            collection_name=collection_name,
            query_vector=(VECTOR_SEMANTIC, query_vector),
            limit=top_k,
            query_filter=query_filter,
            # with_payload=True means the full metadata payload is returned
            # alongside each hit. Required so callers can render source citations.
            with_payload=True,
            # with_vectors=False — we do not need the raw embedding vectors
            # back in the response. This reduces response size significantly.
            with_vectors=False,
        )

        return self._format_results(results)

    def search_structural(
        self,
        owner:        str,
        repo:         str,
        query_vector: list[float],
        top_k:        int = 20,
        language:     Optional[str] = None,
    ) -> list[dict]:
        """
        Performs ANN search using the structural (call-graph) vector.

        WHY STRUCTURAL SEARCH:
        Used for "relational" query types — e.g. "what functions are similar in
        structure to authenticate_user?". node2vec embeddings capture a function's
        neighbourhood in the call graph: two functions that are called by the same
        callers or that call the same callees will have similar structural vectors,
        even if they do completely different things semantically.

        In practice, structural search is less frequently used as the primary
        retrieval method and more often used as a complementary signal in
        hybrid search (see search_hybrid below).

        Args:
            owner:        Repository owner.
            repo:         Repository name.
            query_vector: 128-dimensional structural embedding of a reference
                          function node. Produced by node2vec in graph_embedder.py.
            top_k:        Number of nearest neighbours to return.
            language:     Optional language filter.

        Returns:
            List of dicts with chunk payload + similarity score.
        """
        collection_name = settings.qdrant_collection_name(owner, repo)
        query_filter = self._build_filter(language=language)

        results = self._client.search(
            collection_name=collection_name,
            query_vector=(VECTOR_STRUCTURAL, query_vector),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        )

        return self._format_results(results)

    def search_hybrid(
        self,
        owner:              str,
        repo:               str,
        semantic_vector:    list[float],
        structural_vector:  list[float],
        top_k:              int = 20,
        semantic_weight:    float = 0.7,
        structural_weight:  float = 0.3,
        language:           Optional[str] = None,
    ) -> list[dict]:
        """
        Performs hybrid search combining semantic + structural similarity scores.

        WHY HYBRID SEARCH:
        Neither semantic nor structural search alone is sufficient for all
        query types. Consider the query "find all functions that handle errors
        the same way as process_payment":
          - Semantic search finds functions that TALK about payment/error handling.
          - Structural search finds functions that LIVE in the same call graph
            neighbourhood as process_payment.
          - Hybrid search combines both signals to find functions that are both
            semantically about error handling AND structurally similar.

        The default weights (0.7 semantic, 0.3 structural) come from the Day 8
        experiment in notebooks/hybrid_retrieval_experiment.ipynb, where this
        split produced the best Precision@5 on the evaluation set. The weights
        are configurable via settings so you can tune them without code changes.

        IMPLEMENTATION: Two separate ANN searches, then score fusion.
        Qdrant does not natively support multi-vector query fusion in a single
        API call (as of v1.9). We therefore run two searches and merge results
        in Python using a weighted sum. The union of both result sets is scored;
        chunks that only appear in one result set get a score of 0 for the
        missing modality.

        Args:
            owner:              Repository owner.
            repo:               Repository name.
            semantic_vector:    768-dim CodeBERT query embedding.
            structural_vector:  128-dim node2vec query embedding.
            top_k:              Number of final results to return after fusion.
            semantic_weight:    Weight for semantic score (default 0.7).
            structural_weight:  Weight for structural score (default 0.3).
            language:           Optional language filter applied to both searches.

        Returns:
            List of dicts sorted by fused score descending, length = top_k.
        """
        # Run both searches independently, fetching 2x top_k each so that
        # after fusion we still have enough candidates to return top_k.
        sem_results  = self.search_semantic(
            owner, repo, semantic_vector,   top_k=top_k * 2, language=language
        )
        struct_results = self.search_structural(
            owner, repo, structural_vector, top_k=top_k * 2, language=language
        )

        # Build a score map keyed by point ID.
        # Each entry: {"semantic": float, "structural": float, "payload": dict}
        score_map: dict[str, dict] = {}

        for hit in sem_results:
            score_map[hit["id"]] = {
                "semantic":   hit["score"],
                "structural": 0.0,          # Will be filled in if this ID also appears in struct results
                "payload":    hit["payload"],
            }

        for hit in struct_results:
            if hit["id"] in score_map:
                # This chunk appeared in both result sets — update its structural score.
                score_map[hit["id"]]["structural"] = hit["score"]
            else:
                # This chunk only appeared in structural results.
                score_map[hit["id"]] = {
                    "semantic":   0.0,
                    "structural": hit["score"],
                    "payload":    hit["payload"],
                }

        # Compute fused score and sort.
        fused = []
        for point_id, scores in score_map.items():
            fused_score = (
                semantic_weight   * scores["semantic"] +
                structural_weight * scores["structural"]
            )
            fused.append({
                "id":      point_id,
                "score":   round(fused_score, 6),
                "payload": scores["payload"],
            })

        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[:top_k]

    # ─────────────────────────────────────────────────────────────────────
    # Private Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _build_filter(
        self,
        language:  Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> Optional[qmodels.Filter]:
        """
        Builds a Qdrant Filter object from optional filter arguments.

        WHY A HELPER:
        Multiple search methods need to build filters from the same set of
        optional parameters. Centralising this logic avoids repetition and
        ensures all methods use consistent filter construction.

        Returns None if no filter arguments were provided, which tells Qdrant
        to search the full collection without filtering.

        Args:
            language:  If provided, restrict search to chunks of this language.
            file_path: If provided, restrict search to chunks from this file.

        Returns:
            A qmodels.Filter object, or None if no filters are needed.
        """
        conditions = []

        if language:
            conditions.append(
                qmodels.FieldCondition(
                    key="language",
                    match=qmodels.MatchValue(value=language),
                )
            )

        if file_path:
            conditions.append(
                qmodels.FieldCondition(
                    key="file_path",
                    match=qmodels.MatchValue(value=file_path),
                )
            )

        if not conditions:
            return None  # No filter — search entire collection

        # "must" is Qdrant's equivalent of SQL AND — all conditions must be satisfied.
        return qmodels.Filter(must=conditions)

    @staticmethod
    def _format_results(raw_results: list) -> list[dict]:
        """
        Converts Qdrant ScoredPoint objects into plain dicts.

        WHY CONVERT TO DICTS:
        Qdrant returns ScoredPoint objects (Pydantic models from the
        qdrant-client library). Converting to plain dicts here means the
        rest of the codebase — retrieval layer, generation layer, API routes —
        does not need to import or know about qdrant-client types. This keeps
        the dependency boundary clean.

        Args:
            raw_results: List of qdrant_client.models.ScoredPoint objects.

        Returns:
            List of dicts: [{"id": str, "score": float, "payload": dict}, ...]
        """
        return [
            {
                "id":      str(hit.id),
                "score":   round(hit.score, 6),
                "payload": hit.payload or {},
            }
            for hit in raw_results
        ]

    # ─────────────────────────────────────────────────────────────────────
    # Health Check
    # ─────────────────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """
        Verifies that the Qdrant instance is reachable and responding.

        WHY THIS EXISTS:
        Called at FastAPI startup (in api/main.py lifespan handler) to fail
        fast if Qdrant is down, rather than letting the app start and then
        fail on the first actual request. Also used by the /health endpoint
        to expose infrastructure status to monitoring tools.

        Returns:
            True if Qdrant responds to a collections list request, False otherwise.
        """
        try:
            self._client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False
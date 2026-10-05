"""
================================================================================
api/routes/summarize.py — Codebase Summarization Endpoints
================================================================================

WHY THIS FILE EXISTS:
    The project spec defines summarization as one of the four distinct query
    types ("Explain what the data pipeline does"). While query.py CAN handle
    summarization questions through its query classifier, this file exists
    as a SEPARATE, purpose-built endpoint for a different reason: scope.

    A summarization request is fundamentally different in shape from a
    point-lookup question:
        - query.py:     "Where is the login function?" → 1-5 chunks, 1 answer.
        - summarize.py:  "Explain the auth module."     → potentially dozens
                          of files, multiple synthesis passes, and a
                          structured multi-section output (overview, key
                          components, data flow, entry points).

    Cramming this into the generic /query endpoint would force every query
    response to support fields that only summarization needs (e.g., a list
    of "key components" or "module overview" sections). Splitting it out
    keeps both endpoints' response schemas honest and avoids bloating
    QueryResponse with fields that are almost always null.

WHY THIS DIFFERS FROM query.py's SUMMARIZATION QUERY_TYPE:
    The query_classifier in retrieval/query_classifier.py can still classify
    a question typed into the chat box as "summarization" and route it through
    the standard query pipeline for a quick, single-paragraph answer.

    This file is for the EXPLICIT "Summarize this file/module/directory" action
    — e.g., a button in the frontend's file tree ("Summarize this file") or
    a deliberate request for a structured overview rather than a quick answer.
    It supports inputs that /query does not: an entire file path or directory
    path, rather than only a natural-language question.

ENDPOINTS IN THIS FILE:
    POST /api/v1/summarize
        → Summarizes a specific file, directory, or the whole indexed
          repository. Returns a structured, multi-section summary.

    POST /api/v1/summarize/module
        → Specialized variant: summarizes a logical "module" by retrieving
          all chunks under a directory prefix and synthesizing across them.
          This is what backs a "summarize this folder" action in the
          frontend's file tree.

HOW IT FITS IN THE PIPELINE:
    HTTP POST /summarize
        → Validate request (file_path OR directory_path, not a free-text question)
            → Retrieve ALL chunks under that path from Qdrant (not top-K — exhaustive)
                → If too many chunks: cluster + sample representative chunks
                    → Multi-chunk synthesis prompt to the LLM
                        → Structured summary response

WHY EXHAUSTIVE RETRIEVAL INSTEAD OF TOP-K SEMANTIC SEARCH:
    A standard query asks "find the most relevant 5 chunks for this question."
    Summarization asks "tell me about everything in this scope." These are
    different retrieval needs — top-K ANN search would miss less "central"
    but still important functions in the module (e.g., a small utility
    function that's rarely similar to other code but is still part of the
    module's behavior). We instead use a metadata filter (file_path prefix
    match) to pull ALL chunks under that scope, then sample/cluster if there
    are too many to fit in the LLM's context window.
================================================================================
"""

from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from loguru import logger
import asyncio
import time

from config import get_settings, Settings
from generation.generator import generate_summary
from generation.response_schema import SummaryGenerationResponse


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER INSTANCE
# ─────────────────────────────────────────────────────────────────────────────

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# WHY THIS LIMIT EXISTS:
#   The LLM has a large context window, but stuffing hundreds of full function
#   bodies into one prompt is slow and expensive, and the model's attention
#   degrades over very long contexts ("lost in the middle" effect — models
#   are measurably worse at using information buried in the middle of a long
#   prompt). Capping at 40 chunks keeps the prompt focused and keeps latency
#   within a reasonable range for what is already an expensive, multi-second
#   operation.
MAX_CHUNKS_FOR_SUMMARY = 40

# WHY WE CLUSTER WHEN OVER THE LIMIT:
#   If a module has 200 functions, we can't send all 200 to the LLM.
#   Instead, we use k-means clustering (the same technique used for
#   inconsistency detection in features/inconsistency_detector.py) to group
#   similar chunks, then take the single most "central" chunk from each
#   cluster as a representative. This gives the LLM a diverse, representative
#   sample of the module's behavior rather than an arbitrary truncation
#   (e.g., just the first 40 alphabetically) which could miss whole
#   sub-components.
CLUSTERING_THRESHOLD = MAX_CHUNKS_FOR_SUMMARY


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST & RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class SummarizeRequest(BaseModel):
    """
    Schema for POST /summarize.

    Exactly one of file_path or directory_path must be provided —
    enforced by the validator below. If neither is given, the entire
    repository is summarized (a high-level architectural overview).

    Example — summarize a single file:
        {
            "repo_url": "https://github.com/tiangolo/fastapi",
            "file_path": "fastapi/security/oauth2.py"
        }

    Example — summarize a directory:
        {
            "repo_url": "https://github.com/tiangolo/fastapi",
            "directory_path": "fastapi/security/"
        }
    """

    repo_url: str = Field(
        ...,
        description="GitHub URL of the repository to summarize. Must already be ingested.",
        examples=["https://github.com/tiangolo/fastapi"],
    )

    file_path: Optional[str] = Field(
        default=None,
        description=(
            "Exact relative file path to summarize. "
            "Mutually exclusive with directory_path."
        ),
        examples=["fastapi/security/oauth2.py"],
    )

    directory_path: Optional[str] = Field(
        default=None,
        description=(
            "Directory prefix to summarize. All files whose path starts with "
            "this prefix are included. Mutually exclusive with file_path."
        ),
        examples=["fastapi/security/", "src/utils/"],
    )

    focus_hint: Optional[str] = Field(
        default=None,
        description=(
            "Optional natural-language hint to focus the summary on a "
            "specific aspect, e.g. 'focus on error handling' or "
            "'focus on the public API surface'. If omitted, the summary "
            "covers general purpose, structure, and key components."
        ),
        max_length=200,
    )

    @field_validator("directory_path")
    @classmethod
    def validate_exclusive_scope(cls, v, info):
        """
        WHY THIS VALIDATOR EXISTS:
            file_path and directory_path represent two different retrieval
            strategies (exact match vs. prefix match). Allowing both to be
            set simultaneously would create ambiguity about which scope
            the user actually wants summarized. We fail fast with a clear
            422 error rather than silently picking one and ignoring the other.

        NOTE ON PYDANTIC v2 CROSS-FIELD VALIDATION:
            `info.data` gives access to previously-validated fields in this
            model. Since field_path is declared before directory_path in
            the class body, it's already validated and available here.
        """
        file_path = info.data.get("file_path")
        if v is not None and file_path is not None:
            raise ValueError(
                "Provide either file_path or directory_path, not both. "
                "Omit both to summarize the entire repository."
            )
        return v


class KeyComponent(BaseModel):
    """
    A single notable component (function, class, or sub-module) identified
    during summarization. Used to populate the structured "Key Components"
    section of the summary, which the frontend renders as a clickable list.
    """
    name: str = Field(..., description="Function, class, or component name.")
    file_path: str = Field(..., description="File where this component is defined.")
    start_line: int = Field(..., description="Starting line number (1-indexed).")
    role: str = Field(
        ...,
        description="One-sentence description of this component's role in the module.",
    )


class SummarizeResponse(BaseModel):
    """
    Schema for POST /summarize response.

    The summary is intentionally structured into sections rather than one
    long paragraph — this lets the frontend render a scannable overview
    instead of a wall of text, which matters for a developer tool where
    people are skimming, not reading linearly.
    """

    scope: str = Field(
        ...,
        description="What was summarized: a file path, directory path, or 'entire repository'.",
    )
    overview: str = Field(
        ...,
        description="A 2-4 sentence high-level explanation of what this code does and why it exists.",
    )
    key_components: list[KeyComponent] = Field(
        default_factory=list,
        description="The most important functions/classes in this scope, with their roles.",
    )
    data_flow: Optional[str] = Field(
        default=None,
        description=(
            "Description of how data moves through this scope, if applicable. "
            "Null for very small scopes (e.g., a single utility file) where "
            "there isn't a meaningful flow to describe."
        ),
    )
    entry_points: list[str] = Field(
        default_factory=list,
        description=(
            "Functions or classes that are likely called from OUTSIDE this scope "
            "(i.e., the public-facing surface of this module). Identified using "
            "call graph data — these are nodes with incoming edges from other modules."
        ),
    )
    files_analyzed: int = Field(
        ...,
        description="Number of files included in this summary.",
    )
    chunks_analyzed: int = Field(
        ...,
        description="Number of code chunks (functions/classes) included in this summary.",
    )
    was_sampled: bool = Field(
        ...,
        description=(
            "True if the scope had more chunks than MAX_CHUNKS_FOR_SUMMARY and "
            "clustering/sampling was applied. False if every chunk in scope was "
            "analyzed directly. Shown in the UI as a transparency note: "
            "'Summary based on a representative sample of 40 of 212 functions.'"
        ),
    )
    repo_url: str
    latency_ms: float = Field(
        ...,
        description="Total server-side processing time in milliseconds.",
    )


class SummarizeModuleRequest(BaseModel):
    """
    Schema for POST /summarize/module.

    Simpler than SummarizeRequest — module summaries are always scoped to
    a directory and always include a comparison against the rest of the
    repo (e.g., "this module is the only one using camelCase").
    """
    repo_url: str = Field(..., description="GitHub URL of the repository.")
    module_path: str = Field(
        ...,
        description="Directory path representing the module. E.g. 'src/payments/'.",
        examples=["src/payments/", "fastapi/middleware/"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — VERIFY REPOSITORY IS INDEXED
# ─────────────────────────────────────────────────────────────────────────────

def _verify_repo_indexed(redis, owner: str, repo: str) -> dict:
    """
    WHY THIS DUPLICATES THE HELPER IN query.py:
        Each route file is kept self-contained on purpose — see the note in
        query.py's equivalent helper. Importing across route modules would
        create a dependency between sibling files that should otherwise be
        independent (e.g., a future refactor of query.py's internals
        shouldn't risk breaking summarize.py). The few lines of duplication
        here are a worthwhile trade for that isolation.
    """
    import json
    key = f"indexed_repo:{owner}:{repo}"
    raw = redis.get(key)
    if not raw:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Repository '{owner}/{repo}' has not been ingested yet. "
                f"Run POST /api/v1/ingest with the repository URL first."
            ),
        )
    return json.loads(raw)


def _parse_github_url(repo_url: str) -> tuple[str, str]:
    """Extracts (owner, repo) from a GitHub URL. See query.py for full docs."""
    parts = repo_url.rstrip("/").split("/")
    return parts[-2], parts[-1]


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — RETRIEVE ALL CHUNKS IN SCOPE
# ─────────────────────────────────────────────────────────────────────────────

async def _retrieve_chunks_in_scope(
    qdrant_client,
    collection_name: str,
    file_path: Optional[str],
    directory_path: Optional[str],
) -> list[dict]:
    """
    WHY THIS HELPER EXISTS:
        Summarization needs EVERY chunk in a given scope, not just the top-K
        most similar to a query embedding. Qdrant's scroll API is built
        exactly for this: it paginates through all points matching a filter,
        without requiring a query vector at all.

    HOW IT WORKS:
        - If file_path is given: filter where payload.file_path == file_path exactly.
        - If directory_path is given: filter where payload.file_path starts with
          that prefix (a "text" match in Qdrant, since exact prefix filters
          aren't natively supported — text match approximates it well enough
          for this use case given our file_path values are stored lowercase
          and slash-delimited).
        - If neither is given: no filter — scroll the entire collection
          (used for whole-repository summaries).

    WHY scroll() INSTEAD OF search():
        Qdrant's search() requires a query vector and returns the top-K by
        similarity. scroll() is a cursor-based "give me everything matching
        this filter" API — exactly the exhaustive retrieval semantics we
        need here. Reaching for search() with a dummy vector would be a
        misuse of the API and could silently truncate results based on
        unrelated similarity scoring.
    """
    qdrant_filter = None
    if file_path:
        qdrant_filter = {
            "must": [{"key": "file_path", "match": {"value": file_path}}]
        }
    elif directory_path:
        qdrant_filter = {
            "must": [{"key": "file_path", "match": {"text": directory_path.rstrip("/")}}]
        }

    all_chunks: list[dict] = []
    offset = None

    # Qdrant's scroll() is paginated — we loop until offset comes back None,
    # which signals there are no more points to fetch.
    while True:
        points, offset = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=qdrant_filter,
            limit=100,            # Fetch in batches of 100 for memory efficiency
            offset=offset,
            with_payload=True,
            with_vectors=False,   # We don't need vectors for summarization, only metadata + text
        )
        for point in points:
            all_chunks.append(point.payload)
        if offset is None:
            break

    return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — SAMPLE REPRESENTATIVE CHUNKS VIA CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def _sample_representative_chunks(chunks: list[dict], target_count: int) -> list[dict]:
    """
    WHY THIS HELPER EXISTS:
        When a scope has more chunks than fit comfortably in an LLM prompt,
        we need to pick a SMALLER set that still represents the diversity of
        the module — not just the first N alphabetically, which could
        accidentally summarize only one sub-component and miss the rest.

    HOW IT WORKS:
        1. Re-embed all chunk snippets (or reuse cached embeddings if available
           in the payload — see indexing/chunk_schema.py for the embedding
           cache fields).
        2. Run k-means with k = target_count clusters using scikit-learn.
        3. From each cluster, select the single chunk closest to the cluster
           centroid — this is the "most representative" member of that group.
        4. Return exactly target_count chunks, one per cluster.

    WHY THIS APPROACH MIRRORS INCONSISTENCY DETECTION:
        features/inconsistency_detector.py uses the same k-means-on-embeddings
        technique to group similar implementations and flag outliers. Here we
        use the same clustering machinery for the opposite purpose: instead of
        flagging the outlier, we want one representative per cluster so the
        summary covers the full breadth of behavior in the module.

    NOTE:
        This function is intentionally synchronous (not async) — k-means on
        a few hundred low-dimensional vectors completes in milliseconds and
        does not justify the overhead of an async wrapper.
    """
    from sklearn.cluster import KMeans
    import numpy as np

    # Use the cached semantic embedding if present in the payload; this avoids
    # paying for a redundant CodeBERT inference pass during summarization.
    vectors = np.array([c["semantic_embedding"] for c in chunks])

    # n_clusters cannot exceed the number of samples — guard against the edge
    # case where target_count happens to be larger than the chunk count
    # (shouldn't happen given the caller only invokes this when over the
    # threshold, but defensive coding avoids a confusing sklearn ValueError).
    n_clusters = min(target_count, len(chunks))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vectors)

    representatives = []
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_vectors = vectors[cluster_indices]
        centroid = kmeans.cluster_centers_[cluster_id]

        # Find the chunk whose embedding is closest (Euclidean distance) to
        # the centroid — this is the most "typical" example of this cluster.
        distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
        closest_local_idx = np.argmin(distances)
        closest_global_idx = cluster_indices[closest_local_idx]

        representatives.append(chunks[closest_global_idx])

    return representatives


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT: POST /api/v1/summarize
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=SummarizeResponse,
    status_code=200,
    summary="Summarize a file, directory, or repository",
    description=(
        "Produces a structured summary (overview, key components, data flow, "
        "entry points) for a specific file, an entire directory, or the whole "
        "repository if neither is specified. Unlike /query, this performs "
        "exhaustive retrieval over the requested scope rather than top-K search."
    ),
)
async def summarize(
    body: SummarizeRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> SummarizeResponse:
    """
    WHY WE BUILD THE "scope" LABEL EXPLICITLY:
        The response needs a human-readable description of what was
        summarized for the frontend to display as a header (e.g.,
        "Summary of fastapi/security/oauth2.py"). Rather than have the
        frontend reconstruct this from file_path/directory_path, we build
        it once here so the logic for "what does an empty scope mean"
        lives in exactly one place.

    WHY WE CHECK was_sampled AND SURFACE IT TO THE USER:
        If we silently sampled 40 out of 212 functions without telling the
        user, someone might assume the summary is exhaustive and miss a
        component that wasn't selected as a cluster representative. Surfacing
        this is a transparency requirement — it's the same philosophy as the
        confidence score in query.py: never let the user think they have more
        certainty than they actually do.
    """
    start_time = time.perf_counter()

    owner, repo = _parse_github_url(body.repo_url)
    redis = request.app.state.redis
    qdrant = request.app.state.qdrant

    # Guard: repo must be indexed
    _verify_repo_indexed(redis, owner, repo)

    collection_name = settings.qdrant_collection_name(owner, repo)

    # Determine the human-readable scope label
    if body.file_path:
        scope_label = body.file_path
    elif body.directory_path:
        scope_label = body.directory_path
    else:
        scope_label = "entire repository"

    logger.info(
        f"Summarize request: scope='{scope_label}' "
        f"[repo={owner}/{repo}] "
        f"[focus_hint={body.focus_hint}]"
    )

    # ── Step 1: Exhaustive retrieval over the requested scope ───────────────
    chunks = await _retrieve_chunks_in_scope(
        qdrant_client=qdrant,
        collection_name=collection_name,
        file_path=body.file_path,
        directory_path=body.directory_path,
    )

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No indexed code found for scope '{scope_label}' in "
                f"{owner}/{repo}. Check the path is correct and the "
                f"repository has been fully ingested."
            ),
        )

    # Track files_analyzed before any sampling reduces the chunk list —
    # the user should know how many files were TOUCHED, even if not every
    # chunk within them was sent to the LLM.
    files_analyzed = len({c["file_path"] for c in chunks})
    total_chunks_found = len(chunks)

    # ── Step 2: Sample if too large for the prompt ───────────────────────────
    was_sampled = False
    if len(chunks) > MAX_CHUNKS_FOR_SUMMARY:
        logger.info(
            f"Scope '{scope_label}' has {len(chunks)} chunks — "
            f"sampling down to {MAX_CHUNKS_FOR_SUMMARY} via clustering"
        )
        chunks = _sample_representative_chunks(chunks, MAX_CHUNKS_FOR_SUMMARY)
        was_sampled = True

    # ── Step 3: Generate the structured summary via LLM ───────────────────
    # generation/generator.py's generate_summary() builds a multi-chunk
    # synthesis prompt distinct from generate_answer() — it asks for
    # overview / key_components / data_flow / entry_points as separate
    # structured fields rather than a single free-text answer.
    # generate_summary() makes a blocking Groq API call — run it off the
    # event loop thread, same pattern used by api/routes/query.py for
    # generate()/classify().
    generation: SummaryGenerationResponse = await asyncio.to_thread(
        generate_summary,
        scope_label=scope_label,
        chunks=chunks,
        focus_hint=body.focus_hint,
        settings=settings,
    )

    latency_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        f"Summary complete: scope='{scope_label}' "
        f"[files={files_analyzed}] "
        f"[chunks_found={total_chunks_found}] "
        f"[chunks_used={len(chunks)}] "
        f"[sampled={was_sampled}] "
        f"[latency={latency_ms:.1f}ms]"
    )

    return SummarizeResponse(
        scope=scope_label,
        overview=generation.overview,
        key_components=[
            KeyComponent(**kc) for kc in generation.key_components
        ],
        data_flow=generation.data_flow,
        entry_points=generation.entry_points,
        files_analyzed=files_analyzed,
        chunks_analyzed=len(chunks),
        was_sampled=was_sampled,
        repo_url=body.repo_url,
        latency_ms=round(latency_ms, 2),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT: POST /api/v1/summarize/module
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/module",
    response_model=SummarizeResponse,
    status_code=200,
    summary="Summarize a logical module (directory)",
    description=(
        "Convenience wrapper around the main summarize endpoint, specialized "
        "for the frontend's 'Summarize this folder' action in the file tree. "
        "Always scopes to a directory_path."
    ),
)
async def summarize_module(
    body: SummarizeModuleRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> SummarizeResponse:
    """
    WHY THIS ENDPOINT EXISTS SEPARATELY FROM /summarize:
        The frontend's file tree has a right-click or hover action:
        "Summarize this folder." That interaction only ever provides a
        directory path — never a file path, never a focus hint, never the
        choice to summarize the whole repo. Exposing the full SummarizeRequest
        shape for that one UI action would mean the frontend always sends
        two null fields just to satisfy the schema. This endpoint gives the
        frontend a narrower, purpose-fit contract for that specific action.

    HOW IT'S IMPLEMENTED:
        Internally, this just constructs a SummarizeRequest with
        directory_path set and calls the shared summarize() logic directly
        — there is no duplicated retrieval/generation code, only a thinner
        request schema at the boundary.
    """
    wrapped_request = SummarizeRequest(
        repo_url=body.repo_url,
        directory_path=body.module_path,
    )
    return await summarize(wrapped_request, request, settings)
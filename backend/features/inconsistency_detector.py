"""
features/inconsistency_detector.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY THIS FILE EXISTS
────────────────────
Real codebases accumulate inconsistency over time. Ten engineers work
on the same codebase over two years, each with slightly different
habits, and you end up with:

  - Five different ways to handle exceptions (bare except, typed except,
    logging before re-raising, swallowing silently, wrapping in a custom
    error class)
  - Three different authentication patterns across API endpoints
  - Four different database access styles (raw SQL, ORM, raw connection,
    connection pool)

None of these are individually broken. Together, they are a maintenance
nightmare. New engineers copy the wrong pattern. Code reviews miss them
because reviewers are focused on logic, not style. Static linters don't
catch semantic inconsistencies, only syntactic ones.

This module detects semantic inconsistencies by:
  1. Taking a set of code chunks retrieved for a query like "how is
     error handling done?"
  2. Embedding each chunk using the semantic embedder
  3. Clustering the embeddings with K-Means to find natural groups
  4. Flagging chunks whose embedding is far from their cluster's center
     as OUTLIERS — implementations that differ significantly from the
     majority pattern

The output tells an engineer: "Here are the 3 main patterns used in
this codebase, and here are the 5 places that don't follow any of them."

WHY THIS MATTERS IN FAANG INTERVIEWS
─────────────────────────────────────
This is the kind of tooling Google, Meta, and Amazon build internally
with dedicated teams. Nobody has this in a portfolio project because it
requires combining retrieval, embeddings, AND unsupervised ML — three
separate bodies of knowledge. Explaining how K-Means outlier detection
works on code embeddings demonstrates exactly the "breadth of ML
knowledge applied to engineering problems" that FAANG ML/SWE hybrid
roles look for.

DESIGN DECISIONS
────────────────
K-Means was chosen over alternatives (DBSCAN, hierarchical clustering)
because:
  - It produces a fixed number of clusters (k), which maps naturally
    to "there are N patterns in this codebase" — a human-interpretable
    result.
  - It is fast: O(n * k * iterations) on small code chunk sets (usually
    20-100 chunks retrieved per query).
  - Outlier detection via centroid distance is simple and explainable:
    "this chunk's embedding is 2.1 standard deviations from the nearest
    cluster center" is something engineers understand intuitively.

DBSCAN was considered but rejected for v1 because it requires tuning
epsilon (the neighborhood radius) per dataset — hard to generalize
across codebases. Worth revisiting in v2.

DEPENDENCIES
────────────
- numpy:        vector arithmetic for centroid distance computation
- scikit-learn: KMeans implementation
- The chunk embeddings are passed in as numpy arrays — this module
  does NOT call the embedding layer itself. The retrieval layer
  (retrieval/hybrid_retriever.py) is responsible for fetching chunks
  AND their embeddings, then passing both here.
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.exceptions import ConvergenceWarning
import warnings


# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

# Number of standard deviations from cluster center above which a chunk
# is flagged as an outlier. 2.0 is a common threshold in anomaly
# detection (corresponds roughly to the top 5% most distant points in
# a normal distribution). Increasing this reduces false positives but
# misses subtler inconsistencies.
OUTLIER_THRESHOLD_STD = 2.0

# Minimum number of chunks required to run clustering.
# With fewer than this, there's not enough signal to identify patterns.
# We can't cluster 2 things meaningfully.
MIN_CHUNKS_FOR_CLUSTERING = 4

# Maximum number of clusters to try in the auto-k selection heuristic.
# We don't want to over-segment small chunk sets.
MAX_AUTO_K = 5

# KMeans random seed for reproducibility. Using a fixed seed ensures
# that the same set of chunks always produces the same clustering result,
# which is important for consistent UI behavior and for testing.
KMEANS_RANDOM_STATE = 42

# Number of KMeans initializations to run. sklearn runs the algorithm
# n_init times and picks the best result (lowest inertia). More runs
# reduce the chance of a bad local minimum.
KMEANS_N_INIT = 10


# ─────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────

class InconsistencyLevel(str, Enum):
    """
    Severity classification for a detected outlier.

    Why three levels?
    Not all outliers are equally concerning. A chunk that is slightly
    different from its cluster (MINOR) might just be a newer, improved
    version of the pattern. A chunk with z-score > 4 (SEVERE) is
    implementing something fundamentally different and is likely a
    genuine defect or a copy-paste from a different codebase.

    Giving engineers a severity level helps them prioritize which
    inconsistencies to investigate first in a code review.
    """
    MINOR  = "MINOR"   # z-score between THRESHOLD and THRESHOLD * 1.5
    MODERATE = "MODERATE"  # z-score between THRESHOLD * 1.5 and THRESHOLD * 2
    SEVERE = "SEVERE"  # z-score above THRESHOLD * 2 — starkly different


@dataclass
class CodeChunk:
    """
    A single unit of code passed to the inconsistency detector.

    This mirrors the chunk schema from indexing/chunk_schema.py but
    is defined here as a lightweight dataclass so this module doesn't
    need to import the full Pydantic model (which carries database
    connection overhead).

    Fields
    ──────
    chunk_id : str
        Unique identifier, matching the ID in Qdrant. Used to link
        detected inconsistencies back to the vector store for retrieval
        of full context.

    function_id : str
        Human-readable identifier: "{file_path}::{function_name}".
        Used in the UI to show the engineer where the inconsistency is.

    source_code : str
        The raw source code text of this chunk. Included in the output
        so the engineer can read what's inconsistent without a separate
        API call.

    embedding : np.ndarray
        The semantic embedding vector for this chunk. Must be a 1-D
        numpy array of floats. Shape: (embedding_dim,).
        This is pre-computed by the embedding layer and passed in here.

    file_path : str
        The file this chunk came from. Used for grouping results by file
        in the UI ("these 3 inconsistencies are all in the auth/ folder").

    language : str
        Programming language of this chunk. Inconsistency detection only
        makes sense within a single language — comparing a Python chunk
        to a Java chunk would produce meaningless outlier scores.
    """
    chunk_id:    str
    function_id: str
    source_code: str
    embedding:   np.ndarray
    file_path:   str
    language:    str


@dataclass
class ClusterSummary:
    """
    Summary of a single cluster identified by K-Means.

    Each cluster represents a distinct pattern of implementing something
    (e.g. "error handling pattern A: log then raise", "pattern B: wrap
    in custom exception"). The engineer can look at the representative
    chunks to understand what the pattern is.

    Fields
    ──────
    cluster_id : int
        Zero-indexed cluster number, assigned by K-Means.

    size : int
        Number of chunks in this cluster. Larger clusters represent
        more commonly used patterns.

    representative_chunk_ids : list[str]
        The 3 chunk IDs closest to this cluster's centroid.
        These are the "canonical examples" of the pattern — the most
        typical implementations that define what this cluster looks like.

    centroid : np.ndarray
        The mean embedding vector for this cluster. Used to compute
        distances for outlier detection. Not displayed in the UI.

    intra_cluster_std : float
        Average distance of chunks from the centroid within this cluster.
        Low std = tight cluster, chunks are very similar.
        High std = loose cluster, more variation within the pattern.
        This is used to compute per-chunk z-scores.
    """
    cluster_id:               int
    size:                     int
    representative_chunk_ids: list[str]     = field(default_factory=list)
    centroid:                 np.ndarray    = field(default_factory=lambda: np.array([]))
    intra_cluster_std:        float         = 0.0


@dataclass
class InconsistentChunk:
    """
    A code chunk flagged as inconsistent with the dominant patterns.

    Fields
    ──────
    chunk : CodeChunk
        The full chunk data, including source code for display.

    nearest_cluster_id : int
        Which cluster this chunk is closest to (even though it's an
        outlier, we assign it to the nearest cluster for context).

    distance_from_centroid : float
        Euclidean distance from this chunk's embedding to the nearest
        cluster centroid. Higher = more different from any known pattern.

    z_score : float
        Normalized distance: (distance - cluster_mean) / cluster_std.
        This is the primary ranking signal. A z_score of 3.0 means this
        chunk is 3 standard deviations more unusual than the average
        chunk in its cluster — very likely a genuine inconsistency.

    inconsistency_level : InconsistencyLevel
        Human-readable severity derived from the z_score.

    explanation : str
        A short plain-English explanation of why this was flagged,
        suitable for display in the UI or injection into the LLM
        prompt. Example: "This implementation is 2.8 std deviations
        from the dominant error-handling pattern in cluster 1."
    """
    chunk:                   CodeChunk
    nearest_cluster_id:      int
    distance_from_centroid:  float
    z_score:                 float
    inconsistency_level:     InconsistencyLevel
    explanation:             str


@dataclass
class InconsistencyReport:
    """
    The full output of a single inconsistency detection run.

    This is what gets serialized and returned to the /query API
    endpoint when a query triggers the analytical retrieval path
    (i.e. the query classifier routes to inconsistency detection).

    Fields
    ──────
    query : str
        The original user query that triggered this analysis.
        E.g. "how is error handling done across the codebase?"

    clusters : list[ClusterSummary]
        The dominant patterns found, sorted by size descending.
        Largest cluster first = most common pattern first.

    inconsistencies : list[InconsistentChunk]
        The outlier chunks, sorted by z_score descending.
        Most inconsistent first.

    total_chunks_analyzed : int
        Total number of chunks passed to the detector.

    total_inconsistencies_found : int
        Number of outliers flagged.

    skipped_reason : str | None
        If detection was skipped (e.g. not enough chunks), explains why.
        None when analysis ran normally.
    """
    query:                      str
    clusters:                   list[ClusterSummary]       = field(default_factory=list)
    inconsistencies:            list[InconsistentChunk]    = field(default_factory=list)
    total_chunks_analyzed:      int                        = 0
    total_inconsistencies_found: int                       = 0
    skipped_reason:             Optional[str]              = None


# ─────────────────────────────────────────────────────────────────────
# Helper: Automatic K Selection
# ─────────────────────────────────────────────────────────────────────

def _select_k(n_chunks: int) -> int:
    """
    Automatically selects the number of clusters (k) based on the
    number of chunks available.

    Why not always use k=3?
    If we have only 5 chunks, k=3 creates clusters of size 2, 2, 1 —
    too small to be meaningful. If we have 80 chunks, k=3 might merge
    genuinely distinct patterns into one cluster.

    The heuristic used here: k = min(sqrt(n_chunks / 2), MAX_AUTO_K).
    This is a well-known rule of thumb for K-Means that produces
    reasonable clusters across a wide range of dataset sizes.

    In practice, the number of retrieved chunks is typically 20-40
    (RETRIEVAL_TOP_K from settings), so this will usually return k=3
    or k=4, which maps well to "there are 3-4 main patterns here."

    Args:
        n_chunks: Number of code chunks being clustered.

    Returns:
        Integer k, the number of clusters to use.
    """
    k = max(2, min(int(np.sqrt(n_chunks / 2)), MAX_AUTO_K))
    logger.debug(f"Auto-selected k={k} for {n_chunks} chunks")
    return k


# ─────────────────────────────────────────────────────────────────────
# Helper: Outlier Severity Classification
# ─────────────────────────────────────────────────────────────────────

def _classify_inconsistency(z_score: float) -> InconsistencyLevel:
    """
    Converts a z_score into an InconsistencyLevel severity.

    Thresholds are relative to OUTLIER_THRESHOLD_STD (default 2.0):
      - MINOR:    2.0 ≤ z < 3.0   — somewhat unusual, worth noting
      - MODERATE: 3.0 ≤ z < 4.0   — clearly different from the pattern
      - SEVERE:   z ≥ 4.0          — starkly different, likely a defect

    Args:
        z_score: The normalized distance from cluster centroid.

    Returns:
        InconsistencyLevel enum value.
    """
    threshold = OUTLIER_THRESHOLD_STD
    if z_score >= threshold * 2:
        return InconsistencyLevel.SEVERE
    elif z_score >= threshold * 1.5:
        return InconsistencyLevel.MODERATE
    else:
        return InconsistencyLevel.MINOR


# ─────────────────────────────────────────────────────────────────────
# Helper: Build Human-Readable Explanation
# ─────────────────────────────────────────────────────────────────────

def _build_explanation(
    chunk:             CodeChunk,
    cluster_id:        int,
    cluster_size:      int,
    z_score:           float,
    level:             InconsistencyLevel,
) -> str:
    """
    Generates a plain-English explanation for why a chunk was flagged.

    This explanation is injected into the LLM generation prompt
    alongside the source code, so the model can explain the
    inconsistency to the engineer in natural language.

    It is also displayed directly in the UI tooltip on the inconsistency
    panel.

    Args:
        chunk:        The flagged CodeChunk.
        cluster_id:   The nearest cluster this chunk almost belongs to.
        cluster_size: How many chunks are in the dominant pattern cluster.
        z_score:      The z-score of this chunk.
        level:        The severity classification.

    Returns:
        A single-paragraph string explaining the inconsistency.
    """
    severity_map = {
        InconsistencyLevel.MINOR:    "slightly",
        InconsistencyLevel.MODERATE: "notably",
        InconsistencyLevel.SEVERE:   "significantly",
    }
    adverb = severity_map[level]

    return (
        f"'{chunk.function_id}' is {adverb} inconsistent with the dominant "
        f"pattern (cluster {cluster_id}, which contains {cluster_size} similar "
        f"implementations). Its embedding is {z_score:.2f} standard deviations "
        f"from the cluster centroid, indicating its approach differs "
        f"{'somewhat' if level == InconsistencyLevel.MINOR else 'substantially'} "
        f"from the majority. Review this implementation for alignment with the "
        f"codebase's standard approach."
    )


# ─────────────────────────────────────────────────────────────────────
# Helper: Filter Chunks to Single Language
# ─────────────────────────────────────────────────────────────────────

def _filter_by_language(chunks: list[CodeChunk]) -> list[CodeChunk]:
    """
    Filters the chunk list to only include the most common language.

    Why filter?
    Comparing embeddings of Python code against Java code is meaningless.
    CodeBERT produces different embedding distributions for different
    languages — a Python function that is semantically similar to a
    Java function will NOT have similar embeddings, and vice versa.

    Strategy: find the most common language and keep only those chunks.
    If the codebase is predominantly Python, discard the 2 Java files
    in the retrieved set for this analysis. Log a warning.

    Args:
        chunks: List of CodeChunk objects with potentially mixed languages.

    Returns:
        Filtered list containing only chunks of the most common language.
    """
    if not chunks:
        return chunks

    from collections import Counter
    language_counts = Counter(c.language for c in chunks)
    dominant_language = language_counts.most_common(1)[0][0]
    filtered = [c for c in chunks if c.language == dominant_language]

    if len(filtered) < len(chunks):
        logger.warning(
            f"Filtered out {len(chunks) - len(filtered)} chunks of non-dominant "
            f"languages. Keeping {len(filtered)} chunks in '{dominant_language}'."
        )

    return filtered


# ─────────────────────────────────────────────────────────────────────
# Core Clustering Logic
# ─────────────────────────────────────────────────────────────────────

def _cluster_embeddings(
    embeddings: np.ndarray,
    k: int,
) -> tuple[KMeans, np.ndarray]:
    """
    Runs K-Means clustering on the embedding matrix.

    Why L2-normalize embeddings before clustering?
    Cosine similarity is the standard metric for comparing semantic
    embeddings (it ignores magnitude and focuses on direction). K-Means
    uses Euclidean distance by default. On L2-normalized vectors,
    Euclidean distance and cosine similarity are monotonically related:
    minimizing Euclidean distance between unit vectors is equivalent to
    maximizing cosine similarity. So normalizing first gives us cosine-
    like behavior from a standard Euclidean K-Means.

    Args:
        embeddings: 2D numpy array of shape (n_chunks, embedding_dim).
                    Each row is the embedding of one code chunk.
        k:          Number of clusters.

    Returns:
        Tuple of (fitted KMeans object, cluster_labels array of shape (n_chunks,)).
    """
    # Normalize to unit vectors so cosine similarity ≈ Euclidean distance
    normalized = normalize(embeddings, norm="l2")

    # Suppress ConvergenceWarning on small datasets — not a problem for
    # typical query chunk sets where n_chunks << n_iterations
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        kmeans = KMeans(
            n_clusters=k,
            n_init=KMEANS_N_INIT,
            random_state=KMEANS_RANDOM_STATE,
        )
        labels = kmeans.fit_predict(normalized)

    return kmeans, labels


def _build_cluster_summaries(
    chunks:     list[CodeChunk],
    labels:     np.ndarray,
    kmeans:     KMeans,
    k:          int,
) -> list[ClusterSummary]:
    """
    Builds ClusterSummary objects from K-Means results.

    For each cluster:
    1. Find all chunks assigned to this cluster.
    2. Identify the 3 chunks closest to the centroid — these are the
       "representative" examples of the pattern.
    3. Compute intra-cluster standard deviation of distances from
       the centroid. This is used later for z-score computation.

    Args:
        chunks: The original list of CodeChunk objects.
        labels: Cluster assignment for each chunk (from K-Means).
        kmeans: The fitted KMeans object (for centroid access).
        k:      Number of clusters.

    Returns:
        List of ClusterSummary objects, one per cluster.
    """
    normalized_embeddings = normalize(
        np.array([c.embedding for c in chunks]), norm="l2"
    )
    summaries = []

    for cluster_id in range(k):
        # Find indices of chunks in this cluster
        member_indices = np.where(labels == cluster_id)[0]

        if len(member_indices) == 0:
            # Empty cluster — can happen with small datasets and large k.
            # Skip it rather than producing a summary with no members.
            logger.debug(f"Cluster {cluster_id} is empty, skipping.")
            continue

        centroid = kmeans.cluster_centers_[cluster_id]

        # Compute distance of each member from centroid
        member_embeddings = normalized_embeddings[member_indices]
        distances = np.linalg.norm(member_embeddings - centroid, axis=1)

        # intra-cluster std — needed for z-score normalization later
        intra_std = float(np.std(distances)) if len(distances) > 1 else 1.0
        # Guard against zero std (all chunks identical) to avoid division by zero
        intra_std = max(intra_std, 1e-8)

        # Representative chunks: 3 closest to centroid
        closest_indices = member_indices[np.argsort(distances)[:3]]
        representative_ids = [chunks[i].chunk_id for i in closest_indices]

        summaries.append(ClusterSummary(
            cluster_id=cluster_id,
            size=len(member_indices),
            representative_chunk_ids=representative_ids,
            centroid=centroid,
            intra_cluster_std=intra_std,
        ))

    # Sort by size descending — largest (most common) pattern first
    summaries.sort(key=lambda s: s.size, reverse=True)
    return summaries


def _detect_outliers(
    chunks:     list[CodeChunk],
    labels:     np.ndarray,
    summaries:  list[ClusterSummary],
) -> list[InconsistentChunk]:
    """
    Identifies chunks that are statistical outliers within their cluster.

    For each chunk:
    1. Find its assigned cluster.
    2. Compute its distance from the cluster centroid.
    3. Compute the z-score: (distance - 0) / cluster_std.
       (We use distance from centroid directly, since by definition
       the centroid is the mean — distance 0 is "perfectly average".)
    4. If z_score > OUTLIER_THRESHOLD_STD, flag as inconsistent.

    Why z-score normalization?
    Without normalization, a chunk in a tight cluster (low intra_std)
    and a chunk in a loose cluster (high intra_std) might have the same
    raw distance but very different "unusual-ness". Z-score accounts for
    the spread of each cluster, so outlier detection is fair across
    clusters of different tightness.

    Args:
        chunks:    The original CodeChunk list.
        labels:    Cluster assignment per chunk.
        summaries: ClusterSummary list (for centroid and std access).

    Returns:
        List of InconsistentChunk objects, sorted by z_score descending.
    """
    # Build a lookup from cluster_id → ClusterSummary for fast access
    summary_by_id = {s.cluster_id: s for s in summaries}

    normalized_embeddings = normalize(
        np.array([c.embedding for c in chunks]), norm="l2"
    )
    outliers: list[InconsistentChunk] = []

    for i, chunk in enumerate(chunks):
        cluster_id = int(labels[i])

        if cluster_id not in summary_by_id:
            # Chunk belongs to an empty cluster — skip
            continue

        summary = summary_by_id[cluster_id]
        distance = float(np.linalg.norm(
            normalized_embeddings[i] - summary.centroid
        ))

        # z-score: how many standard deviations from the centroid is this?
        z_score = distance / summary.intra_cluster_std

        if z_score > OUTLIER_THRESHOLD_STD:
            level = _classify_inconsistency(z_score)
            explanation = _build_explanation(
                chunk=chunk,
                cluster_id=cluster_id,
                cluster_size=summary.size,
                z_score=z_score,
                level=level,
            )
            outliers.append(InconsistentChunk(
                chunk=chunk,
                nearest_cluster_id=cluster_id,
                distance_from_centroid=distance,
                z_score=z_score,
                inconsistency_level=level,
                explanation=explanation,
            ))

    # Most inconsistent first
    outliers.sort(key=lambda o: o.z_score, reverse=True)
    return outliers


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

def detect_inconsistencies(
    query:  str,
    chunks: list[CodeChunk],
    k:      Optional[int] = None,
) -> InconsistencyReport:
    """
    Main entry point. Given a user query and the code chunks retrieved
    for that query, finds implementations that deviate from the dominant
    patterns.

    This is called by the query layer when the query classifier routes
    to the "analytical" query type — queries like:
      - "how is error handling done?"
      - "find all places where authentication is handled"
      - "show me inconsistent logging patterns"

    Algorithm
    ─────────
    1. Validate: enough chunks? single language?
    2. Extract embedding matrix from chunks.
    3. Auto-select k (or use provided k).
    4. Run K-Means on L2-normalized embeddings.
    5. Build ClusterSummary objects (dominant patterns).
    6. Detect outliers via z-score thresholding.
    7. Package into InconsistencyReport and return.

    Args:
        query:  The original user query string. Stored in the report
                for traceability.
        chunks: List of CodeChunk objects from the retrieval layer.
                Each must have a valid embedding (pre-computed).
        k:      Number of clusters. If None, auto-selected via
                _select_k(). Providing k explicitly is useful in tests
                or when the caller has domain knowledge about the number
                of expected patterns.

    Returns:
        InconsistencyReport. If analysis was skipped (too few chunks,
        empty input), returns report with skipped_reason set.

    Example
    ───────
        # Retrieve chunks for the query using the retrieval layer
        chunks = retriever.retrieve_with_embeddings("error handling patterns")

        # Convert to CodeChunk objects
        code_chunks = [CodeChunk(...) for r in chunks]

        # Detect inconsistencies
        report = detect_inconsistencies("error handling patterns", code_chunks)

        for outlier in report.inconsistencies:
            print(f"[{outlier.inconsistency_level}] {outlier.chunk.function_id}")
            print(f"  {outlier.explanation}")
    """
    logger.info(
        f"Starting inconsistency detection for query='{query}', "
        f"chunks={len(chunks)}"
    )

    # ── Step 1: Validate input ────────────────────────────────────
    if not chunks:
        return InconsistencyReport(
            query=query,
            skipped_reason="No code chunks were provided for analysis.",
        )

    if len(chunks) < MIN_CHUNKS_FOR_CLUSTERING:
        return InconsistencyReport(
            query=query,
            total_chunks_analyzed=len(chunks),
            skipped_reason=(
                f"Too few chunks to cluster ({len(chunks)} provided, "
                f"minimum {MIN_CHUNKS_FOR_CLUSTERING} required). "
                f"Try broadening the query to retrieve more results."
            ),
        )

    # Filter to single dominant language — cross-language comparison
    # produces meaningless embedding distances
    filtered_chunks = _filter_by_language(chunks)

    if len(filtered_chunks) < MIN_CHUNKS_FOR_CLUSTERING:
        return InconsistencyReport(
            query=query,
            total_chunks_analyzed=len(chunks),
            skipped_reason=(
                f"After filtering to the dominant language, only "
                f"{len(filtered_chunks)} chunks remain (minimum "
                f"{MIN_CHUNKS_FOR_CLUSTERING} required)."
            ),
        )

    # ── Step 2: Build embedding matrix ───────────────────────────
    embeddings = np.array([c.embedding for c in filtered_chunks])

    # Validate embedding shapes — all chunks must have the same dimension
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D embedding matrix, got shape {embeddings.shape}. "
            f"Each chunk embedding must be a 1D numpy array."
        )

    dims = set(c.embedding.shape[0] for c in filtered_chunks)
    if len(dims) > 1:
        raise ValueError(
            f"Inconsistent embedding dimensions across chunks: {dims}. "
            f"All chunks must use the same embedding model."
        )

    # ── Step 3: Select k ─────────────────────────────────────────
    selected_k = k if k is not None else _select_k(len(filtered_chunks))
    # Never request more clusters than we have chunks
    selected_k = min(selected_k, len(filtered_chunks))

    # ── Step 4: Cluster ───────────────────────────────────────────
    kmeans, labels = _cluster_embeddings(embeddings, selected_k)

    # ── Step 5: Build cluster summaries ──────────────────────────
    cluster_summaries = _build_cluster_summaries(
        filtered_chunks, labels, kmeans, selected_k
    )

    # ── Step 6: Detect outliers ───────────────────────────────────
    inconsistencies = _detect_outliers(
        filtered_chunks, labels, cluster_summaries
    )

    logger.info(
        f"Inconsistency detection complete. "
        f"Clusters: {len(cluster_summaries)}, "
        f"Outliers: {len(inconsistencies)} / {len(filtered_chunks)} chunks"
    )

    return InconsistencyReport(
        query=query,
        clusters=cluster_summaries,
        inconsistencies=inconsistencies,
        total_chunks_analyzed=len(filtered_chunks),
        total_inconsistencies_found=len(inconsistencies),
        skipped_reason=None,
    )


def summarize_inconsistencies(report: InconsistencyReport) -> str:
    """
    Converts an InconsistencyReport into a plain-English summary for
    injection into the LLM generation prompt.

    The generation layer needs to explain inconsistencies to the user
    in natural language. This function produces a structured text
    summary that the LLM can reason about without needing to parse
    the full dataclass structure.

    Args:
        report: InconsistencyReport returned by detect_inconsistencies().

    Returns:
        Multi-line plain-English string.

    Example output:
        "Analysis of 'error handling patterns' found 3 dominant patterns
         across 42 code chunks:
           Pattern 1 (18 chunks): typical implementation — see auth/login.py
           Pattern 2 (16 chunks): alternative approach — see utils/errors.py
           Pattern 3 (8 chunks): less common variant

         5 inconsistencies detected:
           [SEVERE] services/payment.py::charge_card — 3.8 std from pattern 1
           [MODERATE] api/routes/user.py::delete_user — 3.1 std from pattern 2"
    """
    if report.skipped_reason:
        return f"Inconsistency analysis skipped: {report.skipped_reason}"

    if report.total_inconsistencies_found == 0:
        return (
            f"Analysis of '{report.query}' found {len(report.clusters)} "
            f"implementation patterns across {report.total_chunks_analyzed} "
            f"code chunks with no significant inconsistencies detected. "
            f"The codebase appears consistent for this pattern."
        )

    lines = [
        f"Analysis of '{report.query}' found {len(report.clusters)} dominant "
        f"patterns across {report.total_chunks_analyzed} code chunks:",
    ]
    for i, cluster in enumerate(report.clusters):
        lines.append(f"  Pattern {i+1} ({cluster.size} chunks)")

    lines.append(
        f"\n{report.total_inconsistencies_found} inconsistenc"
        f"{'y' if report.total_inconsistencies_found == 1 else 'ies'} detected:"
    )
    for outlier in report.inconsistencies[:10]:  # Cap at 10 for prompt length
        lines.append(
            f"  [{outlier.inconsistency_level.value}] "
            f"{outlier.chunk.function_id} — {outlier.explanation}"
        )
    if report.total_inconsistencies_found > 10:
        lines.append(
            f"  ... and {report.total_inconsistencies_found - 10} more."
        )

    return "\n".join(lines)
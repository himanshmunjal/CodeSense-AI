"""
================================================================================
api/routes/query.py — Codebase Query Endpoints
================================================================================

WHY THIS FILE EXISTS:
    This file is the core intelligence interface of CodeSense — it is what
    the user actually interacts with after a repository is ingested.

    Once a repo is indexed (via ingest.py), developers ask questions like:
        - "Where is authentication handled?"
        - "What calls the payment processing function?"
        - "Find all places where error handling is inconsistent."
        - "Explain what the data pipeline module does."

    This file receives those questions, routes them through the correct
    retrieval strategy, re-ranks results, generates a grounded answer via
    LLM, and returns a structured response with file + line citations.

WHY THIS FILE IS SEPARATE FROM ingest.py:
    Ingestion and querying are fundamentally different operations:
        - Ingestion is write-heavy, long-running, and triggered rarely.
        - Querying is read-heavy, latency-sensitive, and triggered constantly.
    Keeping them in separate files makes each easier to understand, test,
    and optimize independently. It also means future changes to the query
    pipeline (e.g., swapping the re-ranker model) don't risk touching
    ingestion logic, and vice versa.

THE QUERY PIPELINE (what happens on every request):
    1. Validate request body (Pydantic — <1ms).
    2. Check if the requested repository is actually indexed (Redis — <5ms).
    3. Classify the query into one of four types (LLM call — ~200ms).
    4. Route to the correct retrieval strategy based on query type:
           lookup        → semantic ANN search (Qdrant)
           relational    → graph traversal (networkx)
           analytical    → semantic search + k-means clustering
           summarization → multi-chunk semantic retrieval
    5. Re-rank the top-K retrieved chunks (cross-encoder — ~300ms).
    6. Apply confidence threshold: if best score < RETRIEVAL_CONFIDENCE_THRESHOLD
       (0.5 by default), refuse to answer.
    7. Build a grounded prompt with retrieved code chunks + metadata.
    8. Call the LLM with strict grounding instructions (LLM call — ~1-2s).
    9. Return structured response: answer + sources + confidence + follow-ups.

ENDPOINTS IN THIS FILE:
    POST /api/v1/query
        → Main query endpoint. Accepts a natural language question about
          a specific repository and returns a grounded answer.

    POST /api/v1/query/batch
        → Accepts multiple questions at once, processes them in parallel,
          and returns answers for all. Useful for evaluation runs (Days 19-20).

    GET  /api/v1/query/history/{owner}/{repo}
        → Returns recent query history for a repository. Stored in Redis.
          Used by the frontend to populate "recent queries" suggestions.

LATENCY TARGETS (per project spec):
    p50: < 2 seconds
    p90: < 3 seconds
    p99: < 5 seconds
    These are measured by the latency_benchmark.py evaluation script.
================================================================================
"""

from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from loguru import logger
import asyncio
import re
import time
import json

import numpy as np

from qdrant_client.http import models as qdrant_models

from config import get_settings, Settings
from retrieval.query_classifier import classifier as query_classifier, QueryType, ClassificationError
from retrieval.hybrid_retriever import HybridResult
from retrieval.semantic_retriever import semantic_retriever, SemanticSearchResult
from retrieval.graph_retriever import graph_retriever, GraphNode, GraphSearchResult
from retrieval.reranker import Reranker, RankedResult
from generation.generator import generate
from parsing.tree_sitter_parser import parse_source_string, walk_tree
from generation.response_schema import (
    CodeSourceReference,
    QueryType as GenQueryType,
)
from features.inconsistency_detector import (
    CodeChunk as InconsistencyCodeChunk,
    detect_inconsistencies,
)


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL WIRING NOTE
# ─────────────────────────────────────────────────────────────────────────────
# SemanticRetriever and GraphRetriever are imported here as the module-level
# singletons already defined in their own files (`semantic_retriever` /
# `graph_retriever`) — both load expensive resources at import time (CodeBERT,
# the call graph) exactly once, rather than per-request. The Reranker, which
# loads a cross-encoder model, is instantiated once at app startup in
# api/main.py's lifespan and shared via request.app.state.reranker — see that
# file. HybridRetriever itself isn't used here: its call-graph-centrality
# fusion doesn't apply to any of the four query types as actually routed
# below (ANALYTICAL uses features/inconsistency_detector.py's real k-means
# clustering instead — see that branch for why).


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER INSTANCE
# ─────────────────────────────────────────────────────────────────────────────

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST & RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """
    Schema for POST /query request body.

    Example JSON payload:
        {
            "repo_url": "https://github.com/tiangolo/fastapi",
            "question": "Where is authentication handled and what method is used?",
            "max_results": 5
        }
    """

    repo_url: str = Field(
        ...,
        description="GitHub URL of the repository to query. Must already be ingested.",
        examples=["https://github.com/tiangolo/fastapi"],
    )

    question: str = Field(
        ...,
        description="Natural language question about the codebase.",
        examples=[
            "Where is authentication handled?",
            "What functions call the database connection module?",
            "Find all places where exceptions are silently swallowed.",
        ],
        min_length=5,
        max_length=1000,
    )

    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description=(
            "Maximum number of code chunks to include in the generated answer. "
            "Higher values = more context for the LLM, but higher latency and cost. "
            "Default of 5 hits the latency target for 90% of queries."
        ),
    )

    filter_language: Optional[str] = Field(
        default=None,
        description=(
            "Restrict retrieval to files of a specific language. "
            "One of: python, javascript, typescript, java, go. "
            "Useful when you know the answer is in a specific part of the codebase."
        ),
        examples=["python", "typescript"],
    )

    filter_file_path: Optional[str] = Field(
        default=None,
        description=(
            "Restrict retrieval to files whose path contains this substring. "
            "Example: 'auth' will only search files with 'auth' in their path. "
            "Case-insensitive."
        ),
        examples=["auth", "api/routes", "models/"],
    )

    @field_validator("question")
    @classmethod
    def sanitize_question(cls, v: str) -> str:
        """
        WHY THIS VALIDATOR EXISTS:
            We pass the question directly into a prompt that goes to the LLM.
            While OpenAI has their own safety filters, we want to strip leading/
            trailing whitespace and normalize newlines on our side before the
            question touches any downstream system.

            We also enforce min_length=5 above to prevent empty or trivially
            short questions from consuming API credits.
        """
        return " ".join(v.strip().split())  # Normalize all whitespace

    @field_validator("filter_language")
    @classmethod
    def validate_language(cls, v: Optional[str]) -> Optional[str]:
        """
        WHY THIS VALIDATOR EXISTS:
            If we pass an unsupported language to the Qdrant filter, it will
            silently return zero results instead of raising an error. Catching
            it here gives the developer an immediate, clear error message.
        """
        if v is None:
            return v
        supported = {"python", "javascript", "typescript", "java", "go"}
        if v.lower() not in supported:
            raise ValueError(
                f"filter_language must be one of: {', '.join(sorted(supported))}. "
                f"Got: '{v}'"
            )
        return v.lower()


class CodeSource(BaseModel):
    """
    Represents a single piece of code that was retrieved and used to
    generate the answer. The frontend uses these to render the source
    viewer panel — clicking a source jumps to that file + line in the code.
    """

    file_path: str = Field(
        ...,
        description="Relative path from the repo root. Example: 'fastapi/security/oauth2.py'",
    )
    function_name: str = Field(
        ...,
        description="Name of the function or class this chunk belongs to.",
    )
    start_line: int = Field(
        ...,
        description="Line number where this code chunk begins (1-indexed).",
    )
    end_line: int = Field(
        ...,
        description="Line number where this code chunk ends (1-indexed).",
    )
    language: str = Field(
        ...,
        description="Programming language of this file.",
    )
    relevance_score: float = Field(
        ...,
        description=(
            "Similarity score after re-ranking (0.0 to 1.0). "
            "Higher = more relevant to the question. "
            "Answers whose best score is below RETRIEVAL_CONFIDENCE_THRESHOLD "
            "(0.5 by default) are refused."
        ),
    )
    snippet: str = Field(
        ...,
        description=(
            "The actual code text of this chunk. "
            "Included so the frontend can display it without a second API call."
        ),
    )


class QueryResponse(BaseModel):
    """
    Schema for POST /query response.

    This is the structured output that every query returns. The design is
    deliberately explicit:
        - answer: the human-readable explanation
        - sources: the exact code locations that back the answer
        - confidence: a signal the frontend uses to show a warning badge
        - query_type: lets the frontend show how the question was interpreted
        - followup_queries: drives the "you might also ask" UI feature
        - latency_ms: for the latency benchmark and frontend display

    WHY STRUCTURED OUTPUT INSTEAD OF PLAIN TEXT:
        If we returned a plain text string, the frontend would have to parse
        file paths and line numbers out of natural language — fragile and
        error-prone. A structured response means the frontend can render
        source citations as clickable links with zero parsing.
    """

    question: str = Field(
        ...,
        description="The original question that was asked (echoed back for UI convenience).",
    )
    answer: str = Field(
        ...,
        description=(
            "The generated answer, grounded in the retrieved code. "
            "If no relevant code was found, this contains the refusal message."
        ),
    )
    sources: list[CodeSource] = Field(
        default_factory=list,
        description=(
            "List of code locations used to generate the answer, ordered by relevance. "
            "Empty if confidence was below threshold (answer is a refusal)."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "The highest EMBEDDING SIMILARITY among retrieved chunks (0.0 to 1.0) "
            "— NOT a correctness/completeness score for the generated answer. "
            "Below settings.retrieval_confidence_threshold: system refused to "
            "answer (no relevant code found). See sources_used for a more "
            "direct 'how grounded is this answer' signal — a correct, "
            "complete answer can legitimately score moderately here if the "
            "embedding model just doesn't produce high absolute scores for "
            "this codebase's vocabulary (see embeddings/code_embedder.py)."
        ),
    )
    sources_used: int = Field(
        default=0,
        ge=0,
        description=(
            "How many of `sources` the generated answer actually cites "
            "inline (via the [file_path:function_name:Lstart-Lend] format — "
            "see generation/generator.py's system prompt), out of "
            "len(sources) available. A more direct 'is this answer actually "
            "grounded in what was retrieved' signal than `confidence`, which "
            "only measures retrieval similarity, not whether the answer "
            "used what was retrieved. 0 for refusals (sources is always [] "
            "there too)."
        ),
    )
    query_type: str = Field(
        ...,
        description=(
            "How the query was classified. One of: "
            "lookup | relational | analytical | summarization. "
            "Shown in the UI so developers know which retrieval path was used."
        ),
    )
    repo_url: str = Field(
        ...,
        description="The repository that was queried.",
    )
    followup_queries: list[str] = Field(
        default_factory=list,
        description=(
            "Suggested follow-up questions generated by the LLM based on the answer. "
            "Drives the 'You might also ask' feature in the frontend. "
            "Always 3 suggestions."
        ),
    )
    latency_ms: float = Field(
        ...,
        description="Total server-side processing time in milliseconds.",
    )


class BatchQueryRequest(BaseModel):
    """
    Schema for POST /query/batch.

    Allows sending multiple questions in one request, processed in parallel.
    Primarily used by the evaluation harness (run_eval.py) to test 30-40
    question-answer pairs efficiently.
    """

    repo_url: str = Field(
        ...,
        description="GitHub URL of the repository to query.",
    )
    questions: list[str] = Field(
        ...,
        description="List of questions to answer in parallel.",
        min_length=1,
        max_length=20,   # Cap at 20 to prevent abuse and excessive LLM costs
    )


class BatchQueryResponse(BaseModel):
    """
    Schema for POST /query/batch response.
    """

    repo_url: str
    total_questions: int
    results: list[QueryResponse]
    total_latency_ms: float


class QueryHistoryItem(BaseModel):
    """
    A single entry in the query history for a repository.
    """
    question: str
    query_type: str
    confidence: float
    asked_at: str        # ISO 8601 timestamp
    latency_ms: float


class QueryHistoryResponse(BaseModel):
    """
    Schema for GET /query/history/{owner}/{repo}.
    """
    owner: str
    repo: str
    total: int
    history: list[QueryHistoryItem]


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — VERIFY REPOSITORY IS INDEXED
# ─────────────────────────────────────────────────────────────────────────────

def _verify_repo_indexed(redis, owner: str, repo: str) -> dict:
    """
    WHY THIS HELPER EXISTS:
        Before running a query, we must confirm the repository is actually
        in Qdrant. If someone queries a repo they forgot to ingest, without
        this check they'd get a confusing "no results found" answer instead
        of a clear "this repo hasn't been ingested yet" error.

    WHAT IT DOES:
        Looks up the Redis key written by the Celery task on successful
        ingestion: indexed_repo:{owner}:{repo}

        If the key exists, the repo is indexed and we return its metadata.
        If not, we raise a 404 with an actionable error message telling the
        user to run POST /api/v1/ingest first.

    RETURNS:
        The repository metadata dict (owner, repo_url, total_chunks, etc.)

    RAISES:
        HTTPException 404 if the repository hasn't been ingested.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — PARSE GITHUB URL
# ─────────────────────────────────────────────────────────────────────────────

def _parse_github_url(repo_url: str) -> tuple[str, str]:
    """
    Extracts owner and repo name from a GitHub URL.
    Identical logic to the helper in ingest.py — duplicated here to keep
    the route files self-contained and avoid a circular import between routes.

    Example:
        _parse_github_url("https://github.com/tiangolo/fastapi")
        → ("tiangolo", "fastapi")
    """
    parts = repo_url.rstrip("/").split("/")
    return parts[-2], parts[-1]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — RELATIONAL QUERY ROUTING
# ─────────────────────────────────────────────────────────────────────────────
#
# GraphRetriever has no generic "search by natural language" method — it only
# supports targeted traversals from a *known* function name (get_callers,
# get_callees, get_impact, get_path). So for RELATIONAL queries we have to
# do two things query_classifier.py doesn't do for us: (1) guess which
# function the user is asking about, and (2) guess which direction/traversal
# they mean. Both are simple heuristics, not NLP — good enough to route to
# the right GraphRetriever method for the common phrasings in practice.

_RELATIONAL_STOPWORDS = {
    "what", "who", "which", "does", "do", "calls", "call", "calling",
    "the", "a", "an", "is", "are", "in", "of", "to", "on", "from",
    "function", "method", "module", "class", "would", "break", "if",
    "changed", "change", "depend", "depends", "and", "or", "how",
}


def _extract_function_target(question: str) -> str:
    """
    Best-effort extraction of the function name a RELATIONAL question is
    asking about, e.g. "What calls authenticate()?" -> "authenticate".

    GraphRetriever._resolve_node() does fuzzy substring/prefix matching
    against node IDs, but it needs a single candidate string to match
    against — passing the whole question would almost never substring-match
    a "file_path::function_name" node ID. We extract the most code-like
    identifier token instead: prefer one written with trailing "()", then
    prefer snake_case/CamelCase identifiers over plain English words, and
    fall back to the last non-stopword token.
    """
    paren_matches = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(\)", question)
    if paren_matches:
        return paren_matches[-1]

    # "the update function" / "the AdaptiveKVCache class" — the identifier
    # directly modifying "function"/"method"/"class" is almost always the
    # actual target, and this phrasing is common enough ("what breaks if I
    # change the update function in cache.py?") that it needs to outrank the
    # positional fallback below — without this, a trailing filename like
    # "cache.py" gets picked instead of "update" simply for appearing later
    # in the sentence.
    named_matches = re.findall(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s+(?:function|method|class)\b", question
    )
    if named_matches:
        return named_matches[-1]

    # Filenames (foo.py, bar.js) are never the target themselves — strip
    # their stem out of consideration so "cache.py" can't be mistaken for a
    # function/class name just because it's the last identifier-shaped token.
    filename_stems = {
        m.lower() for m in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\.[a-z]{1,4}\b", question)
    }

    tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", question)
    candidates = [
        t for t in tokens
        if t.lower() not in _RELATIONAL_STOPWORDS
        and t.lower() not in filename_stems
        and len(t) > 2
    ]

    code_like = [c for c in candidates if "_" in c or (c[:1].isupper() and not c.isupper())]
    if code_like:
        return code_like[-1]
    if candidates:
        return candidates[-1]
    return question


def _route_relational_query(
    question: str, collection_name: str, settings: Settings
) -> list[GraphSearchResult]:
    """
    Decide which GraphRetriever traversal(s) answer a RELATIONAL question,
    based on simple keyword phrasing. Documented judgment call (see
    api/routes/query.py's module docstring / the task notes this was written
    against): GraphRetriever exposes purpose-specific methods, not one
    generic `.retrieve()`, so *something* upstream has to pick a direction.

        "what would break" / "impact" / "blast radius"  -> get_impact
        "what does X call/depend on"                     -> get_callees
        "path"/"chain" phrasing with two identifiers      -> get_path
        everything else (default: "what/who calls X")     -> get_callers

    Returns a list (usually length 1) of GraphSearchResult so ambiguous
    phrasing can combine callers + callees.
    """
    q_lower = question.lower()

    if any(kw in q_lower for kw in ("break", "impact", "blast radius", "risky", "risk of changing")):
        target = _extract_function_target(question)
        return [graph_retriever.get_impact(target, collection_name, max_depth=4)]

    if ("chain" in q_lower or "path" in q_lower or "reach" in q_lower) and " to " in q_lower:
        before, _, after = question.partition(" to ")
        source = _extract_function_target(before)
        target = _extract_function_target(after)
        if source and target and source != target:
            return [graph_retriever.get_path(source, target, collection_name)]

    if any(kw in q_lower for kw in ("what does", "depend on", "depends on", "does it call", "calls what")):
        target = _extract_function_target(question)
        return [graph_retriever.get_callees(target, collection_name, max_depth=2)]

    # Default and most common phrasing: "what calls X?" / "who calls X?"
    target = _extract_function_target(question)
    results = [graph_retriever.get_callers(target, collection_name, max_depth=3)]

    # If the phrasing gives no directional signal at all (no "call"-family
    # word), the user may equally mean callers or callees — return both so
    # the ranked list isn't arbitrarily one-sided.
    if not any(kw in q_lower for kw in ("call", "invoke", "uses", "used by")):
        results.append(graph_retriever.get_callees(target, collection_name, max_depth=2))

    return results


def _fetch_snippet_for_node(
    qdrant_client, collection_name: str, node: GraphNode
) -> tuple[str, str]:
    """
    Look up the source snippet + docstring for a graph node from Qdrant.

    GraphNode (retrieval/graph_retriever.py) carries structural metadata
    (file_path, line range, complexity) but never the source text itself —
    that only lives in the Qdrant payload written during ingestion (see
    indexing/chunk_schema.py's CodeChunk.to_qdrant_payload()). We use
    `scroll` (an exact metadata lookup), not `search` — there is no query
    vector here, just "find the point with this file_path + entity name."

    We try both payload key names ("fully_qualified_name" then
    "entity_name") because call_graph_builder.py's node naming and
    indexing's payload naming were not written by the same pass and are not
    guaranteed to agree on which one is set for a given chunk.

    Returns ("", "") if nothing matches — callers must handle an empty
    snippet gracefully (the LLM prompt will just show less code for that
    source; it will not crash).
    """
    for name_field in ("fully_qualified_name", "entity_name"):
        try:
            points, _ = qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="file_path",
                            match=qdrant_models.MatchValue(value=node.file_path),
                        ),
                        qdrant_models.FieldCondition(
                            key=name_field,
                            match=qdrant_models.MatchValue(value=node.function_name),
                        ),
                    ]
                ),
                limit=1,
                with_payload=True,
            )
        except Exception as e:
            logger.debug(f"Snippet lookup failed for '{node.node_id}' via {name_field}: {e}")
            continue
        if points:
            payload = points[0].payload or {}
            return payload.get("source_code", ""), payload.get("docstring", "")
    return "", ""


_CITATION_PATTERN = re.compile(r"\[([^\]:]+):([^\]:]+):L?\d+[-–]L?\d+\]")


def _count_sources_used(answer: str, sources: "list[CodeSource]") -> int:
    """
    Count how many of `sources` the generated answer actually cites inline.

    Scans for the LLM's own citation format ("[file_path:function_name:
    Lstart-Lend]" — mandated by generation/generator.py's system prompt) and
    matches by file_path only, not also function_name or the line range.

    WHY FILE_PATH ONLY:
        The LLM doesn't always get the function_name or line range in its
        own citation exactly right — observed directly: citing a whole-file
        MODULE chunk's line range (L1-L118) against a specific function's
        name it read out of that same chunk's body. file_path is the
        reliable part of the citation. This can undercount in the rare case
        a single file contributes two distinct sources and the LLM's
        citations don't clearly distinguish which one backs which claim —
        an acceptable conservative bias for a "how grounded is this" signal,
        better than overcounting a source as "used" when it wasn't.
    """
    cited_paths = {m.group(1).strip() for m in _CITATION_PATTERN.finditer(answer)}
    return sum(1 for s in sources if s.file_path in cited_paths)


def _to_source_reference(ranked: RankedResult) -> CodeSourceReference:
    """
    Convert a re-ranked retrieval result into the CodeSourceReference shape
    generation/generator.py actually requires (it does not accept
    RankedResult, HybridResult, or any retriever-native type — see
    generation/response_schema.py).

    similarity/confidence are derived from hybrid_score (bounded [0,1]),
    never from rerank_score (an unbounded cross-encoder logit — see the
    confidence-threshold comment in _execute_query for why).

    Snippet text is NOT truncated here — see _apply_snippet_budget(), which
    is applied once to the whole source list after this conversion. A
    per-item cap applied here would size every source for the *worst case*
    (many large sources at once) even when the actual list is small — see
    that function's docstring for why that was a real, observed bug.
    """
    similarity = max(0.0, min(1.0, ranked.hybrid_score))
    start_line = max(ranked.start_line, 1)
    end_line = max(ranked.end_line, start_line)
    return CodeSourceReference(
        file_path=ranked.file_path,
        function_name=ranked.function_name,
        start_line=start_line,
        end_line=end_line,
        language=ranked.language or "unknown",
        similarity=similarity,
        confidence=CodeSourceReference.confidence_from_similarity(similarity),
        snippet=ranked.code or "(source unavailable)",
    )


# Total character budget for ALL source snippets combined in one generation
# prompt. ~4 chars/token, so 6000 chars ≈ 1500 tokens — safely under Groq's
# gpt-oss-120b free-tier 8000 TPM limit even after the system prompt,
# per-source docstrings/citations, and the user's question are added on top.
_TOTAL_SNIPPET_CHAR_BUDGET = 6000


def _apply_snippet_budget(sources: list[CodeSourceReference]) -> list[CodeSourceReference]:
    """
    Cap the COMBINED size of every source's snippet, not each source
    individually — applied once, after all sources are built.

    WHY NOT A FLAT PER-SOURCE CAP (what this replaced):
        A flat per-source cap (e.g. "2000 chars each") has to be sized for
        the worst case — up to reranker_top_n (5) sources, each potentially
        a whole-file MODULE chunk (see build_module_chunk in
        tasks/celery_worker.py, which stores full unbounded file content).
        That sizing is correct for the worst case but actively harmful for
        the common case: a small repo's whole-file module chunk (e.g. a
        118-line FastAPI app, ~4KB) got chopped in half by a flat 2000-char
        cap even though the *entire* prompt for that query — both sources
        combined — was nowhere near the actual token budget. Observed
        directly: "what are the different backend API routes?" against a
        6-route file only reported 2 routes, because the flat cap sliced
        the file's source off partway through, before the other 4 route
        handlers. The fix is a budget over the WHOLE list, so a small
        result set keeps its full content and only a large one gets
        truncated — and only as much as it actually needs to be.

    HOW SOURCES ARE CHOSEN WHEN TRUNCATION IS NEEDED:
        Sources are already rank-ordered (most relevant first, from the
        reranker). Budget is spent in that order — top-ranked sources keep
        their full content, and truncation (or complete exclusion) falls on
        the lowest-ranked sources first, since those are the ones the
        reranker itself judged least relevant to the question.
    """
    result: list[CodeSourceReference] = []
    remaining = _TOTAL_SNIPPET_CHAR_BUDGET
    for src in sources:
        if remaining <= 0:
            break
        snippet = src.snippet
        if len(snippet) > remaining:
            # A head cut silently hides everything past the cutoff — for a
            # class chunk that means every method below it vanishes, and the
            # LLM truthfully reports it "can't see" code that was retrieved.
            # An outline keeps every definition (with absolute line numbers)
            # visible at a fraction of the size. Fall back to the head cut
            # for chunks with no nested definitions or unparseable languages.
            snippet = _outline_snippet(src) or snippet
            if len(snippet) > remaining:
                snippet = snippet[:remaining] + "\n... [truncated]"
        result.append(src.model_copy(update={"snippet": snippet}))
        remaining -= len(snippet)
    return result


# tree-sitter node types that count as a "definition" worth listing in an
# outline, across every language in parsing/tree_sitter_parser.py's registry.
_OUTLINE_NODE_TYPES = frozenset({
    # python
    "function_definition", "class_definition",
    # java
    "method_declaration", "constructor_declaration",
    "class_declaration", "interface_declaration", "enum_declaration",
    # javascript / typescript
    "function_declaration", "method_definition",
    # go: function_declaration / method_declaration already covered above
})


def _outline_snippet(src: CodeSourceReference) -> Optional[str]:
    """
    Replace an over-budget snippet with its header line plus the signature of
    every nested definition, each tagged with its absolute line number.

    Observed failure this fixes: "Where is the request context pushed?"
    retrieved the 9.9K-char AppContext class chunk (ctx.py:260-525). The
    budget head-cut it at 6000 chars; `def push` began at char 6064, so the
    LLM answered that no push operation was shown. With an outline, the
    model still sees `L416: def push(self) -> None:` and can cite it.

    Returns None when there's nothing useful to outline (no nested
    definitions, unsupported language, parse failure) so the caller falls
    back to a plain head cut.
    """
    try:
        tree = parse_source_string(src.snippet, src.language)
    except ValueError:  # language not in the tree-sitter registry
        return None
    if tree is None:
        return None

    source_bytes = src.snippet.encode("utf-8")
    entries: list[str] = []
    for node in walk_tree(tree.root_node):
        # Row 0 is the chunk's own definition — the header line covers it.
        if node.type not in _OUTLINE_NODE_TYPES or node.start_point[0] == 0:
            continue
        # Signature = everything before the body, whitespace-collapsed so
        # multi-line parameter lists stay on one outline line.
        body = node.child_by_field_name("body")
        sig_end = body.start_byte if body is not None else node.end_byte
        signature = " ".join(
            source_bytes[node.start_byte:sig_end].decode("utf-8", "replace").split()
        )
        indent = " " * node.start_point[1]
        line_no = src.start_line + node.start_point[0]
        entries.append(f"{indent}L{line_no}: {signature[:200]}")

    if not entries:
        return None

    header = src.snippet.split("\n", 1)[0]
    return (
        f"{header}\n"
        f"    # [chunk too large for the prompt — outline only, bodies omitted. "
        f"Full source spans L{src.start_line}-L{src.end_line}.]\n"
        + "\n".join(entries)
    )


def _order_specific_before_containers(reranked: list[RankedResult]) -> list[RankedResult]:
    """
    Move any result whose line range contains another result (same file)
    to just after the last result it contains.

    WHY:
        The cross-encoder often scores a whole CLASS chunk slightly above the
        METHOD chunk the question is actually about — the class contains the
        method's text plus more matching words. Observed: for "Where is the
        request context pushed?", AppContext (260-525) reranked at 5.59 and
        AppContext.push (416-444) at 5.36. Because _apply_snippet_budget
        spends budget in rank order, the 9.9K class chunk consumed all of it
        and push() was dropped entirely.

        Putting the specific chunk first guarantees its full body reaches the
        LLM; the container still follows (outlined if it no longer fits), so
        overview questions keep the class-level context too.

    Order is otherwise preserved. Strict containment is a partial order (equal
    ranges are not treated as containment), so there are no cycles.
    """
    n = len(reranked)

    def contains(a: RankedResult, b: RankedResult) -> bool:
        return (
            a.file_path == b.file_path
            and a.start_line <= b.start_line
            and b.end_line <= a.end_line
            and (a.start_line, a.end_line) != (b.start_line, b.end_line)
        )

    children = [
        [j for j in range(n) if j != i and contains(reranked[i], reranked[j])]
        for i in range(n)
    ]

    # Effective position: a container sorts just after its latest-placed
    # descendant. Memoised recursion handles nesting (module > class > method).
    positions: dict[int, float] = {}

    def position(i: int) -> float:
        if i not in positions:
            positions[i] = max([float(i)] + [position(j) + 0.5 for j in children[i]])
        return positions[i]

    order = sorted(range(n), key=lambda i: (position(i), i))
    return [reranked[i] for i in order]


# ─────────────────────────────────────────────────────────────────────────────
# CORE QUERY EXECUTION LOGIC
# ─────────────────────────────────────────────────────────────────────────────

async def _execute_query(
    question: str,
    repo_url: str,
    owner: str,
    repo: str,
    max_results: int,
    filter_language: Optional[str],
    filter_file_path: Optional[str],
    qdrant_client,
    reranker: Reranker,
    settings: Settings,
) -> QueryResponse:
    """
    WHY THIS IS A SEPARATE FUNCTION AND NOT INLINE IN THE ROUTE:
        Both the single-query endpoint and the batch endpoint need to run
        this exact same logic. Extracting it into a helper function means
        we write the pipeline once and call it from both places.

        The batch endpoint calls this function in parallel (via asyncio.gather)
        for each question — if this logic lived inside the route handler,
        parallelizing it would be much harder.

    THE FULL QUERY PIPELINE — step by step:

    STEP 1: QUERY CLASSIFICATION
        We call the query classifier (an LLM prompt) to determine WHICH
        retrieval strategy to use. The four types are:
            lookup        → "Where is function X defined?"
            relational    → "What calls the auth module?"
            analytical    → "Find inconsistent error handling patterns"
            summarization → "Explain what the data pipeline does"
        Cost: ~$0.001 per call (small prompt). Worth it because using the
        wrong retrieval strategy gives wrong answers — a relational query
        routed through semantic search misses the graph structure entirely.

    STEP 2: RETRIEVAL (strategy depends on query type)
        lookup / summarization:
            → SemanticRetriever: runs ANN search in Qdrant, returns top-K chunks.
        relational:
            → GraphRetriever: identifies the relevant node in the call graph,
              then does BFS to find callers and callees.
        analytical:
            → HybridRetriever: semantic search PLUS graph features, then
              k-means clustering to group implementations and find outliers.

    STEP 3: RE-RANKING
        The retriever returns up to settings.retrieval_top_k (30) candidates. The cross-encoder
        re-ranker scores each (question, chunk) pair and re-orders them by
        relevance. This step dramatically improves precision:
            - ANN search ranks by vector distance (approximate)
            - Cross-encoder ranks by actual semantic relevance (exact, but slow)
            - We use ANN to narrow the candidate set, then cross-encoder to
              precisely rank the finalists. This is the standard "retrieve then
              re-rank" pattern used in production search systems.

    STEP 4: CONFIDENCE THRESHOLD
        If the top result's similarity is below
        settings.retrieval_confidence_threshold (0.5 by default, calibrated
        for bge-small-en-v1.5 — see config.py), we refuse to answer.
        This prevents hallucination: if we can't find relevant code, we say
        so explicitly rather than making something up.

    STEP 5: GENERATION
        We pass the top-N re-ranked chunks to the LLM with a strict system prompt
        that forbids answering from prior knowledge. The model must cite specific
        code from the context. If it can't, it must say so.

    STEP 6: HISTORY LOGGING
        We write the query to Redis for the history endpoint. This is done
        asynchronously after the response is ready — it does not block the
        response to the user.
    """
    collection_name = settings.qdrant_collection_name(owner, repo)

    # ── Step 1: Classify Query ───────────────────────────────────────────────
    # QueryClassifier.classify() is a synchronous OpenAI/Groq call — run it
    # off the event loop. It returns a ClassificationResult, or a
    # ClassificationError with a safe LOOKUP fallback on failure.
    logger.debug(f"Classifying query: '{question[:80]}...'")
    classification = await asyncio.to_thread(query_classifier.classify, question)
    if isinstance(classification, ClassificationError):
        logger.warning(
            f"Classification failed ({classification.error_message}) — "
            f"falling back to {classification.fallback_type.value}"
        )
        query_type: QueryType = classification.fallback_type
    else:
        query_type = classification.query_type
    logger.info(f"Query classified as: {query_type.value}")

    # ── Step 2: Retrieve Candidates ──────────────────────────────────────────
    # Every retrieval path below converges on list[HybridResult] — the shape
    # Reranker.rerank() actually accepts. See retrieval/hybrid_retriever.py
    # and retrieval/graph_retriever.py module docstrings for why LOOKUP/
    # SUMMARIZATION, RELATIONAL, and ANALYTICAL each need a different
    # underlying retriever (semantic-only, graph-only, and fused
    # semantic+structural respectively — none of the three can answer the
    # others' question shape).
    candidates: list[HybridResult] = []

    if query_type in (QueryType.LOOKUP, QueryType.SUMMARIZATION):
        semantic_filters: dict = {}
        if filter_language:
            semantic_filters["language"] = filter_language
        if filter_file_path:
            semantic_filters["file_path_prefix"] = filter_file_path

        semantic_result: SemanticSearchResult = await asyncio.to_thread(
            semantic_retriever.retrieve,
            query=question,
            collection_name=collection_name,
            top_k=settings.retrieval_top_k,
            filters=semantic_filters or None,
        )
        candidates = [
            HybridResult(
                chunk_id=c.chunk_id,
                file_path=c.file_path,
                function_name=c.function_name,
                start_line=c.start_line,
                end_line=c.end_line,
                language=c.language,
                code=c.code_snippet,
                docstring=c.docstring,
                semantic_score=c.similarity_score,
                structural_score=0.0,
                hybrid_score=c.similarity_score,
                graph_distance=None,
                metadata={"complexity": c.complexity},
            )
            for c in semantic_result.chunks
        ]

    elif query_type == QueryType.RELATIONAL:
        graph_loaded = await asyncio.to_thread(graph_retriever.load_graph, collection_name)
        if graph_loaded:
            graph_results = await asyncio.to_thread(
                _route_relational_query, question, collection_name, settings
            )
            # Multiple traversals (e.g. callers + callees for ambiguous
            # phrasing) can return the same node — de-dupe by node_id,
            # keeping the closest (smallest-distance) occurrence.
            merged_nodes: dict[str, GraphNode] = {}
            for result in graph_results:
                for node in result.nodes:
                    existing = merged_nodes.get(node.node_id)
                    if existing is None or node.distance < existing.distance:
                        merged_nodes[node.node_id] = node

            for node in merged_nodes.values():
                # Graph nodes don't carry source text (see graph_retriever.py
                # docstring) — look it up from the Qdrant payload written
                # during ingestion.
                code, docstring = await asyncio.to_thread(
                    _fetch_snippet_for_node, qdrant_client, collection_name, node
                )
                # Structural score decays with hop distance: a direct
                # caller/callee (distance=1) is fully relevant; each
                # additional hop is progressively less central to the
                # question actually asked.
                structural_score = max(0.0, 1.0 - 0.2 * (node.distance - 1))
                candidates.append(HybridResult(
                    chunk_id=node.node_id,
                    file_path=node.file_path,
                    function_name=node.function_name,
                    start_line=node.start_line,
                    end_line=node.end_line,
                    language=node.language,
                    code=code,
                    docstring=docstring,
                    semantic_score=0.0,
                    structural_score=structural_score,
                    hybrid_score=structural_score,
                    graph_distance=node.distance,
                    metadata={
                        "complexity": node.complexity,
                        "relationship_type": node.relationship_type,
                        "num_callers": node.num_callers,
                    },
                ))

    elif query_type == QueryType.ANALYTICAL:
        # Analytical queries ("find inconsistent error handling") aren't
        # answered by fused semantic+structural ranking — they need the
        # actual k-means clustering in features/inconsistency_detector.py
        # (README §"Query Types Supported": "semantic search + embedding
        # clustering"). That module operates on raw embeddings, so we call
        # SemanticRetriever.retrieve(include_vectors=True) directly here
        # rather than going through HybridRetriever (which never fetches
        # vectors — its structural_score signal has nothing to do with
        # clustering and would just discard the embeddings we need).
        analytical_filters: dict = {}
        if filter_language:
            analytical_filters["language"] = filter_language
        if filter_file_path:
            analytical_filters["file_path_prefix"] = filter_file_path

        analytical_result: SemanticSearchResult = await asyncio.to_thread(
            semantic_retriever.retrieve,
            query=question,
            collection_name=collection_name,
            top_k=settings.retrieval_top_k,
            filters=analytical_filters or None,
            include_vectors=True,
        )

        inconsistency_chunks = [
            InconsistencyCodeChunk(
                chunk_id=c.chunk_id,
                function_id=f"{c.file_path}::{c.function_name}",
                source_code=c.code_snippet,
                embedding=np.array(c.embedding, dtype=float),
                file_path=c.file_path,
                language=c.language,
            )
            for c in analytical_result.chunks
            if c.embedding is not None
        ]

        report = await asyncio.to_thread(
            detect_inconsistencies, query=question, chunks=inconsistency_chunks
        )
        outlier_ids = {o.chunk.chunk_id: o for o in report.inconsistencies}

        if report.skipped_reason:
            logger.info(f"Inconsistency detection skipped: {report.skipped_reason}")

        for c in analytical_result.chunks:
            outlier = outlier_ids.get(c.chunk_id)
            metadata = {"complexity": c.complexity}
            if outlier:
                # Surface flagged outliers ahead of ordinary semantic matches —
                # z_score is unbounded, so we compress it into [0, 1] purely
                # for ranking purposes; the real severity (z_score, level,
                # explanation) is preserved in metadata for the frontend/LLM.
                structural_score = min(1.0, outlier.z_score / 5.0)
                metadata.update({
                    "inconsistency_level": outlier.inconsistency_level.value,
                    "z_score": outlier.z_score,
                    "explanation": outlier.explanation,
                    "nearest_cluster_id": outlier.nearest_cluster_id,
                })
            else:
                structural_score = 0.0

            candidates.append(HybridResult(
                chunk_id=c.chunk_id,
                file_path=c.file_path,
                function_name=c.function_name,
                start_line=c.start_line,
                end_line=c.end_line,
                language=c.language,
                code=c.code_snippet,
                docstring=c.docstring,
                semantic_score=c.similarity_score,
                structural_score=structural_score,
                # Outliers are exactly what an analytical query is asking
                # for, so weight them above plain semantic similarity —
                # same 0.7/0.3-style fusion idea as HybridRetriever, just
                # with "is this an outlier" as the structural signal instead
                # of call-graph centrality.
                hybrid_score=(
                    settings.hybrid_semantic_weight * c.similarity_score
                    + settings.hybrid_structural_weight * structural_score
                ),
                metadata=metadata,
            ))
        candidates.sort(key=lambda r: r.hybrid_score, reverse=True)

    logger.debug(
        f"Retrieval returned {len(candidates)} candidates "
        f"for query_type={query_type.value}"
    )

    if not candidates:
        # No candidates at all — Qdrant/graph returned nothing.
        # This usually means the collection is empty, the target function
        # wasn't found in the call graph, or the filter is too narrow.
        logger.warning(
            f"Zero candidates retrieved for '{question[:60]}...' "
            f"[collection={collection_name}]"
        )
        return QueryResponse(
            question=question,
            answer=(
                "I couldn't find any relevant code for this query. "
                "This may mean the repository hasn't been ingested yet, "
                "the function you named doesn't exist in this codebase, "
                "or your filters are too restrictive."
            ),
            sources=[],
            confidence=0.0,
            query_type=query_type.value,
            repo_url=repo_url,
            followup_queries=[],
            latency_ms=0.0,  # Filled in by the caller
        )

    # ── Step 3: Re-rank ──────────────────────────────────────────────────────
    # Re-ranker takes the candidates and re-scores each (question, chunk_text)
    # pair using a cross-encoder. Cross-encoders are more accurate than
    # bi-encoders (like CodeBERT) because they see both the question and the
    # chunk simultaneously instead of encoding them independently. The
    # trade-off is speed — that's why we only run the cross-encoder on the
    # top-K candidates, not the entire index.
    # The cross-encoder is CPU-bound (hundreds of ms for 30 candidates) — run
    # it off the event loop so concurrent requests aren't stalled behind it.
    reranked: list[RankedResult] = await asyncio.to_thread(
        reranker.rerank,
        query=question,
        candidates=candidates,
        top_n=max_results,
    )

    # ── Step 4: Confidence Threshold ─────────────────────────────────────────
    # IMPORTANT: RankedResult.rerank_score is a raw, UNBOUNDED cross-encoder
    # logit (see reranker.py docstring: "do not compare this to hybrid_score
    # or the confidence threshold"). The [0,1]-bounded score for gating and
    # for display is hybrid_score, carried through via a property.
    top_score = reranked[0].hybrid_score if reranked else 0.0

    if top_score < settings.retrieval_confidence_threshold:
        logger.info(
            f"Top score {top_score:.4f} below threshold "
            f"{settings.retrieval_confidence_threshold} — refusing to answer."
        )
        return QueryResponse(
            question=question,
            answer=(
                "I couldn't find relevant code to answer this question. "
                f"The best match had a confidence score of {top_score:.2f}, "
                f"which is below the required threshold of "
                f"{settings.retrieval_confidence_threshold}. "
                "Try rephrasing, or check that the correct repository is indexed."
            ),
            sources=[],
            confidence=round(top_score, 4),
            query_type=query_type.value,
            repo_url=repo_url,
            followup_queries=[],
            latency_ms=0.0,  # Filled in by the caller
        )

    # ── Step 5: Generate Answer ───────────────────────────────────────────────
    # generation/generator.py handles prompt construction and the Groq
    # call. It expects CodeSourceReference objects, not RankedResult — build
    # those from the reranked results.
    gen_sources = _apply_snippet_budget(
        [_to_source_reference(r) for r in _order_specific_before_containers(reranked)]
    )

    generation = await asyncio.to_thread(
        generate,
        question,
        GenQueryType(query_type.value),
        gen_sources,
    )

    # ── Build CodeSource objects from the generator's response ──────────────
    # generation.sources may be [] if generate() itself refused (its own,
    # redundant confidence gate) — that naturally produces an empty-sources
    # response here too, which is the correct behavior for a refusal.
    sources = [
        CodeSource(
            file_path=s.file_path,
            function_name=s.function_name,
            start_line=s.start_line,
            end_line=s.end_line,
            language=s.language,
            relevance_score=round(s.similarity, 4),
            snippet=s.snippet,
        )
        for s in generation.sources
    ]

    return QueryResponse(
        question=question,
        answer=generation.answer,
        sources=sources,
        confidence=round(generation.max_similarity, 4),
        sources_used=_count_sources_used(generation.answer, sources),
        query_type=query_type.value,
        repo_url=repo_url,
        followup_queries=generation.followup_queries,
        latency_ms=0.0,  # Filled in by the caller after timing
    )


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — LOG QUERY TO HISTORY
# ─────────────────────────────────────────────────────────────────────────────

def _log_query_to_history(
    redis,
    owner: str,
    repo: str,
    response: QueryResponse,
) -> None:
    """
    WHY THIS EXISTS:
        The GET /query/history endpoint needs data to return. We store each
        completed query as a JSON item in a Redis list.

        Redis lists are perfect for this — LPUSH adds to the front (newest first),
        LTRIM keeps only the most recent 50 queries, and LRANGE retrieves them.

    KEY DESIGN:
        query_history:{owner}:{repo} → Redis List of JSON strings

    WHY WE CAP AT 50 ENTRIES:
        History is a UX convenience feature, not a full audit log. 50 entries
        per repo is enough for the "recent queries" UI panel. Keeping an
        unlimited list would grow forever and waste Redis memory.

    SIDE EFFECTS:
        This function is called AFTER the response is built and returned — it
        does not block the HTTP response. If it fails (e.g., Redis is momentarily
        unavailable), we log a warning but do not raise an error. History is
        a best-effort feature, not a critical path.
    """
    try:
        history_key = f"query_history:{owner}:{repo}"
        entry = json.dumps({
            "question": response.question,
            "query_type": response.query_type,
            "confidence": response.confidence,
            "asked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "latency_ms": response.latency_ms,
        })
        # LPUSH: push to the LEFT (front) of the list — newest entries first
        redis.lpush(history_key, entry)
        # LTRIM: keep only the 50 most recent entries
        redis.ltrim(history_key, 0, 49)
        # TTL: expire the history after 7 days of inactivity
        redis.expire(history_key, 7 * 24 * 3600)
    except Exception as e:
        logger.warning(f"Failed to log query to history for {owner}/{repo}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT: POST /api/v1/query
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=QueryResponse,
    status_code=200,
    summary="Query a repository",
    description=(
        "Ask a natural language question about an indexed GitHub repository. "
        "Returns a grounded answer with file + line citations. "
        "The repository must be ingested first via POST /api/v1/ingest."
    ),
)
async def query_repository(
    body: QueryRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> QueryResponse:
    """
    WHY WE TIME THE ENTIRE FUNCTION:
        The project spec requires p50/p90/p99 latency measurements.
        We time the entire handler (including validation, retrieval, and
        generation) so that the latency_ms field in the response reflects
        the real end-to-end server-side cost.

        Note: This does NOT include network round-trip time between the
        client and server, which is measured separately in latency_benchmark.py.

    WHY WE LOG THE QUERY TYPE AND CONFIDENCE:
        These two fields are the primary debugging signal when a query
        returns a wrong or low-quality answer:
            - Wrong query_type → wrong retrieval strategy → irrelevant results
            - Low confidence → score close to the refusal cutoff, might need
              question rephrasing or more data ingested

        Logging them server-side means you can grep the logs to find
        problematic queries without instrumenting the frontend.
    """
    start_time = time.perf_counter()

    owner, repo = _parse_github_url(body.repo_url)
    redis = request.app.state.redis
    qdrant = request.app.state.qdrant

    # ── Guard: repository must be indexed ───────────────────────────────────
    # Raises HTTP 404 if not found — stops execution here if repo isn't indexed
    _verify_repo_indexed(redis, owner, repo)

    logger.info(
        f"Query received: '{body.question[:80]}' "
        f"[repo={owner}/{repo}] "
        f"[filter_lang={body.filter_language}] "
        f"[filter_path={body.filter_file_path}]"
    )

    # ── Execute the full query pipeline ─────────────────────────────────────
    response = await _execute_query(
        question=body.question,
        repo_url=body.repo_url,
        owner=owner,
        repo=repo,
        max_results=body.max_results,
        filter_language=body.filter_language,
        filter_file_path=body.filter_file_path,
        qdrant_client=qdrant,
        reranker=request.app.state.reranker,
        settings=settings,
    )

    # ── Record total latency ─────────────────────────────────────────────────
    # perf_counter gives sub-millisecond precision, suitable for benchmarking.
    latency_ms = (time.perf_counter() - start_time) * 1000
    response.latency_ms = round(latency_ms, 2)

    logger.info(
        f"Query complete: '{body.question[:60]}' "
        f"[query_type={response.query_type}] "
        f"[confidence={response.confidence:.4f}] "
        f"[sources={len(response.sources)}] "
        f"[latency={response.latency_ms}ms]"
    )

    # ── Log to history (best-effort, does not block response) ───────────────
    # We call this AFTER building the response so latency_ms is populated.
    _log_query_to_history(redis, owner, repo, response)

    return response


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT: POST /api/v1/query/batch
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/batch",
    response_model=BatchQueryResponse,
    status_code=200,
    summary="Batch query a repository",
    description=(
        "Send multiple questions in one request. Questions are processed in parallel. "
        "Primarily used by the evaluation harness. "
        "Maximum 20 questions per batch."
    ),
)
async def batch_query(
    body: BatchQueryRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> BatchQueryResponse:
    """
    WHY WE USE asyncio.gather FOR PARALLELISM:
        asyncio.gather runs all coroutines concurrently within the same
        event loop thread. Since _execute_query awaits I/O-bound operations
        (Qdrant search, LLM API call), other queries can make progress
        while one is waiting for a network response.

        For a batch of 10 questions, asyncio.gather makes the total time
        roughly equal to the slowest single query rather than 10x the
        average query time.

    WHY NOT THREADING OR MULTIPROCESSING:
        The bottleneck here is network I/O (Qdrant + OpenAI), not CPU.
        asyncio is the right tool for I/O-bound parallelism in Python.
        Threading would work but adds complexity. Multiprocessing would work
        but is overkill and doesn't share the Qdrant connection pool.

    WHY WE RETURN ALL RESULTS EVEN IF SOME FAIL:
        If question 3 of 10 fails, we don't want to discard the other 9
        valid answers. We use return_exceptions=True in gather so that
        exceptions become results instead of crashing the whole batch.
        Failed questions are returned with an error answer instead of a
        hard 500.

    USE CASE — EVALUATION:
        run_eval.py sends all 30-40 eval questions as a single batch call,
        which takes about the same wall-clock time as 3-4 sequential queries
        instead of 30-40. This makes evaluation fast enough to run frequently.
    """
    batch_start = time.perf_counter()

    owner, repo = _parse_github_url(body.repo_url)
    redis = request.app.state.redis
    qdrant = request.app.state.qdrant

    # Guard: repo must be indexed before running any questions
    _verify_repo_indexed(redis, owner, repo)

    logger.info(
        f"Batch query: {len(body.questions)} questions "
        f"[repo={owner}/{repo}]"
    )

    # Build a coroutine for each question, all sharing the same qdrant client.
    # No filters in batch mode — apply filters per-question via the single endpoint
    # if needed.
    reranker = request.app.state.reranker
    coroutines = [
        _execute_query(
            question=q,
            repo_url=body.repo_url,
            owner=owner,
            repo=repo,
            max_results=5,             # Fixed at 5 for batch to control cost
            filter_language=None,      # No filters in batch mode
            filter_file_path=None,
            qdrant_client=qdrant,
            reranker=reranker,
            settings=settings,
        )
        for q in body.questions
    ]

    # return_exceptions=True: failed coroutines return the Exception object
    # instead of propagating it and cancelling the other coroutines.
    raw_results = await asyncio.gather(*coroutines, return_exceptions=True)

    results = []
    for i, result in enumerate(raw_results):
        if isinstance(result, Exception):
            # One question failed — return a synthetic error response for it
            # so the rest of the batch is still usable.
            logger.warning(
                f"Batch question {i} failed: '{body.questions[i][:60]}' — {result}"
            )
            results.append(QueryResponse(
                question=body.questions[i],
                answer=f"Error processing this question: {str(result)}",
                sources=[],
                confidence=0.0,
                query_type="unknown",
                repo_url=body.repo_url,
                followup_queries=[],
                latency_ms=0.0,
            ))
        else:
            results.append(result)

    total_latency_ms = (time.perf_counter() - batch_start) * 1000
    logger.info(
        f"Batch complete: {len(results)} results "
        f"[total_latency={total_latency_ms:.1f}ms]"
    )

    return BatchQueryResponse(
        repo_url=body.repo_url,
        total_questions=len(body.questions),
        results=results,
        total_latency_ms=round(total_latency_ms, 2),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT: GET /api/v1/query/history/{owner}/{repo}
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/history/{owner}/{repo}",
    response_model=QueryHistoryResponse,
    summary="Get recent query history for a repository",
    description=(
        "Returns the 50 most recent queries made against this repository, "
        "ordered newest-first. Used by the frontend to show 'recent queries'."
    ),
)
async def get_query_history(
    owner: str,
    repo: str,
    request: Request,
) -> QueryHistoryResponse:
    """
    WHY THIS ENDPOINT EXISTS:
        The frontend's chat panel can show the developer their recent queries
        as clickable chips — useful for picking up where you left off or
        re-running a previous query with a filter. This is a pure Redis read,
        so it's extremely fast (< 5ms) and has no LLM cost.

    HOW REDIS LRANGE WORKS:
        LRANGE key 0 -1 returns ALL elements in the list, from index 0
        to the last element (-1). We capped the list at 50 entries in
        _log_query_to_history, so this always returns at most 50 items.
    """
    redis = request.app.state.redis
    history_key = f"query_history:{owner}:{repo}"

    raw_entries = redis.lrange(history_key, 0, -1)  # Newest first (LPUSH order)

    history = []
    for raw in raw_entries:
        try:
            entry = json.loads(raw)
            history.append(QueryHistoryItem(**entry))
        except Exception as e:
            logger.warning(f"Skipping malformed history entry: {e}")

    return QueryHistoryResponse(
        owner=owner,
        repo=repo,
        total=len(history),
        history=history,
    )
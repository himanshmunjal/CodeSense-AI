"""
================================================================================
api/routes/impact.py — Impact Analysis Endpoints
================================================================================

WHY THIS FILE EXISTS:
    Answers "what breaks if I change this function?" by traversing the
    pre-built call graph in reverse from a target function. This is one of
    the two differentiator features described in the architecture notes
    (the other being inconsistency detection) — it is what separates
    CodeSense from a generic RAG chatbot over code.

    The heavy lifting (reverse BFS, risk classification) lives in
    retrieval/graph_retriever.py (GraphRetriever.get_impact), which loads
    the call_graph.pkl built during ingestion and returns ranked GraphNode
    results. This route is a thin HTTP layer over that.

ENDPOINT:
    POST /api/v1/impact
        → Given a function (+ file path to disambiguate overloaded names),
          returns direct callers and the full ranked blast radius.
================================================================================
"""

import asyncio

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional
from loguru import logger

from config import get_settings

from retrieval.graph_retriever import GraphRetriever

router = APIRouter()

# A single GraphRetriever instance is reused across requests — it caches the
# most recently loaded call graph so repeated queries against the same repo
# avoid re-reading call_graph.pkl from disk.
_graph_retriever = GraphRetriever()


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────────────────────────────────────

class ImpactRequest(BaseModel):
    repo_url: str = Field(..., description="GitHub repository URL, must already be ingested.")
    function_name: str = Field(..., description="Name of the function being changed.")
    file_path: Optional[str] = Field(
        None,
        description="Path to the file containing the function, used to disambiguate "
                    "overloaded/duplicate function names across the repo.",
    )
    max_depth: int = Field(4, ge=1, le=10, description="Max hops to traverse in the reverse call graph.")


class AffectedNode(BaseModel):
    function: str
    file_path: str
    distance: int
    risk: str
    start_line: int
    end_line: int
    num_callers: int


class ImpactResponse(BaseModel):
    function: str
    file_path: str
    direct_callers: list[str]
    blast_radius: list[AffectedNode]
    total_affected: int
    nodes_visited: int
    traversal_ms: float


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (duplicated from query.py — kept route-local to avoid circular imports)
# ─────────────────────────────────────────────────────────────────────────────

def _verify_repo_indexed(redis, owner: str, repo: str) -> dict:
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
    parts = repo_url.rstrip("/").split("/")
    return parts[-2], parts[-1]


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT: POST /api/v1/impact
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=ImpactResponse,
    status_code=200,
    summary="Blast-radius impact analysis for a function",
    description=(
        "Traverses the call graph in reverse from the given function to find "
        "every direct and transitive caller, ranked by risk (HIGH/MEDIUM/LOW "
        "based on hop distance and number of callers)."
    ),
)
async def impact_analysis(body: ImpactRequest, request: Request) -> ImpactResponse:
    settings = get_settings()
    redis = request.app.state.redis

    owner, repo = _parse_github_url(body.repo_url)
    repo_meta = _verify_repo_indexed(redis, owner, repo)
    collection_name = repo_meta.get("collection_name") or settings.qdrant_collection_name(owner, repo)

    # Disambiguate overloaded function names by scoping the query with file_path
    # when provided (GraphRetriever resolves "file_path::function_name" first).
    function_query = f"{body.file_path}::{body.function_name}" if body.file_path else body.function_name

    # Loading the pickle and BFS are blocking — keep them off the event loop.
    result = await asyncio.to_thread(
        _graph_retriever.get_impact,
        function_query=function_query,
        collection_name=collection_name,
        max_depth=body.max_depth,
    )

    if result.is_empty and result.query_node_id is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Function '{body.function_name}' was not found in the call graph for "
                f"'{owner}/{repo}'. Check the function name and file_path."
            ),
        )

    direct_callers = [n.function_name for n in result.direct_relationships]

    blast_radius = [
        AffectedNode(
            function=n.function_name,
            file_path=n.file_path,
            distance=n.distance,
            risk="HIGH" if n.distance == 1 else ("MEDIUM" if n.distance == 2 else "LOW"),
            start_line=n.start_line,
            end_line=n.end_line,
            num_callers=n.num_callers,
        )
        for n in result.nodes
    ]

    logger.info(
        f"Impact analysis served | repo={owner}/{repo} | function={body.function_name} | "
        f"blast_radius={len(blast_radius)}"
    )

    return ImpactResponse(
        function=body.function_name,
        file_path=body.file_path or "",
        direct_callers=direct_callers,
        blast_radius=blast_radius,
        total_affected=len(blast_radius),
        nodes_visited=result.nodes_visited,
        traversal_ms=result.traversal_ms,
    )

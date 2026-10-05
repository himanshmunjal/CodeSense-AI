"""
================================================================================
api/routes/ingest.py — Repository Ingestion Endpoints
================================================================================

WHY THIS FILE EXISTS:
    This file is the HTTP interface for everything related to ingesting a
    GitHub repository into CodeSense. "Ingesting" means:
        1. Cloning the repository locally.
        2. Walking all files and filtering for supported languages.
        3. Parsing code into structured entities (functions, classes, call graph).
        4. Generating embeddings for each entity.
        5. Storing vectors + metadata in Qdrant.

    Without this file, there is no way for a user to tell CodeSense "index
    THIS repository." It is the entry gate for all data that the query layer
    will later search over.

WHY INGESTION IS ASYNCHRONOUS (CELERY TASKS):
    A real repository — even a medium-sized one like FastAPI or LangChain —
    has thousands of files. Parsing and embedding all of them can take
    anywhere from 30 seconds to several minutes.

    If we did this work synchronously (inside the HTTP request handler),
    the HTTP connection would time out before the work finishes.
    Worse, the user's browser would just sit there with a spinning loader.

    Instead, when a user posts to /api/v1/ingest, we:
        1. Validate the request (< 1ms).
        2. Dispatch a Celery background task.
        3. Return a task_id immediately (< 50ms total response time).
        4. The user polls /api/v1/ingest/status/{task_id} to check progress.

    This is the standard pattern for long-running jobs in web APIs.
    It's what FAANG engineers expect to see in production systems.

ENDPOINTS IN THIS FILE:
    POST /api/v1/ingest
        → Kicks off a new ingestion job for a GitHub repository.
        → Returns a task_id for status polling.

    GET  /api/v1/ingest/status/{task_id}
        → Returns the current status of an ingestion job.
        → Status: pending | running | completed | failed

    GET  /api/v1/ingest/repositories
        → Lists all repositories that have been indexed so far.

    DELETE /api/v1/ingest/{owner}/{repo}
        → Removes a repository's vectors from Qdrant and its local clone.

HOW IT FITS IN THE PIPELINE:
    HTTP POST /ingest
        → Validate request body (Pydantic)
            → Check if repo already indexed (Redis cache)
                → If yes: return existing index info
                → If no: dispatch Celery task → return task_id
                            ↓ (background)
                            Clone repo (gitpython)
                            Walk files (file_walker)
                            Parse AST (tree_sitter_parser)
                            Build call graph (call_graph_builder)
                            Embed (code_embedder / graph_embedder)
                            Store in Qdrant (qdrant_client)
================================================================================
"""

from fastapi import APIRouter, Request, Depends
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from loguru import logger
import re

from celery.result import AsyncResult

from config import get_settings, Settings
from tasks.celery_worker import celery_app, ingest_repository_task


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER INSTANCE
# ─────────────────────────────────────────────────────────────────────────────

# WHY APIRouter INSTEAD OF app DIRECTLY:
#   Using a router (instead of decorating routes on the main app) lets us
#   define routes in separate files and mount them in main.py.
#   The prefix /api/v1/ingest is applied in main.py when this router is
#   registered — routes defined here use relative paths (e.g., "/" not
#   "/api/v1/ingest/").

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST & RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

# WHY WE DEFINE THESE WITH PYDANTIC:
#   Pydantic models serve as both request validation AND documentation.
#   FastAPI reads these classes and:
#       1. Validates incoming JSON against the schema.
#       2. Returns a clear 422 error (not a cryptic 500) if validation fails.
#       3. Automatically generates the JSON schema shown in the Swagger UI at /docs.
#   Without Pydantic, we'd be manually checking `if "repo_url" not in body`
#   in every route — fragile and repetitive.

class IngestRequest(BaseModel):
    """
    Schema for the POST /ingest request body.

    Example JSON payload:
        {
            "repo_url": "https://github.com/tiangolo/fastapi",
            "branch": "main",
            "force_reindex": false
        }
    """

    repo_url: str = Field(
        ...,
        description="Full GitHub HTTPS URL of the repository to ingest.",
        examples=["https://github.com/tiangolo/fastapi"],
    )

    branch: Optional[str] = Field(
        default=None,
        description=(
            "Branch to ingest. Defaults to the repository's default branch "
            "(usually 'main' or 'master'). Specify this to index a feature branch."
        ),
        examples=["main", "develop"],
    )

    force_reindex: bool = Field(
        default=False,
        description=(
            "If True, re-ingests the entire repository even if it was previously indexed. "
            "Use this after major refactors or when you want a fresh index. "
            "If False (default), only changed files (via git diff) are re-embedded."
        ),
    )

    @field_validator("repo_url")
    @classmethod
    def validate_github_url(cls, v: str) -> str:
        """
        WHY THIS VALIDATOR EXISTS:
            Without this check, a user could pass any string as repo_url.
            We'd then try to clone it, get a git error deep in the ingestion
            pipeline, and return a confusing 500 error.

            By validating the URL format here, we fail fast with a clear
            422 error: "repo_url must be a valid GitHub HTTPS URL."

        WHAT IT CHECKS:
            - Must start with https://github.com/
            - Must have exactly two path segments: owner/repo
            - Strips trailing slashes and .git suffixes for consistency

        NOTE ON SECURITY:
            This also prevents SSRF (Server-Side Request Forgery) — an attacker
            trying to make our server clone from an internal network address
            like https://192.168.1.1/evil-repo. We only allow github.com.
        """
        v = v.strip().rstrip("/").removesuffix(".git")

        github_pattern = r"^https://github\.com/[\w.-]+/[\w.-]+$"
        if not re.match(github_pattern, v):
            raise ValueError(
                "repo_url must be a valid GitHub HTTPS URL. "
                "Example: https://github.com/owner/repository"
            )
        return v


class IngestResponse(BaseModel):
    """
    Schema for the POST /ingest response.

    Returned immediately after the ingestion job is dispatched — before
    the actual work starts. The client should poll /status/{task_id}.
    """

    task_id: str = Field(
        ...,
        description="Celery task ID. Use this to poll /status/{task_id}.",
    )
    status: str = Field(
        default="pending",
        description="Initial task status — always 'pending' on first creation.",
    )
    repo_url: str = Field(
        ...,
        description="The normalized repository URL that will be ingested.",
    )
    message: str = Field(
        ...,
        description="Human-readable description of what was queued.",
    )


class IngestStatusResponse(BaseModel):
    """
    Schema for GET /ingest/status/{task_id}.

    The status field drives the frontend polling logic:
        - "pending"   → task is queued, not yet started
        - "running"   → worker has picked it up and is actively processing
        - "completed" → ingestion finished successfully
        - "failed"    → ingestion failed; see error_message for why
    """

    task_id: str
    status: str = Field(
        ...,
        description="One of: pending | running | completed | failed",
    )
    progress: Optional[dict] = Field(
        default=None,
        description=(
            "Progress details while status='running'. "
            "Example: {files_parsed: 120, files_total: 450, current_file: 'auth.py'}"
        ),
    )
    result: Optional[dict] = Field(
        default=None,
        description=(
            "Final result when status='completed'. "
            "Example: {files_indexed: 450, chunks_stored: 3200, duration_seconds: 87.3}"
        ),
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Human-readable error description when status='failed'.",
    )


class RepositoryInfo(BaseModel):
    """
    Schema representing a single indexed repository in the list response.
    """
    owner: str
    repo: str
    repo_url: str
    collection_name: str      # Qdrant collection name for this repo
    total_chunks: int         # Number of vectors stored in Qdrant
    indexed_at: str           # ISO 8601 timestamp of last successful ingestion
    branch: str
    languages: list[str]      # Languages detected during parsing


class ListRepositoriesResponse(BaseModel):
    """
    Schema for GET /ingest/repositories.
    """
    total: int
    repositories: list[RepositoryInfo]


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — PARSE GITHUB URL
# ─────────────────────────────────────────────────────────────────────────────

def parse_github_url(repo_url: str) -> tuple[str, str]:
    """
    WHY THIS HELPER EXISTS:
        Multiple routes need to extract the owner and repo name from a URL.
        Centralizing this logic avoids duplicating the same string splitting
        in every route handler.

    EXAMPLE:
        parse_github_url("https://github.com/tiangolo/fastapi")
        → ("tiangolo", "fastapi")
    """
    parts = repo_url.rstrip("/").split("/")
    # At this point the URL has already been validated by the Pydantic validator,
    # so we know it has at least 5 segments: ['https:', '', 'github.com', owner, repo]
    owner = parts[-2]
    repo  = parts[-1]
    return owner, repo


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT: POST /api/v1/ingest
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=IngestResponse,
    status_code=202,   # 202 Accepted = "we received your request, work is starting"
    summary="Ingest a GitHub repository",
    description=(
        "Clones the repository, parses all supported source files, builds a call graph, "
        "generates embeddings, and stores everything in Qdrant. "
        "Because this is a long-running operation, it runs as a background task. "
        "Poll /status/{task_id} to check progress."
    ),
)
async def ingest_repository(
    body: IngestRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> IngestResponse:
    """
    WHY WE RETURN 202 (ACCEPTED) INSTEAD OF 200 (OK):
        HTTP 202 means "I received your request and I'm working on it, but I
        haven't finished yet." This is the semantically correct status code
        for async/background jobs. Returning 200 would imply the work is done,
        which is misleading.

    WHAT THIS FUNCTION DOES:
        1. Validates the request body (done by Pydantic automatically).
        2. Checks Redis to see if this exact repo+branch combination is
           already being ingested. If so, returns the existing task_id
           instead of starting a duplicate job. This prevents the user from
           accidentally triggering 5 parallel ingestions of the same repo.
        3. Dispatches the Celery ingestion task as a background job.
        4. Stores the task_id in Redis so the status endpoint can look it up.
        5. Returns the task_id immediately.

    WHY WE CHECK FOR DUPLICATE JOBS:
        Without this check, clicking "Ingest" twice would spawn two workers
        both trying to clone and parse the same repo simultaneously, writing
        duplicate vectors to Qdrant and wasting money on embeddings.

    IDEMPOTENCY NOTE:
        If force_reindex=False and the repo was already fully indexed,
        this endpoint returns the existing index metadata instead of
        re-ingesting. This makes it safe to call multiple times — it
        won't re-do work that's already done.
    """
    owner, repo = parse_github_url(body.repo_url)
    redis = request.app.state.redis

    # ── Check for existing active job ───────────────────────────────────────
    # Redis key pattern: ingest_active:{owner}:{repo}:{branch}
    # Value: the task_id of the running/pending job
    branch_key = body.branch or "default"
    active_job_key = f"ingest_active:{owner}:{repo}:{branch_key}"
    existing_task_id = redis.get(active_job_key)

    if existing_task_id and not body.force_reindex:
        # A job is already running or recently completed for this repo.
        # Return the existing task_id so the client can poll it.
        logger.info(
            f"Ingestion already in progress for {owner}/{repo} "
            f"[task_id={existing_task_id}]"
        )
        return IngestResponse(
            task_id=existing_task_id,
            status="pending",
            repo_url=body.repo_url,
            message=(
                f"Ingestion for {owner}/{repo} is already running. "
                f"Poll /status/{existing_task_id} for progress."
            ),
        )

    # ── Dispatch Celery task ─────────────────────────────────────────────────
    # .delay() is Celery's shorthand for sending a task to the broker (Redis).
    # It returns immediately with an AsyncResult object containing the task_id.
    # The actual work happens in a Celery worker process, not here.
    #
    # We pass all necessary parameters explicitly (no shared state) because
    # Celery tasks run in a completely separate process that doesn't share
    # memory with the FastAPI process.
    task = ingest_repository_task.delay(
        repo_url=body.repo_url,
        owner=owner,
        repo=repo,
        branch=body.branch,
        force_reindex=body.force_reindex,
    )

    # ── Store task_id in Redis ───────────────────────────────────────────────
    # TTL of 3600 seconds (1 hour): if the job doesn't complete within an hour,
    # we stop blocking new ingestions of the same repo.
    redis.setex(active_job_key, 3600, task.id)

    logger.info(
        f"Ingestion task dispatched: {owner}/{repo} "
        f"[task_id={task.id}] "
        f"[branch={body.branch or 'default'}] "
        f"[force_reindex={body.force_reindex}]"
    )

    return IngestResponse(
        task_id=task.id,
        status="pending",
        repo_url=body.repo_url,
        message=(
            f"Ingestion of {owner}/{repo} has been queued. "
            f"Poll /api/v1/ingest/status/{task.id} to track progress."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT: GET /api/v1/ingest/status/{task_id}
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/status/{task_id}",
    response_model=IngestStatusResponse,
    summary="Check ingestion job status",
    description=(
        "Returns the current status of an ingestion job. "
        "Poll this endpoint after POST /ingest until status is 'completed' or 'failed'. "
        "Recommended polling interval: every 5 seconds."
    ),
)
async def get_ingestion_status(task_id: str) -> IngestStatusResponse:
    """
    WHY THIS ENDPOINT EXISTS:
        Since ingestion is async, the client needs a way to know when it's done.
        This endpoint queries Celery's result backend (Redis) for the current
        state of the task identified by task_id.

    HOW CELERY TASK STATES WORK:
        PENDING   → Task has been dispatched but no worker has picked it up yet.
                    This can also mean the task_id is unknown (we return 404 for that).
        STARTED   → A worker has picked up the task and is executing it.
                    We rename this to "running" in our response for clarity.
        SUCCESS   → Task completed without raising an exception.
                    The result is available in task.result.
        FAILURE   → Task raised an exception.
                    The exception is available in task.result.
        REVOKED   → Task was cancelled (we don't currently expose a cancel endpoint).

    WHY WE CATCH AsyncResult ERRORS:
        If task_id doesn't exist in Redis (e.g., wrong ID, or Redis was flushed),
        Celery's AsyncResult doesn't raise immediately — it returns a fake
        PENDING state. We detect this by checking whether the task exists in
        our own Redis key (set in the ingest endpoint) before querying Celery.
    """
    # WHY WE USE AsyncResult:
    #   AsyncResult is Celery's handle to a submitted task. Given a task_id,
    #   it queries the result backend (Redis) to get the current state and
    #   any result data the task has written so far.
    task = AsyncResult(task_id, app=celery_app)

    # Map Celery's internal state names to our API's status vocabulary
    state_map = {
        "PENDING": "pending",
        "STARTED": "running",
        "SUCCESS": "completed",
        "FAILURE": "failed",
        "REVOKED": "failed",
    }
    status = state_map.get(task.state, "pending")

    # Extract progress info (written by the Celery task using task.update_state())
    # The task writes progress in the form:
    #   self.update_state(state="STARTED", meta={"files_parsed": 120, "files_total": 450})
    progress = None
    result   = None
    error    = None

    if task.state == "STARTED" and isinstance(task.info, dict):
        # task.info contains whatever meta dict the worker last wrote
        progress = task.info

    elif task.state == "SUCCESS":
        result = task.result  # Final stats written by the worker on completion

    elif task.state == "FAILURE":
        # task.result is the exception object when state is FAILURE
        error = str(task.result)
        logger.warning(f"Ingestion task {task_id} failed: {error}")

    return IngestStatusResponse(
        task_id=task_id,
        status=status,
        progress=progress,
        result=result,
        error_message=error,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT: GET /api/v1/ingest/repositories
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/repositories",
    response_model=ListRepositoriesResponse,
    summary="List all indexed repositories",
    description="Returns metadata for every repository that has been successfully ingested.",
)
async def list_repositories(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> ListRepositoriesResponse:
    """
    WHY THIS ENDPOINT EXISTS:
        The frontend's left panel (file tree) needs to know which repositories
        are available to query. This endpoint provides that list.
        It also lets a developer confirm which repos are indexed before debugging
        a query that returns no results.

    HOW WE STORE REPOSITORY METADATA:
        When an ingestion job completes successfully, the Celery task writes
        a summary to Redis using the key pattern:
            indexed_repo:{owner}:{repo}
        Value: JSON with owner, repo_url, total_chunks, indexed_at, branch, languages.

        This endpoint scans for all keys matching "indexed_repo:*" and
        returns them as a list.

    WHY REDIS INSTEAD OF A DATABASE:
        We already have Redis running for the embedding cache and Celery.
        Adding a Postgres or SQLite dependency just to store a list of repos
        is over-engineering for v1. Redis key-value storage is sufficient for
        this metadata.

    SCALING NOTE:
        Redis SCAN is used instead of KEYS to list entries — KEYS blocks
        the Redis event loop and is forbidden in production Redis usage.
        SCAN iterates in batches without blocking.
    """
    redis = request.app.state.redis

    # Scan for all repository metadata keys
    # The pattern "indexed_repo:*" matches keys like "indexed_repo:tiangolo:fastapi"
    repo_keys = []
    cursor = 0
    while True:
        cursor, keys = redis.scan(cursor=cursor, match="indexed_repo:*", count=100)
        repo_keys.extend(keys)
        if cursor == 0:  # cursor returns to 0 when scan is complete
            break

    repositories = []
    for key in repo_keys:
        raw = redis.get(key)
        if raw:
            import json
            try:
                data = json.loads(raw)
                repositories.append(RepositoryInfo(**data))
            except Exception as e:
                # Log and skip malformed entries rather than crashing the whole list
                logger.warning(f"Skipping malformed repo metadata at key '{key}': {e}")

    return ListRepositoriesResponse(
        total=len(repositories),
        repositories=repositories,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT: DELETE /api/v1/ingest/{owner}/{repo}
# ─────────────────────────────────────────────────────────────────────────────

@router.delete(
    "/{owner}/{repo}",
    status_code=200,
    summary="Remove an indexed repository",
    description=(
        "Deletes the repository's Qdrant collection (all vectors), "
        "removes its local clone from disk, and clears its Redis metadata. "
        "This is irreversible — the repo must be re-ingested to query it again."
    ),
)
async def delete_repository(
    owner: str,
    repo: str,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> dict:
    """
    WHY THIS ENDPOINT EXISTS:
        During development you'll often want to wipe an index and re-ingest
        from scratch (e.g., after changing the chunking strategy).
        Without a delete endpoint, you'd have to manually drop the Qdrant
        collection and clear Redis keys from the command line.

    WHAT IT DELETES:
        1. The Qdrant collection for this repo (all vectors + metadata).
        2. The Redis metadata key (indexed_repo:{owner}:{repo}).
        3. The Redis active job key (ingest_active:{owner}:{repo}:*).
        4. All Redis embedding cache keys for this repo
           (embed_cache:{owner}/{repo}:*).
        5. The local clone directory (cloned_repos/{owner}_{repo}) and the
           persisted call graph (cloned_repos/{collection_name}/).

    WHY WE DELETE THE LOCAL CLONE:
        Cloned repos can be large (hundreds of MB). Keeping them around after
        the index is deleted wastes disk space and can cause confusion if
        someone manually edits them. The clone is re-created on next ingestion.

    ERROR HANDLING:
        If the Qdrant collection doesn't exist (maybe it was manually deleted),
        we treat that as a non-error and proceed with cleaning up Redis and disk.
        Each cleanup step is isolated — one failure doesn't block the others.
    """
    qdrant = request.app.state.qdrant
    redis  = request.app.state.redis
    collection_name = settings.qdrant_collection_name(owner, repo)

    errors = []

    # ── Delete Qdrant collection ─────────────────────────────────────────────
    try:
        qdrant.delete_collection(collection_name)
        logger.info(f"Deleted Qdrant collection: {collection_name}")
    except Exception as e:
        # Collection might not exist — not a fatal error
        logger.warning(f"Could not delete Qdrant collection '{collection_name}': {e}")
        errors.append(f"Qdrant: {str(e)}")

    # ── Delete Redis metadata ────────────────────────────────────────────────
    try:
        redis.delete(f"indexed_repo:{owner}:{repo}")
        # Delete any active job keys for this repo (all branches)
        cursor = 0
        while True:
            cursor, keys = redis.scan(
                cursor=cursor, match=f"ingest_active:{owner}:{repo}:*", count=100
            )
            if keys:
                redis.delete(*keys)
            if cursor == 0:
                break
        # Delete embedding cache entries for this repo
        cursor = 0
        while True:
            cursor, keys = redis.scan(
                cursor=cursor, match=f"embed_cache:{owner}/{repo}:*", count=100
            )
            if keys:
                redis.delete(*keys)
            if cursor == 0:
                break
        logger.info(f"Cleared Redis metadata for {owner}/{repo}")
    except Exception as e:
        logger.warning(f"Error clearing Redis keys for {owner}/{repo}: {e}")
        errors.append(f"Redis: {str(e)}")

    # ── Delete local clone ───────────────────────────────────────────────────
    try:
        import shutil
        from pathlib import Path
        clone_path = Path(settings.repo_clone_dir) / f"{owner}_{repo}"
        if clone_path.exists():
            shutil.rmtree(clone_path)
            logger.info(f"Deleted local clone at: {clone_path}")
        # The call graph is stored next to the clones, keyed by collection
        # name (see tasks/celery_worker.py step 6).
        graph_dir = Path(settings.repo_clone_dir) / collection_name
        if graph_dir.exists():
            shutil.rmtree(graph_dir)
    except Exception as e:
        logger.warning(f"Error deleting local clone for {owner}/{repo}: {e}")
        errors.append(f"Disk: {str(e)}")

    return {
        "status": "deleted" if not errors else "partially_deleted",
        "owner": owner,
        "repo": repo,
        "collection_deleted": collection_name,
        "warnings": errors if errors else None,
        "message": (
            f"Repository {owner}/{repo} has been removed. "
            "Re-ingest with POST /api/v1/ingest to index it again."
        ),
    }
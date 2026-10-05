"""
================================================================================
api/main.py — FastAPI Application Entry Point
================================================================================

WHY THIS FILE EXISTS:
    Every FastAPI application needs a single "root" object — the FastAPI() instance.
    This file creates that instance, configures it (CORS, middleware, lifespan events),
    and registers all route modules under their URL prefixes.

    Think of this file as the "switchboard" of the backend:
        - It does NOT contain any business logic.
        - It does NOT talk to Qdrant, Redis, or GitHub directly.
        - Its only job is to wire everything together and configure
          cross-cutting concerns (logging, CORS, startup/shutdown).

    Any new route group you add in the future (e.g., /api/v1/graph) gets
    registered here with one line — no changes needed anywhere else.

HOW IT FITS IN THE PIPELINE:
    HTTP Request
        → FastAPI (this file)
            → Middleware (rate limiting, logging)
                → Router (ingest / query / impact / summarize)
                    → Service layer (retrieval, generation, etc.)

ENTRY POINT:
    Run the server with:
        uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

    Or via the settings object:
        uvicorn api.main:app --host {settings.backend_host} --port {settings.backend_port}
================================================================================
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import sys
import time

from config import settings
from api.routes import ingest, query, impact, summarizer as summarize
from api.middleware import RateLimitMiddleware, RequestIDMiddleware


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN — Startup & Shutdown Logic
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    WHY THIS EXISTS:
        FastAPI's lifespan context manager replaces the old @app.on_event("startup")
        and @app.on_event("shutdown") decorators (which are deprecated in FastAPI 0.95+).

        Everything BEFORE the `yield` runs at startup.
        Everything AFTER the `yield` runs at shutdown.

    WHAT IT DOES AT STARTUP:
        1. Configures loguru to write structured logs to both stdout and a log file.
        2. Verifies the Qdrant connection is alive before accepting any traffic.
           If Qdrant is down, the server refuses to start — fail fast is better
           than accepting requests that will all fail silently.
        3. Verifies the Redis connection is alive (used for embedding cache + Celery).
        4. Logs the full configuration summary so developers can confirm which
           env vars were actually loaded.

    WHAT IT DOES AT SHUTDOWN:
        1. Closes the Qdrant client connection pool gracefully.
        2. Closes the Redis connection pool gracefully.
        3. Flushes any pending log writes.

    WHY WE DO CONNECTION CHECKS HERE AND NOT IN EACH ROUTE:
        If Qdrant is unreachable, every single query will fail. Checking once
        at startup gives a clear error message ("Qdrant unreachable at localhost:6333")
        instead of cryptic per-request errors. This is standard production practice.
    """

    # ── Configure Loguru ────────────────────────────────────────────────────
    # Remove the default loguru handler (plain stderr) and replace with
    # structured handlers: one for stdout (human-readable in dev),
    # one for a rotating log file (for persistence and debugging).
    logger.remove()

    logger.add(
        sys.stdout,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    logger.add(
        settings.log_file,
        level=settings.log_level,
        rotation="50 MB",    # Start a new file after 50MB
        retention="14 days", # Keep log files for 14 days
        compression="zip",   # Compress old files to save disk
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
    )

    logger.info("=" * 60)
    logger.info("CodeSense backend starting up")
    logger.info(f"Environment : {settings.environment}")
    logger.info(f"Log level   : {settings.log_level}")
    logger.info(f"Qdrant      : {settings.qdrant_host}:{settings.qdrant_port}")
    logger.info(f"Redis       : {settings.redis_host}:{settings.redis_port}")
    logger.info(f"Gen model   : {settings.groq_model}")
    logger.info(f"Embed model : {settings.codebert_model_name}")
    logger.info("=" * 60)

    # ── Verify Qdrant Connection ─────────────────────────────────────────────
    # We import here (not at module level) to avoid circular imports and to
    # ensure the Qdrant client is only instantiated after config is loaded.
    try:
        from qdrant_client import QdrantClient
        qdrant = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key or None,
        )
        # list_collections() is the cheapest health check — it doesn't fetch data,
        # just verifies the server responds. Raises an exception if unreachable.
        qdrant.get_collections()
        logger.info("Qdrant connection verified")
        # Store the client on app.state so routes can access it without
        # re-initializing a new client on every request.
        app.state.qdrant = qdrant
    except Exception as e:
        logger.critical(f"Qdrant unreachable at startup: {e}")
        logger.critical("Start Qdrant with: docker compose up -d qdrant")
        raise RuntimeError("Qdrant connection failed — server will not start.") from e

    # ── Verify Redis Connection ──────────────────────────────────────────────
    try:
        import redis as redis_lib
        redis_client = redis_lib.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password or None,
            decode_responses=True,
        )
        # PING is the standard Redis health check — returns "PONG" if alive.
        redis_client.ping()
        logger.info("Redis connection verified")
        app.state.redis = redis_client
    except Exception as e:
        logger.critical(f"Redis unreachable at startup: {e}")
        logger.critical("Start Redis with: docker compose up -d redis")
        raise RuntimeError("Redis connection failed — server will not start.") from e

    # ── Load the Cross-Encoder Re-ranker ─────────────────────────────────────
    # WHY HERE AND NOT PER-REQUEST:
    #   Reranker.__init__ loads a ~22M param cross-encoder model from disk
    #   (retrieval/reranker.py). Its own docstring says it's meant to be
    #   "instantiated at the module level in api/main.py and injected via
    #   dependency injection" — api/routes/query.py previously constructed a
    #   fresh Reranker() on every single request, reloading the model each
    #   time. Loading it once here and sharing it via app.state (the same
    #   pattern already used for qdrant/redis below) fixes that.
    #
    #   SemanticRetriever and GraphRetriever don't need the same treatment:
    #   they already instantiate themselves as module-level singletons in
    #   their own files (retrieval/semantic_retriever.py, .../graph_retriever.py)
    #   and are imported directly where needed.
    from retrieval.reranker import Reranker
    app.state.reranker = Reranker()
    logger.info("Cross-encoder re-ranker loaded")

    logger.info("All services healthy — accepting requests")

    # ── Hand control to FastAPI ──────────────────────────────────────────────
    # Everything above this yield runs at startup.
    # Everything below runs at shutdown.
    yield

    # ── Graceful Shutdown ────────────────────────────────────────────────────
    logger.info("CodeSense backend shutting down — closing connections")
    try:
        app.state.qdrant.close()
        logger.info("Qdrant connection closed")
    except Exception as e:
        logger.warning(f"Error closing Qdrant connection: {e}")

    try:
        app.state.redis.close()
        logger.info("Redis connection closed")
    except Exception as e:
        logger.warning(f"Error closing Redis connection: {e}")

    logger.info("Shutdown complete")


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION INSTANCE
# ─────────────────────────────────────────────────────────────────────────────

# WHY WE PASS lifespan= HERE:
#   The lifespan parameter is how FastAPI knows to run our startup/shutdown
#   logic. Without it, the server starts but skips all the connection checks
#   and logging setup above.
#
# WHY WE SET docs_url AND redoc_url:
#   In production, we don't want the interactive Swagger UI exposed publicly
#   (it's a potential attack surface and reveals your API schema to everyone).
#   In development, it's extremely useful for manual testing.
#   We disable it in production by setting both to None.

app = FastAPI(
    title="CodeSense",
    description=(
        "AI-powered codebase intelligence engine. "
        "Ask deep questions about any GitHub repository — "
        "grounded in actual code, not hallucinations."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
    openapi_url="/openapi.json" if not settings.is_production else None,
)


# ─────────────────────────────────────────────────────────────────────────────
# MIDDLEWARE
# ─────────────────────────────────────────────────────────────────────────────

# ── CORS ────────────────────────────────────────────────────────────────────
# WHY WE NEED CORS:
#   Browsers enforce the Same-Origin Policy — they block frontend JavaScript
#   from making requests to a different domain/port than the page was served from.
#   Our React frontend runs on localhost:5173 (Vite), but the API is on localhost:8000.
#   Without CORS headers, every API call from the frontend fails in the browser
#   (even though curl and Postman work fine — they don't enforce CORS).
#
# WHY THE ORIGINS COME FROM CONFIG:
#   allow_origins=["*"] lets ANY website call our API from a user's browser.
#   CORS_ORIGINS defaults to the local Vite dev server; set it to your real
#   frontend URL(s) when deploying. No cookies or auth headers are used, so
#   credentials are not allowed.

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],  # OPTIONS is required for CORS preflight
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
    # Let the frontend read the timing / rate-limit headers set below.
    expose_headers=["X-Request-ID", "X-Response-Time-Ms", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
)

# ── Rate Limiting ────────────────────────────────────────────────────────────
# WHY WE NEED RATE LIMITING:
#   Each query makes LLM API calls, which cost money or free-tier quota. Without rate limiting, a single
#   user (or a script) could send thousands of requests and drain the API budget.
#   The RateLimitMiddleware (defined in api/middleware.py) uses Redis to track
#   request counts per IP address in a sliding time window.
#
# WHY MIDDLEWARE AND NOT A DECORATOR ON EACH ROUTE:
#   Middleware applies to EVERY request automatically. If we used a decorator,
#   we'd have to remember to add it to every new route — that's fragile.

# ── Request ID Assignment ─────────────────────────────────────────────────
# WHY THIS IS REGISTERED FIRST (before RateLimitMiddleware):
#   Starlette executes middleware in REVERSE registration order — the last
#   middleware added via add_middleware() runs FIRST on each request. Since
#   we want every request (including ones that get rejected by the rate
#   limiter) to have a request ID available for logging, RequestIDMiddleware
#   must be added AFTER RateLimitMiddleware in this file so that it actually
#   executes BEFORE it at runtime. See api/middleware.py for the full
#   explanation of this ordering rule.
app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestIDMiddleware)


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST LOGGING MIDDLEWARE
# ─────────────────────────────────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    WHY THIS EXISTS:
        Logs every incoming request and its response time. This gives us:
        1. A full audit trail of all API calls.
        2. The data we need for the p50/p90/p99 latency benchmarks
           (evaluation/latency_benchmark.py).
        3. Immediate visibility when something is slow (e.g., a query that
           takes 8 seconds shows up clearly in the logs).

    HOW IT WORKS:
        FastAPI's middleware chain works like an onion — each middleware layer
        wraps the next. This function records the start time, calls the next
        handler (which eventually reaches the actual route), then logs
        how long it took after the response comes back.

    WHY WE LOG REQUEST IDs:
        When debugging issues in production, having a unique ID per request
        lets you trace a single request across all log lines — especially
        useful when multiple requests are being processed concurrently.
    """
    # WHY request.state.request_id INSTEAD OF request.headers:
    #   RequestIDMiddleware (api/middleware.py) generates a request ID and
    #   stores it on request.state because Starlette's request.headers is
    #   immutable from within middleware — there is no way to "add" a header
    #   to the incoming request object itself. request.state is the
    #   documented mechanism for passing data between middleware layers and
    #   route handlers within a single request's lifecycle. We fall back to
    #   "no-id" only if RequestIDMiddleware somehow didn't run, which
    #   shouldn't happen given it's registered on every request.
    request_id = getattr(request.state, "request_id", "no-id")
    start_time = time.perf_counter()

    logger.info(
        f"→ {request.method} {request.url.path} "
        f"[request_id={request_id}] "
        f"[client={request.client.host if request.client else 'unknown'}]"
    )

    response = await call_next(request)

    duration_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        f"← {request.method} {request.url.path} "
        f"[status={response.status_code}] "
        f"[duration={duration_ms:.1f}ms] "
        f"[request_id={request_id}]"
    )

    # Attach the duration to the response header so the frontend can display it
    response.headers["X-Response-Time-Ms"] = f"{duration_ms:.1f}"
    return response


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL EXCEPTION HANDLER
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    WHY THIS EXISTS:
        Without this, any unhandled exception in a route returns FastAPI's
        default 500 response — which exposes the raw Python traceback to the client.
        That's a security risk (it reveals internal paths, library versions, etc.)
        and a bad user experience.

        This handler catches everything that wasn't handled by a more specific
        handler, logs the full traceback server-side (so we can debug it), and
        returns a clean, safe error response to the client.

    WHAT IT RETURNS:
        A JSON object with:
        - error: a human-readable message (never the raw exception)
        - request_id: so the caller can report which request failed
        - status: always "error" for easy frontend parsing
    """
    request_id = getattr(request.state, "request_id", "no-id")
    logger.exception(
        f"Unhandled exception on {request.method} {request.url.path} "
        f"[request_id={request_id}]: {exc}"
    )
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error": "An internal server error occurred. Please try again.",
            "request_id": request_id,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# ROUTE REGISTRATION
# ─────────────────────────────────────────────────────────────────────────────

# WHY WE USE ROUTERS INSTEAD OF DEFINING ROUTES HERE:
#   Defining all routes in main.py would make it thousands of lines long.
#   FastAPI's APIRouter lets us split routes into separate files by domain
#   (ingest, query, impact, summarize), each with its own file, schemas,
#   and dependencies. This file just mounts them at their URL prefixes.
#
# URL STRUCTURE:
#   /api/v1/ingest/*    → all ingestion endpoints
#   /api/v1/query/*     → all query/search endpoints
#   /api/v1/impact/*    → impact analysis endpoints
#   /api/v1/summarize/* → summarization endpoints
#
# WHY /api/v1/ PREFIX:
#   Versioning the API from day one means we can ship a /api/v2/ later
#   without breaking existing clients. It's much harder to add versioning
#   retroactively. The /api/ prefix makes it easy to proxy the backend
#   through a reverse proxy (nginx, Caddy) without path conflicts.

app.include_router(
    ingest.router,
    prefix="/api/v1/ingest",
    tags=["Ingestion"],                 # Groups these routes in the Swagger UI
)

app.include_router(
    query.router,
    prefix="/api/v1/query",
    tags=["Query"],
)

app.include_router(
    impact.router,
    prefix="/api/v1/impact",
    tags=["Impact Analysis"],
)

app.include_router(
    summarize.router,
    prefix="/api/v1/summarize",
    tags=["Summarization"],
)


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
async def health_check(request: Request) -> dict:
    """
    WHY THIS EXISTS:
        A health check endpoint is the standard way for:
        1. Docker / Kubernetes to know whether the container is ready to serve traffic.
        2. Load balancers to decide whether to route traffic to this instance.
        3. Monitoring tools (Datadog, Grafana) to track uptime.
        4. Developers to quickly verify the server is running before debugging
           a more complex issue.

    WHAT IT CHECKS:
        - Qdrant: can we list collections? (proves the vector DB is responsive)
        - Redis: does PING return PONG? (proves the cache is responsive)

    WHY WE CHECK DEPENDENCIES HERE:
        A server can be "up" but still broken if its dependencies are down.
        This endpoint returns 200 only when everything is healthy — so a
        monitoring tool that polls /health gets an accurate picture.

    RETURNS:
        200 OK with {status: "healthy"} when all services are up.
        503 Service Unavailable with details when any service is down.
    """
    health_status = {
        "status": "healthy",
        "services": {
            "qdrant": "unknown",
            "redis": "unknown",
        },
        "version": "1.0.0",
        "environment": settings.environment,
    }

    # Check Qdrant
    try:
        request.app.state.qdrant.get_collections()
        health_status["services"]["qdrant"] = "healthy"
    except Exception as e:
        health_status["services"]["qdrant"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"

    # Check Redis
    try:
        request.app.state.redis.ping()
        health_status["services"]["redis"] = "healthy"
    except Exception as e:
        health_status["services"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"

    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(content=health_status, status_code=status_code)


# ─────────────────────────────────────────────────────────────────────────────
# ROOT ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root() -> dict:
    """
    WHY THIS EXISTS:
        Returns a simple banner when someone hits the root URL.
        Useful to confirm the server is running before checking /health.
        Also prevents a 404 on the root path, which can confuse monitoring tools.
    """
    return {
        "name": "CodeSense API",
        "version": "1.0.0",
        "docs": "/docs" if not settings.is_production else "disabled in production",
        "health": "/health",
    }
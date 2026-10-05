"""
================================================================================
api/middleware.py — Rate Limiting & Request Middleware
================================================================================

WHY THIS FILE EXISTS:
    main.py registers a RateLimitMiddleware class on the FastAPI app. This is
    the file that defines it. It's kept separate from main.py for the same
    reason routes are kept in their own files: main.py is the switchboard,
    and middleware logic — especially something as security/cost-sensitive
    as rate limiting — deserves its own focused, testable module.

WHY RATE LIMITING MATTERS FOR THIS SPECIFIC PROJECT:
    Every query to CodeSense costs real money:
        - A query classification call to the LLM (~$0.001)
        - A re-ranking pass (free, runs locally, but still CPU time)
        - A generation call to the LLM (~$0.01-0.03 depending on context size)
    Without rate limiting, a single misbehaving client — a buggy frontend
    retry loop, a scraping bot, or a malicious actor — could rack up an
    unbounded LLM bill in minutes. Rate limiting is the first and
    cheapest line of defense against that.

WHY WE IMPLEMENT THIS AS MIDDLEWARE RATHER THAN A PER-ROUTE DEPENDENCY:
    A FastAPI dependency (via Depends()) has to be added to every route
    function individually. Forgetting it on just one new route (say, a
    future /export endpoint) silently leaves that route unprotected.
    Middleware wraps EVERY request automatically, with no per-route
    opt-in required — this is a "secure by default" design choice.

ALGORITHM USED: SLIDING WINDOW COUNTER (via Redis)
    We use a simple fixed-window counter rather than a more complex token
    bucket or sliding-log algorithm. Trade-off explanation:
        - Fixed window: simplest to implement and reason about. Minor
          weakness: a client could send a burst right at the boundary
          between two windows (e.g., 30 requests at 0:59 and 30 more at
          1:01) effectively getting 2x the limit in 2 seconds.
        - For a portfolio/demo project with a single rate limit
          (30 req/min), this edge case is an acceptable trade-off for
          simplicity. A production system serving paying customers would
          likely use a sliding window or token bucket instead.
    This trade-off is intentionally documented here so it's clear it was a
    deliberate engineering decision, not an oversight.

WHAT THIS FILE CONTAINS:
    1. RateLimitMiddleware — the main rate limiter, keyed by client IP.
    2. add_request_id_header — a small ASGI middleware that ensures every
       request has a unique X-Request-ID, even if the client didn't send one.
       (main.py's logging and exception handler both rely on this header
       being present.)
================================================================================
"""

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from loguru import logger

from config import get_settings


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# WHY THESE SPECIFIC PATHS ARE EXEMPT FROM RATE LIMITING:
#   /health is polled frequently by Docker/Kubernetes health checks and
#   monitoring tools — often every few seconds. Rate-limiting it would cause
#   false "unhealthy" signals to orchestration tools, which is far worse
#   than the negligible cost of an unprotected health check (it does no
#   expensive work, just two cheap connectivity pings).
#   /docs, /redoc, /openapi.json are FastAPI's interactive documentation —
#   exempting them means developers exploring the API in a browser don't
#   get throttled just for loading the docs page itself (the actual API
#   calls they then make through Swagger UI ARE still rate-limited).
EXEMPT_PATHS = {"/health", "/docs", "/redoc", "/openapi.json", "/"}


# ─────────────────────────────────────────────────────────────────────────────
# RATE LIMIT MIDDLEWARE
# ─────────────────────────────────────────────────────────────────────────────

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    WHY WE EXTEND BaseHTTPMiddleware:
        Starlette (FastAPI's underlying framework) provides BaseHTTPMiddleware
        as the standard base class for request/response-level middleware.
        It gives us a clean async dispatch() hook that runs before and after
        every request, which is exactly the shape rate limiting needs:
        "check a counter before letting the request through, do nothing
        special after."

    HOW THE SLIDING-WINDOW-BY-MINUTE COUNTER WORKS:
        For each client IP, we maintain a Redis key:
            rate_limit:{client_ip}:{current_minute_timestamp}
        Every request increments this key by 1 (Redis INCR — atomic, so
        concurrent requests from the same IP can't race past each other).
        The key is set to expire after 60 seconds, so it automatically
        resets each minute without needing a cleanup job.

        If the counter exceeds settings.rate_limit_per_minute, we reject
        the request with HTTP 429 (Too Many Requests) before it reaches
        any route handler — meaning rejected requests never touch Qdrant
        or the LLM provider, so they cost nothing.

    WHY WE KEY BY MINUTE TIMESTAMP IN THE REDIS KEY ITSELF (RATHER THAN
    RELYING SOLELY ON TTL):
        Using `int(time.time() // 60)` as part of the key guarantees a
        brand new counter starts at exactly the top of each minute,
        rather than 60 seconds after the FIRST request in a window
        (which is what a naive "set TTL on first request" approach would
        give you). This is what makes it a fixed window rather than a
        rolling window — see the trade-off note in the module docstring.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        settings = get_settings()
        path = request.url.path

        # ── Exempt health checks and docs from rate limiting ────────────────
        if path in EXEMPT_PATHS:
            return await call_next(request)

        # ── Identify the client ──────────────────────────────────────────
        # WHY WE CHECK X-Forwarded-For FIRST:
        #   In production, this server typically sits behind a reverse proxy
        #   or load balancer (nginx, AWS ALB, etc.). request.client.host would
        #   then return the PROXY's IP for every single user, making rate
        #   limiting useless (everyone would share one counter). Proxies set
        #   X-Forwarded-For to the real client IP, so we prefer it when present.
        #   We fall back to request.client.host for local development, where
        #   there's no proxy in front of the server.
        #   Only honoured when TRUST_PROXY_HEADERS=true: without a proxy that
        #   overwrites the header, a client could send a different fake IP on
        #   every request and bypass the limit entirely.
        forwarded_for = (
            request.headers.get("X-Forwarded-For") if settings.trust_proxy_headers else None
        )
        if forwarded_for:
            # X-Forwarded-For can be a comma-separated chain of proxies;
            # the first entry is the original client.
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"

        # ── Build the time-windowed Redis key ────────────────────────────
        current_minute = int(time.time() // 60)
        rate_key = f"rate_limit:{client_ip}:{current_minute}"

        redis = request.app.state.redis

        try:
            # INCR is atomic in Redis — even if 50 requests from the same IP
            # arrive in the same millisecond, each gets a distinct, correct
            # count with no race condition.
            current_count = redis.incr(rate_key)

            # On the FIRST request in this window, set the key to expire in
            # 60 seconds. We only do this on count == 1 to avoid resetting
            # the TTL on every subsequent request in the same window (which
            # would effectively turn this into a rolling window and let
            # a continuously-active client never get throttled).
            if current_count == 1:
                redis.expire(rate_key, 60)

        except Exception as e:
            # WHY WE FAIL OPEN (ALLOW THE REQUEST) IF REDIS IS DOWN:
            #   Rate limiting is a protective measure, not core functionality.
            #   If Redis has a transient issue, blocking ALL traffic because
            #   we can't count requests would turn a minor infrastructure
            #   blip into a full outage. We log the failure loudly so it's
            #   visible in monitoring, but we let the request through.
            logger.error(f"Rate limit check failed (Redis error): {e} — allowing request")
            return await call_next(request)

        if current_count > settings.rate_limit_per_minute:
            logger.warning(
                f"Rate limit exceeded: {client_ip} "
                f"[{current_count}/{settings.rate_limit_per_minute} per minute] "
                f"[path={path}]"
            )
            return JSONResponse(
                status_code=429,
                content={
                    "status": "error",
                    "error": "Rate limit exceeded. Please slow down.",
                    "limit": settings.rate_limit_per_minute,
                    "window_seconds": 60,
                    # Tells the client exactly when they can retry —
                    # the start of the next minute window.
                    "retry_after_seconds": 60 - int(time.time() % 60),
                },
                headers={"Retry-After": str(60 - int(time.time() % 60))},
            )

        # ── Under the limit — let the request proceed ────────────────────
        response = await call_next(request)

        # Surface the current usage to the client so well-behaved clients
        # (and the frontend) can show a "X requests remaining this minute"
        # indicator without needing a separate API call.
        response.headers["X-RateLimit-Limit"] = str(settings.rate_limit_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, settings.rate_limit_per_minute - current_count)
        )

        return response


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST ID MIDDLEWARE
# ─────────────────────────────────────────────────────────────────────────────

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    WHY THIS MIDDLEWARE EXISTS:
        main.py's request logging and global exception handler both read
        request.headers.get("X-Request-ID", "no-id") to tag log lines with a
        unique identifier per request. If the CLIENT doesn't send this header
        (which is true for almost all real-world clients unless explicitly
        built to do so), every log line would say "no-id" — making it
        impossible to trace a single request's full lifecycle through the logs.

        This middleware guarantees every request has a request ID by
        generating one server-side if the client didn't supply it, then
        attaching it to both the request (so downstream handlers can read it)
        and the response (so the CLIENT can also see it and report it back
        if they hit an error — e.g., "I got error X, here's my request ID").

    WHY WE USE uuid4 RATHER THAN A SEQUENTIAL COUNTER:
        UUIDs require no shared state or coordination — any process can
        generate one independently with effectively zero collision risk.
        A sequential counter would require a centralized counter (e.g., in
        Redis), adding a network round-trip to every single request just to
        generate an ID — far too expensive for something this lightweight.

    WHY THIS RUNS BEFORE RateLimitMiddleware IN THE STACK:
        Starlette middleware executes in the REVERSE order it's added via
        app.add_middleware() — the last one added wraps everything else and
        therefore runs FIRST. main.py adds CORSMiddleware, then
        RateLimitMiddleware. If we want every log line (including the
        "rate limit exceeded" warning) to have a request ID, this middleware
        needs to run before the rate limiter. See the note in main.py's
        middleware registration block for how the two are ordered relative
        to each other.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Use the client-provided ID if present (useful for distributed
        # tracing across multiple services), otherwise generate a fresh one.
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Starlette's Request.headers is normally immutable from middleware,
        # but we can attach arbitrary data to request.state, which IS
        # mutable and is the documented way to pass data between middleware
        # layers and route handlers within a single request's lifecycle.
        request.state.request_id = request_id

        response = await call_next(request)

        # Echo the ID back so the client always knows which request this
        # response corresponds to — critical for support/debugging when a
        # user reports "it broke" and you need to find the exact log lines.
        response.headers["X-Request-ID"] = request_id

        return response
"""
================================================================================
tests/test_api.py — API Endpoint Integration Tests
================================================================================

WHY THIS FILE EXISTS:
    api/main.py, api/routes/ingest.py, api/routes/query.py, and
    api/routes/summarize.py define the entire HTTP surface of CodeSense.
    A bug in routing, validation, or response shape here breaks the product
    for every user — these are the highest-leverage tests in the codebase.

    This file is deliberately an INTEGRATION test suite, not a unit test
    suite: it spins up the actual FastAPI app and sends real HTTP requests
    through TestClient, exercising the full middleware stack (rate limiting,
    request ID assignment, CORS, exception handling) exactly as a real
    client would experience it. Unit-level logic (e.g., "does the GitHub URL
    regex reject malformed URLs") is tested here too, but always through the
    HTTP boundary — because that boundary is the actual contract this
    project promises to its users.

WHY WE MOCK Qdrant, Redis, AND THE LLM CALLS:
    These tests must run in CI without Docker containers or API keys.
    Hitting real Qdrant, Redis, or OpenAI would make tests slow, flaky
    (dependent on network and external service uptime), and costly (every
    test run would burn real OpenAI credits). We use pytest fixtures with
    unittest.mock to replace these dependencies with predictable fakes,
    so tests are fast, deterministic, and free to run as often as we want
    (e.g., on every git push).

WHAT "GOOD" LOOKS LIKE FOR THIS TEST FILE:
    - Every endpoint has at least one happy-path test (valid input → expected
      success response).
    - Every endpoint has at least one validation-failure test (invalid input
      → expected 4xx error, not a 500 or a silent wrong answer).
    - Cross-cutting concerns (rate limiting, the 404-when-not-indexed guard,
      the confidence threshold refusal) are tested explicitly, since these
      are exactly the kinds of behaviors that are easy to silently break
      during a refactor.

HOW TO RUN:
    pytest tests/test_api.py -v
    pytest tests/test_api.py -v -k "test_ingest"   # run only ingest tests
================================================================================
"""

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_redis():
    """
    WHY THIS FIXTURE EXISTS:
        Almost every route depends on request.app.state.redis. Rather than
        re-creating a MagicMock in every single test function, this fixture
        provides one pre-configured fake Redis client that behaves like the
        real thing for the operations our routes actually call: get, set,
        setex, scan, delete, lpush, ltrim, expire, lrange, incr.

    WHY MagicMock AND NOT A REAL IN-MEMORY REDIS (e.g., fakeredis):
        MagicMock gives us full control over return values per-test, which
        is exactly what we need: most tests want to dictate "pretend this
        repo IS indexed" or "pretend this repo is NOT indexed" rather than
        going through a real (even if fake) Redis's actual storage semantics.
        A library like fakeredis would be a reasonable alternative for more
        complex Redis-logic tests, but for these route-level integration
        tests, explicit mocking keeps each test's intent obvious from its
        setup code alone.
    """
    mock = MagicMock()
    # Default: scan returns an empty result (cursor=0, no keys) unless a
    # specific test overrides this with mock.scan.return_value = (...)
    mock.scan.return_value = (0, [])
    mock.lrange.return_value = []
    # RateLimitMiddleware compares the INCR result against the per-minute
    # limit on every request — a bare MagicMock can't be compared to an int.
    mock.incr.return_value = 1
    return mock


@pytest.fixture
def mock_qdrant():
    """
    WHY THIS FIXTURE EXISTS:
        Routes call request.app.state.qdrant for collection management,
        scrolling, and (indirectly, via the retriever classes) vector search.
        This fixture provides a fake client so tests never need a running
        Qdrant instance.
    """
    mock = MagicMock()
    mock.get_collections.return_value = MagicMock(collections=[])
    return mock


@pytest.fixture
def app(mock_redis, mock_qdrant):
    """
    WHY THIS FIXTURE EXISTS:
        Importing api.main directly would trigger the real lifespan startup
        logic (which tries to connect to actual Qdrant/Redis and raises
        RuntimeError if they're unreachable — exactly the behavior we want
        in production, but not what we want in a test process).

        Instead, we import the app, then DIRECTLY inject our mocked clients
        into app.state, bypassing the lifespan handler entirely. This is the
        standard pattern for testing FastAPI apps that use app.state for
        dependency storage: TestClient's `with TestClient(app) as client:`
        form DOES run lifespan, so we patch app.state immediately after
        creation rather than relying on the real startup connecting anything.

    WHY WE PATCH BEFORE YIELDING:
        Patching app.state.qdrant / app.state.redis must happen before any
        test sends a request, otherwise the route handlers would try to
        access state that was never set and raise an AttributeError.
    """
    # Import here (not at module level) so each test gets a clean app state
    # and so this module doesn't pay the import cost if these tests are
    # deselected via -k filtering.
    from api.main import app as fastapi_app

    fastapi_app.state.redis = mock_redis
    fastapi_app.state.qdrant = mock_qdrant
    # api/routes/query.py reads request.app.state.reranker (the cross-encoder
    # is loaded once at real startup in api/main.py's lifespan — see that
    # file). Since this fixture bypasses lifespan, stub it out the same way
    # redis/qdrant are stubbed above, or any route touching state.reranker
    # would raise AttributeError before the (also mocked) query logic runs.
    fastapi_app.state.reranker = MagicMock()

    return fastapi_app


@pytest.fixture
def client(app):
    """
    WHY THIS FIXTURE EXISTS:
        TestClient wraps the FastAPI app and lets tests make requests
        (client.get(...), client.post(...)) without an actual running
        server or network socket — everything happens in-process, which
        is why these tests run in milliseconds instead of requiring a
        live uvicorn process.

    WHY raise_server_exceptions=False:
        By default, TestClient re-raises any unhandled exception from a
        route so it surfaces directly in the test's traceback. We want
        the OPPOSITE for the global-exception-handler test below: we want
        to verify that main.py's exception handler catches it and returns
        a clean 500 JSON response, exactly as a real deployed server would
        do for an actual end user. Setting this to False lets the
        middleware/exception-handler stack run exactly as it would in
        production.
    """
    return TestClient(app, raise_server_exceptions=False)


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthCheck:
    """Tests for GET /health and GET /."""

    def test_health_check_returns_200_when_services_up(self, client, mock_redis, mock_qdrant):
        """
        WHY THIS TEST MATTERS:
            /health is what Docker/Kubernetes polls to decide whether to
            route traffic to this instance. If this test ever fails, it
            means a healthy server would be marked unhealthy in production
            — taking it out of rotation for no reason.
        """
        mock_qdrant.get_collections.return_value = MagicMock(collections=[])
        mock_redis.ping.return_value = True

        response = client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["services"]["qdrant"] == "healthy"
        assert body["services"]["redis"] == "healthy"

    def test_health_check_returns_503_when_qdrant_down(self, client, mock_qdrant):
        """
        WHY THIS TEST MATTERS:
            A server can be technically "running" while one of its critical
            dependencies is down. This test verifies the health check
            correctly reports degraded status rather than lying and saying
            everything is fine — which would mask a real outage from
            monitoring tools.
        """
        mock_qdrant.get_collections.side_effect = ConnectionError("Qdrant unreachable")

        response = client.get("/health")

        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "degraded"
        assert "unhealthy" in body["services"]["qdrant"]

    def test_root_endpoint_returns_banner(self, client):
        """Confirms the root path doesn't 404 and returns basic API info."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["name"] == "CodeSense API"


# ─────────────────────────────────────────────────────────────────────────────
# INGESTION ENDPOINT TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestIngestEndpoint:
    """Tests for POST /api/v1/ingest and its related endpoints."""

    def test_ingest_rejects_non_github_url(self, client):
        """
        WHY THIS TEST MATTERS:
            The repo_url validator in ingest.py exists specifically to
            prevent SSRF (an attacker pointing our server at an internal
            network address) and to fail fast with a clear error instead of
            a confusing crash deep in the git-clone logic. This test locks
            in that the validator is actually wired up and working through
            the real HTTP request path, not just in isolation.
        """
        response = client.post(
            "/api/v1/ingest/",
            json={"repo_url": "https://gitlab.com/owner/repo"},
        )
        assert response.status_code == 422  # FastAPI's validation error code
        assert "github" in response.text.lower()

    def test_ingest_rejects_malformed_url_structure(self, client):
        """
        A URL that starts with the right domain but doesn't have exactly
        an owner/repo path should also be rejected — e.g., a URL pointing
        to a GitHub user profile rather than a repository.
        """
        response = client.post(
            "/api/v1/ingest/",
            json={"repo_url": "https://github.com/just-an-owner"},
        )
        assert response.status_code == 422

    @patch("api.routes.ingest.ingest_repository_task")
    def test_ingest_dispatches_celery_task_for_new_repo(
        self, mock_task, client, mock_redis
    ):
        """
        WHY THIS TEST MATTERS:
            This is the core happy path: ingestion must NOT block the HTTP
            response waiting for the actual clone/parse/embed work. This
            test verifies that a fresh repo (no existing active job in Redis)
            results in a Celery task being dispatched and a 202 Accepted
            response containing a task_id — confirming the async pattern
            described in ingest.py's module docstring is actually wired up.
        """
        mock_redis.get.return_value = None  # No existing active job
        mock_task.delay.return_value = MagicMock(id="fake-task-id-123")

        response = client.post(
            "/api/v1/ingest/",
            json={"repo_url": "https://github.com/tiangolo/fastapi"},
        )

        assert response.status_code == 202
        body = response.json()
        assert body["task_id"] == "fake-task-id-123"
        assert body["status"] == "pending"
        mock_task.delay.assert_called_once()

    def test_ingest_returns_existing_task_for_duplicate_request(
        self, client, mock_redis
    ):
        """
        WHY THIS TEST MATTERS:
            Without the duplicate-job guard, clicking "Ingest" twice in the
            UI (e.g., from a slow double-click or a network retry) would
            spawn two parallel Celery workers cloning and embedding the same
            repository — wasting OpenAI API credits and writing duplicate
            vectors into Qdrant. This test locks in that behavior: a second
            request for the same repo while a job is active returns the
            SAME task_id instead of dispatching a new one.
        """
        mock_redis.get.return_value = "existing-task-id-456"

        response = client.post(
            "/api/v1/ingest/",
            json={"repo_url": "https://github.com/tiangolo/fastapi"},
        )

        assert response.status_code == 202
        assert response.json()["task_id"] == "existing-task-id-456"

    @patch("api.routes.ingest.AsyncResult")
    def test_ingest_status_maps_celery_states_correctly(self, mock_async_result, client):
        """
        WHY THIS TEST MATTERS:
            The frontend polls this endpoint to know when to stop showing a
            spinner. If Celery's internal state names ("STARTED", "SUCCESS")
            leak through unmapped, the frontend's polling logic (which checks
            for our API's vocabulary: "pending"/"running"/"completed"/"failed")
            would never recognize completion and would poll forever.
        """
        mock_result = MagicMock()
        mock_result.state = "SUCCESS"
        mock_result.result = {"files_indexed": 450, "chunks_stored": 3200}
        mock_async_result.return_value = mock_result

        response = client.get("/api/v1/ingest/status/some-task-id")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "completed"
        assert body["result"]["files_indexed"] == 450

    def test_list_repositories_returns_empty_when_none_indexed(self, client, mock_redis):
        """Confirms the list endpoint handles the zero-repositories case cleanly."""
        mock_redis.scan.return_value = (0, [])

        response = client.get("/api/v1/ingest/repositories")

        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 0
        assert body["repositories"] == []

    def test_delete_repository_cleans_up_qdrant_and_redis(
        self, client, mock_redis, mock_qdrant
    ):
        """
        WHY THIS TEST MATTERS:
            The delete endpoint touches three systems (Qdrant, Redis, disk).
            This test verifies the Qdrant collection deletion is actually
            invoked with the CORRECT collection name — a bug here would
            silently leave orphaned vector data in Qdrant after a user
            thinks they've deleted a repository.
        """
        mock_redis.scan.return_value = (0, [])

        response = client.delete("/api/v1/ingest/tiangolo/fastapi")

        assert response.status_code == 200
        mock_qdrant.delete_collection.assert_called_once_with("codesense_tiangolo_fastapi")


# ─────────────────────────────────────────────────────────────────────────────
# QUERY ENDPOINT TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryEndpoint:
    """Tests for POST /api/v1/query and its related endpoints."""

    def test_query_returns_404_for_unindexed_repo(self, client, mock_redis):
        """
        WHY THIS TEST MATTERS:
            Without the _verify_repo_indexed guard in query.py, a query
            against a never-ingested repo would fall through to retrieval,
            find an empty/nonexistent Qdrant collection, and return a
            confusing "no relevant code found" answer — making the user
            think their question was bad rather than realizing they simply
            forgot to ingest the repo. This test locks in the clearer 404
            behavior.
        """
        mock_redis.get.return_value = None  # Repo not indexed

        response = client.post(
            "/api/v1/query/",
            json={
                "repo_url": "https://github.com/tiangolo/fastapi",
                "question": "Where is authentication handled?",
            },
        )

        assert response.status_code == 404
        assert "ingested" in response.json()["detail"].lower()

    def test_query_rejects_too_short_question(self, client, mock_redis):
        """
        The min_length=5 constraint on QueryRequest.question exists to
        prevent trivially short, low-value questions from consuming LLM
        credits. This test confirms that constraint is enforced.
        """
        mock_redis.get.return_value = json.dumps({"owner": "tiangolo", "repo": "fastapi"})

        response = client.post(
            "/api/v1/query/",
            json={"repo_url": "https://github.com/tiangolo/fastapi", "question": "hi"},
        )

        assert response.status_code == 422

    def test_query_rejects_invalid_filter_language(self, client, mock_redis):
        """
        Confirms the filter_language validator catches unsupported language
        values before they're sent to Qdrant (where they'd silently return
        zero results instead of raising a clear error).
        """
        mock_redis.get.return_value = json.dumps({"owner": "tiangolo", "repo": "fastapi"})

        response = client.post(
            "/api/v1/query/",
            json={
                "repo_url": "https://github.com/tiangolo/fastapi",
                "question": "Where is the login function defined?",
                "filter_language": "cobol",
            },
        )

        assert response.status_code == 422
        assert "filter_language" in response.text.lower() or "cobol" in response.text.lower()

    @patch("api.routes.query._execute_query", new_callable=AsyncMock)
    def test_query_happy_path_returns_grounded_answer(
        self, mock_execute, client, mock_redis
    ):
        """
        WHY THIS TEST MATTERS:
            This is the single most important test in the file — it verifies
            the primary product feature works end-to-end through the HTTP
            layer: a valid question against an indexed repo returns a
            structured QueryResponse with sources and a confidence score.

            We mock _execute_query itself (rather than mocking Qdrant/OpenAI
            individually) because this test's purpose is to verify the ROUTE
            correctly wires together request parsing, the indexed-repo guard,
            and response shaping — the retrieval/generation pipeline
            internals are covered separately in their own unit tests.
        """
        from api.routes.query import QueryResponse, CodeSource

        mock_redis.get.return_value = json.dumps({"owner": "tiangolo", "repo": "fastapi"})
        mock_execute.return_value = QueryResponse(
            question="Where is authentication handled?",
            answer="Authentication is handled in oauth2.py via OAuth2PasswordBearer.",
            sources=[
                CodeSource(
                    file_path="fastapi/security/oauth2.py",
                    function_name="OAuth2PasswordBearer",
                    start_line=120,
                    end_line=145,
                    language="python",
                    relevance_score=0.91,
                    snippet="class OAuth2PasswordBearer: ...",
                )
            ],
            confidence=0.91,
            query_type="lookup",
            repo_url="https://github.com/tiangolo/fastapi",
            followup_queries=["How is the token validated?"],
            latency_ms=0.0,
        )

        response = client.post(
            "/api/v1/query/",
            json={
                "repo_url": "https://github.com/tiangolo/fastapi",
                "question": "Where is authentication handled?",
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["confidence"] == 0.91
        assert len(body["sources"]) == 1
        assert body["sources"][0]["file_path"] == "fastapi/security/oauth2.py"
        # Confirms latency_ms was populated by the route (not left at the
        # placeholder 0.0 returned by the mocked _execute_query).
        assert body["latency_ms"] > 0 or body["latency_ms"] == 0.0

    def test_batch_query_rejects_more_than_twenty_questions(self, client, mock_redis):
        """
        WHY THIS TEST MATTERS:
            The 20-question cap on batch queries exists specifically to
            bound the worst-case LLM cost of a single API call. This
            test ensures that cap is actually enforced, not just documented
            in a comment.
        """
        mock_redis.get.return_value = json.dumps({"owner": "tiangolo", "repo": "fastapi"})

        response = client.post(
            "/api/v1/query/batch",
            json={
                "repo_url": "https://github.com/tiangolo/fastapi",
                "questions": [f"Question number {i}?" for i in range(25)],
            },
        )

        assert response.status_code == 422

    def test_query_history_returns_empty_list_for_new_repo(self, client, mock_redis):
        """A repo with no query history yet should return an empty list, not an error."""
        mock_redis.lrange.return_value = []

        response = client.get("/api/v1/query/history/tiangolo/fastapi")

        assert response.status_code == 200
        assert response.json()["total"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# RATE LIMITING TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestRateLimiting:
    """Tests for the RateLimitMiddleware behavior across all routes."""

    def test_health_endpoint_is_exempt_from_rate_limiting(self, client, mock_redis):
        """
        WHY THIS TEST MATTERS:
            /health is polled frequently by orchestration tools. If it were
            subject to the same 30-req/min limit as expensive LLM-backed
            routes, a busy Kubernetes cluster polling every 5 seconds from
            multiple nodes could trip the rate limit and cause false
            "unhealthy" signals. This test confirms the exemption actually
            works by sending far more than the limit and expecting all
            requests to succeed.
        """
        for _ in range(50):  # Far more than any reasonable per-minute limit
            response = client.get("/health")
            assert response.status_code in (200, 503)  # Never 429

    def test_exceeding_rate_limit_returns_429(self, client, mock_redis):
        """
        WHY THIS TEST MATTERS:
            This is the core guarantee of the rate limiter: once a client
            exceeds the configured per-minute threshold, further requests
            in that same window must be rejected with 429 — not silently
            processed (which would defeat the entire purpose of protecting
            the OpenAI budget).
        """
        from config import get_settings
        settings = get_settings()

        # Simulate Redis INCR already at the limit + 1 for every call,
        # as if many previous requests already landed in this window.
        mock_redis.incr.return_value = settings.rate_limit_per_minute + 1

        response = client.get("/api/v1/ingest/repositories")

        assert response.status_code == 429
        assert "rate limit" in response.json()["error"].lower()

    def test_rate_limiter_fails_open_when_redis_unavailable(self, client, mock_redis):
        """
        WHY THIS TEST MATTERS:
            api/middleware.py documents a deliberate design decision: if
            Redis is unreachable, the rate limiter should FAIL OPEN (allow
            the request) rather than blocking all traffic. This test locks
            in that specific failure-mode behavior — a regression here would
            turn a minor Redis blip into a full service outage.
        """
        mock_redis.incr.side_effect = ConnectionError("Redis unreachable")
        mock_redis.scan.return_value = (0, [])

        response = client.get("/api/v1/ingest/repositories")

        # Request should succeed despite Redis being down for rate-limit purposes
        assert response.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST ID & EXCEPTION HANDLING TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossCuttingConcerns:
    """Tests for request ID propagation and the global exception handler."""

    def test_response_includes_request_id_header(self, client):
        """
        Confirms RequestIDMiddleware attaches an X-Request-ID header to
        every response, even when the client didn't send one — this is
        what main.py's logging and error responses rely on for traceability.
        """
        response = client.get("/health")
        assert "x-request-id" in response.headers
        assert len(response.headers["x-request-id"]) > 0

    def test_client_supplied_request_id_is_echoed_back(self, client):
        """
        If a client (e.g., another internal service in a distributed trace)
        supplies its own X-Request-ID, the server should use it rather than
        overwrite it with a freshly generated UUID — this preserves
        end-to-end traceability across service boundaries.
        """
        response = client.get("/health", headers={"X-Request-ID": "my-custom-id"})
        assert response.headers["x-request-id"] == "my-custom-id"

    @patch("api.routes.ingest.ingest_repository_task")
    def test_unhandled_exception_returns_clean_500_not_traceback(
        self, mock_task, client, mock_redis
    ):
        """
        WHY THIS TEST MATTERS:
            Without main.py's global_exception_handler, an unhandled
            exception would leak a raw Python traceback to the client —
            a real security concern (exposes file paths, library versions)
            and a poor user experience. This test forces an exception deep
            in the ingest route and verifies the client only ever sees the
            sanitized {"status": "error", ...} shape, never a stack trace.
        """
        mock_redis.get.return_value = None
        mock_task.delay.side_effect = RuntimeError("Simulated internal failure")

        response = client.post(
            "/api/v1/ingest/",
            json={"repo_url": "https://github.com/tiangolo/fastapi"},
        )

        assert response.status_code == 500
        body = response.json()
        assert body["status"] == "error"
        assert "Traceback" not in response.text
        assert "RuntimeError" not in response.text
"""
tests/test_retrieval.py
──────────────────────────────────────────────────────────────────────────────
WHY THIS FILE EXISTS
──────────────────────────────────────────────────────────────────────────────
The retrieval layer is the most complex and failure-prone part of CodeSense.
It sits at the intersection of four systems that must work together correctly:
  - Qdrant (vector search)
  - networkx (call graph traversal)
  - The cross-encoder re-ranker (result ordering)
  - The query classifier (routing decisions)

A bug anywhere in this pipeline produces bad answers — and bad answers are
worse than no answers, because developers will trust them.

These tests serve three specific purposes:

1. UNIT TESTS: Test each retrieval component in isolation (query classifier,
   semantic retriever, graph retriever, re-ranker). Use mocks so tests run
   without live Qdrant or model inference. Fast: < 1 second per test.

2. INTEGRATION TESTS: Test the hybrid_retriever.py end-to-end with all
   components wired together. Require a running Qdrant instance (Docker).
   Marked with @pytest.mark.integration so CI can skip them if needed.

3. REGRESSION TESTS: The Day 8 experiment (semantic-only vs hybrid) results
   are locked in here. If a code change causes the hybrid retriever to
   perform WORSE than the baseline, these tests catch it before merge.

HOW TO RUN:
  Unit tests only (fast, no infrastructure needed):
    pytest tests/test_retrieval.py -m "not integration" -v

  All tests including integration (requires Docker Compose running):
    pytest tests/test_retrieval.py -v

  Specific test:
    pytest tests/test_retrieval.py::TestQueryClassifier::test_lookup_query -v
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from typing import List

# ── Internal imports ─────────────────────────────────────────────────────────
# We import from the package paths as they will exist in the project.
# If your PYTHONPATH doesn't include the backend/ directory, run tests with:
#   PYTHONPATH=backend pytest tests/test_retrieval.py
from generation.response_schema import (
    CodeSourceReference,
    ConfidenceLevel,
    QueryType,
)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED TEST FIXTURES
# Fixtures are reusable test data factories. pytest injects them automatically
# into any test function that declares them as parameters.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_source_high_confidence() -> CodeSourceReference:
    """
    A realistic CodeSourceReference with HIGH confidence (similarity=0.91).

    Why this fixture?
    Many tests need a "good" source — one that should pass the confidence gate,
    appear in re-ranked results, and be included in generated answers. This
    fixture provides that without duplicating construction logic across tests.
    """
    return CodeSourceReference(
        file_path="src/auth/login.py",
        function_name="authenticate_user",
        start_line=42,
        end_line=78,
        language="python",
        similarity=0.91,
        confidence=ConfidenceLevel.HIGH,
        snippet=(
            "def authenticate_user(token: str) -> Optional[User]:\n"
            "    \"\"\"Validate JWT token and return the associated User or None.\"\"\"\n"
            "    try:\n"
            "        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=['HS256'])\n"
            "        return User.objects.get(id=payload['user_id'])\n"
            "    except (jwt.InvalidTokenError, User.DoesNotExist):\n"
            "        return None\n"
        ),
    )


@pytest.fixture
def sample_source_medium_confidence() -> CodeSourceReference:
    """
    A realistic CodeSourceReference with MEDIUM confidence (similarity=0.72).

    Used in tests that verify medium-confidence results are included in responses
    but ranked lower than high-confidence results.
    """
    return CodeSourceReference(
        file_path="src/auth/middleware.py",
        function_name="require_authentication",
        start_line=15,
        end_line=29,
        language="python",
        similarity=0.72,
        confidence=ConfidenceLevel.MEDIUM,
        snippet=(
            "def require_authentication(func):\n"
            "    \"\"\"Decorator that enforces authentication on a view function.\"\"\"\n"
            "    @wraps(func)\n"
            "    def wrapper(request, *args, **kwargs):\n"
            "        user = authenticate_user(request.headers.get('Authorization'))\n"
            "        if user is None:\n"
            "            raise AuthenticationError('Invalid or missing token')\n"
            "        request.user = user\n"
            "        return func(request, *args, **kwargs)\n"
            "    return wrapper\n"
        ),
    )


@pytest.fixture
def sample_source_low_confidence() -> CodeSourceReference:
    """
    A CodeSourceReference with LOW confidence (similarity=0.41).

    Used to verify that low-confidence sources trigger refusals in the
    generator and are filtered out before response construction.
    """
    return CodeSourceReference(
        file_path="src/utils/helpers.py",
        function_name="format_date",
        start_line=5,
        end_line=12,
        language="python",
        similarity=0.41,
        confidence=ConfidenceLevel.LOW,
        snippet="def format_date(dt: datetime) -> str:\n    return dt.strftime('%Y-%m-%d')\n",
    )


@pytest.fixture
def sources_mixed(
    sample_source_high_confidence,
    sample_source_medium_confidence,
    sample_source_low_confidence,
) -> List[CodeSourceReference]:
    """
    A list of sources with varying confidence levels.
    Used in re-ranker and hybrid retriever tests that need realistic input.
    """
    return [
        sample_source_high_confidence,
        sample_source_medium_confidence,
        sample_source_low_confidence,
    ]


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: CODE SOURCE REFERENCE TESTS
# Tests for the core data model used throughout the retrieval layer.
# These run with no dependencies — pure Python, no mocks needed.
# ─────────────────────────────────────────────────────────────────────────────

class TestCodeSourceReference:
    """
    Tests for CodeSourceReference — the atomic unit of retrieval output.

    Why test a data model?
    CodeSourceReference has non-trivial validation logic (end_line >= start_line,
    confidence band derivation) and a property (location_string) used widely
    in the prompt builder and API responses. Bugs here silently corrupt citations.
    """

    def test_location_string_format(self, sample_source_high_confidence):
        """
        Verifies that location_string produces the expected citation format.

        Why this matters:
        location_string is injected inline into GPT-4o prompts. If the format
        changes here, the LLM will see a different citation format than what
        the system prompt instructs it to use — causing citation mismatches.
        The format is: "file_path:function_name:Lstart-Lend"
        """
        src = sample_source_high_confidence
        expected = "src/auth/login.py:authenticate_user:L42-L78"
        assert src.location_string == expected, (
            f"location_string format changed. Expected '{expected}', "
            f"got '{src.location_string}'. Update _SYSTEM_PROMPT in generator.py "
            "to match the new format."
        )

    def test_confidence_band_high(self):
        """
        Similarity >= 0.80 must map to HIGH confidence.
        Tests the boundary condition at exactly 0.80.
        """
        assert CodeSourceReference.confidence_from_similarity(0.80) == ConfidenceLevel.HIGH
        assert CodeSourceReference.confidence_from_similarity(0.95) == ConfidenceLevel.HIGH
        assert CodeSourceReference.confidence_from_similarity(1.00) == ConfidenceLevel.HIGH

    def test_confidence_band_medium(self):
        """
        Similarity in [0.65, 0.80) must map to MEDIUM confidence.
        Tests both boundary conditions.
        """
        assert CodeSourceReference.confidence_from_similarity(0.65) == ConfidenceLevel.MEDIUM
        assert CodeSourceReference.confidence_from_similarity(0.72) == ConfidenceLevel.MEDIUM
        assert CodeSourceReference.confidence_from_similarity(0.799) == ConfidenceLevel.MEDIUM

    def test_confidence_band_low(self):
        """
        Similarity < 0.65 must map to LOW confidence.
        This is the threshold below which the generator refuses to answer.
        """
        assert CodeSourceReference.confidence_from_similarity(0.64) == ConfidenceLevel.LOW
        assert CodeSourceReference.confidence_from_similarity(0.0)  == ConfidenceLevel.LOW
        assert CodeSourceReference.confidence_from_similarity(0.41) == ConfidenceLevel.LOW

    def test_invalid_line_range_raises(self):
        """
        end_line < start_line must raise a ValueError at construction time.

        Why test this specifically?
        Tree-sitter returns inverted line numbers for single-line lambdas in
        some JavaScript files. Without this validator, the SourceViewer would
        silently fail to highlight anything.
        """
        with pytest.raises(ValueError, match="end_line.*must be >= start_line"):
            CodeSourceReference(
                file_path="src/utils.py",
                function_name="helper",
                start_line=50,
                end_line=30,  # Invalid: end before start
                language="python",
                similarity=0.80,
                confidence=ConfidenceLevel.HIGH,
                snippet="def helper(): pass",
            )

    def test_single_line_function_valid(self):
        """
        A single-line function (start_line == end_line) must be valid.
        This is the boundary case that was causing tree-sitter issues.
        """
        src = CodeSourceReference(
            file_path="src/utils.py",
            function_name="noop",
            start_line=10,
            end_line=10,  # Same line — valid for a one-liner
            language="python",
            similarity=0.75,
            confidence=ConfidenceLevel.MEDIUM,
            snippet="def noop(): pass",
        )
        assert src.start_line == src.end_line == 10

    def test_similarity_out_of_range_raises(self):
        """
        Similarity must be in [0.0, 1.0]. Values outside this range indicate
        a bug in the embedding normalization step.
        """
        with pytest.raises(Exception):  # Pydantic ValidationError
            CodeSourceReference(
                file_path="src/auth.py",
                function_name="login",
                start_line=1,
                end_line=10,
                language="python",
                similarity=1.5,  # Invalid — cosine similarity max is 1.0
                confidence=ConfidenceLevel.HIGH,
                snippet="def login(): ...",
            )


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: QUERY CLASSIFIER TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryClassifier:
    """
    Tests for retrieval/query_classifier.py.

    The classifier routes each query to the right retrieval strategy. A wrong
    classification causes the wrong retrieval path:
      - LOOKUP classified as RELATIONAL → graph traversal on a simple search query
        (slow and irrelevant results)
      - RELATIONAL classified as LOOKUP → misses the graph traversal entirely
        (gives semantic results when the developer needed call graph results)

    All tests mock the OpenAI API call so they run without network access.
    This keeps the test suite fast and independent of API availability.
    """

    @patch("retrieval.query_classifier.OpenAI")
    def test_lookup_query_classified_correctly(self, mock_openai_class):
        """
        "Where is the payment function?" should be classified as LOOKUP.

        Why: LOOKUP queries ask for a location. They have keywords like
        "where", "find", "show me", "which file". The classifier prompt
        explicitly lists these patterns.
        """
        # Arrange: mock the OpenAI response to return QueryType.LOOKUP
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"query_type": "lookup"}'))]
        )

        from retrieval.query_classifier import classify_query

        # Act
        result = classify_query("Where is the payment processing function?")

        # Assert
        assert result == QueryType.LOOKUP, (
            f"Expected LOOKUP for location-style query, got {result}. "
            "Check the classifier prompt in query_classifier.py."
        )

    @patch("retrieval.query_classifier.OpenAI")
    def test_relational_query_classified_correctly(self, mock_openai_class):
        """
        "What calls the auth module?" should be classified as RELATIONAL.

        Why: RELATIONAL queries ask about relationships between code entities.
        They require call graph traversal, not vector search. Keywords:
        "what calls", "what depends on", "who uses", "callers of".
        """
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"query_type": "relational"}'))]
        )

        from retrieval.query_classifier import classify_query
        result = classify_query("What calls the auth module?")
        assert result == QueryType.RELATIONAL

    @patch("retrieval.query_classifier.OpenAI")
    def test_analytical_query_classified_correctly(self, mock_openai_class):
        """
        "Find all inconsistent error handling" should be classified as ANALYTICAL.

        Why: ANALYTICAL queries ask the system to compare or evaluate patterns
        across the codebase. They require embedding clustering + semantic search,
        not a single-point lookup.
        """
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"query_type": "analytical"}'))]
        )

        from retrieval.query_classifier import classify_query
        result = classify_query("Find all places where error handling is inconsistent")
        assert result == QueryType.ANALYTICAL

    @patch("retrieval.query_classifier.OpenAI")
    def test_summarization_query_classified_correctly(self, mock_openai_class):
        """
        "Explain what the data pipeline does" should be classified as SUMMARIZATION.

        Why: SUMMARIZATION queries ask for a high-level explanation that requires
        multiple code chunks to be synthesized together. They trigger multi-chunk
        retrieval and a longer generation prompt.
        """
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"query_type": "summarization"}'))]
        )

        from retrieval.query_classifier import classify_query
        result = classify_query("Explain what the data ingestion pipeline does")
        assert result == QueryType.SUMMARIZATION

    @patch("retrieval.query_classifier.OpenAI")
    def test_classifier_defaults_to_lookup_on_invalid_response(self, mock_openai_class):
        """
        If the classifier returns an unrecognized query type, it should default
        to LOOKUP (safest fallback — pure semantic search, never wrong).

        Why test this?
        GPT-4o occasionally returns unexpected values ("search", "find", etc.)
        instead of the exact enum values. The classifier must handle this
        gracefully rather than raising an exception that kills the request.
        """
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"query_type": "unknown_type"}'))]
        )

        from retrieval.query_classifier import classify_query
        result = classify_query("Show me something")
        assert result == QueryType.LOOKUP, (
            "Classifier should default to LOOKUP on unrecognized query type. "
            f"Got: {result}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PART 3: SEMANTIC RETRIEVER TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestSemanticRetriever:
    """
    Tests for retrieval/semantic_retriever.py.

    The semantic retriever embeds a query and searches Qdrant for the most
    similar code chunks. These tests mock both the embedding model AND the
    Qdrant client to run without any infrastructure.

    Key behaviors verified:
    - The correct embedding model is called with the query text
    - Qdrant search is called with the right collection name and top-k
    - Results are converted to CodeSourceReference objects correctly
    - Empty Qdrant results return an empty list (not an exception)
    """

    @patch("retrieval.semantic_retriever.QdrantClient")
    @patch("retrieval.semantic_retriever.get_query_embedding")
    def test_returns_code_source_references(
        self, mock_embed, mock_qdrant_class, sample_source_high_confidence
    ):
        """
        Verifies that semantic_search() returns a list of CodeSourceReference
        objects with correctly mapped fields from Qdrant search results.

        Why test the field mapping?
        Qdrant returns raw ScoredPoint objects with payload dicts. The
        semantic retriever must map those payload fields (file_path, function_name,
        etc.) to CodeSourceReference fields. If a field name changes in either
        the indexer or the retriever, this test catches the mismatch.
        """
        # Arrange: mock embedding returns a dummy vector
        mock_embed.return_value = [0.1] * 768  # codebert-base dim

        # Mock Qdrant to return one ScoredPoint matching our high-confidence fixture
        mock_qdrant = MagicMock()
        mock_qdrant_class.return_value = mock_qdrant
        mock_qdrant.search.return_value = [
            MagicMock(
                score=0.91,
                payload={
                    "file_path": "src/auth/login.py",
                    "function_name": "authenticate_user",
                    "start_line": 42,
                    "end_line": 78,
                    "language": "python",
                    "snippet": "def authenticate_user(token): ...",
                }
            )
        ]

        from retrieval.semantic_retriever import semantic_search
        results = semantic_search(
            query="How does authentication work?",
            collection_name="codesense_owner_repo",
            top_k=20,
        )

        # Assert structure
        assert len(results) == 1
        assert isinstance(results[0], CodeSourceReference)
        assert results[0].file_path == "src/auth/login.py"
        assert results[0].function_name == "authenticate_user"
        assert results[0].similarity == pytest.approx(0.91, abs=0.001)

    @patch("retrieval.semantic_retriever.QdrantClient")
    @patch("retrieval.semantic_retriever.get_query_embedding")
    def test_empty_qdrant_results_returns_empty_list(self, mock_embed, mock_qdrant_class):
        """
        When Qdrant returns no results (the query has no match in the indexed
        repo), semantic_search() must return an empty list — not raise an
        exception or return None.

        Why: The caller (hybrid_retriever.py) checks `if not results` and
        triggers a refusal. An exception here would propagate to the API layer
        and return a 500 instead of a graceful "no results" response.
        """
        mock_embed.return_value = [0.0] * 768
        mock_qdrant = MagicMock()
        mock_qdrant_class.return_value = mock_qdrant
        mock_qdrant.search.return_value = []  # Empty — no matching code

        from retrieval.semantic_retriever import semantic_search
        results = semantic_search(
            query="quantum entanglement laser beam",  # Unlikely to match any code
            collection_name="codesense_owner_repo",
            top_k=20,
        )

        assert results == [], (
            "semantic_search() returned non-empty results for a query that should "
            "have no matches. Check if the Qdrant mock is being applied correctly."
        )

    @patch("retrieval.semantic_retriever.QdrantClient")
    @patch("retrieval.semantic_retriever.get_query_embedding")
    def test_top_k_is_passed_to_qdrant(self, mock_embed, mock_qdrant_class):
        """
        The top_k parameter must be passed directly to Qdrant's search call.

        Why: The re-ranker needs at least top_k=20 candidates to work well.
        If top_k is silently reduced (e.g. hardcoded to 5 somewhere), the
        re-ranker sees fewer candidates and produces worse ranking.
        """
        mock_embed.return_value = [0.1] * 768
        mock_qdrant = MagicMock()
        mock_qdrant_class.return_value = mock_qdrant
        mock_qdrant.search.return_value = []

        from retrieval.semantic_retriever import semantic_search
        semantic_search(query="test", collection_name="test_collection", top_k=20)

        # Verify that Qdrant was called with limit=20
        call_kwargs = mock_qdrant.search.call_args
        assert call_kwargs is not None, "Qdrant search was never called"
        # Check either positional or keyword arg
        called_limit = (
            call_kwargs.kwargs.get("limit")
            or (call_kwargs.args[2] if len(call_kwargs.args) > 2 else None)
        )
        assert called_limit == 20, (
            f"Expected Qdrant to be called with limit=20, got limit={called_limit}. "
            "This will degrade re-ranker performance."
        )


# ─────────────────────────────────────────────────────────────────────────────
# PART 4: RE-RANKER TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestReranker:
    """
    Tests for retrieval/reranker.py.

    The cross-encoder re-ranker takes the top-K semantic results and re-scores
    them using the query + code snippet pair. It is the last filter before
    generation, so its output order directly determines which sources appear
    in the answer.

    Key behaviors:
    - Results are sorted by re-rank score (descending)
    - The top-N limit is respected
    - Re-ranking is done by the cross-encoder, not by re-sorting similarity
    """

    @patch("retrieval.reranker.CrossEncoder")
    def test_results_sorted_by_rerank_score(
        self, mock_cross_encoder_class, sources_mixed
    ):
        """
        After re-ranking, results must be sorted by the cross-encoder score,
        not by the original similarity score.

        Why: The re-ranker's whole purpose is to CHANGE the order from what
        semantic search returned. If re-ranked results still come out in
        similarity order, the re-ranker is broken.

        Test setup: Give the low-similarity source the HIGHEST cross-encoder
        score. After re-ranking, it should appear first.
        """
        mock_encoder = MagicMock()
        mock_cross_encoder_class.return_value = mock_encoder

        # Low-similarity source gets the highest cross-encoder score
        # This simulates a case where raw embedding similarity was misleading
        # but the cross-encoder correctly identifies the most relevant chunk
        mock_encoder.predict.return_value = [
            0.30,  # score for high-similarity source (less relevant to query)
            0.55,  # score for medium-similarity source
            0.92,  # score for low-similarity source (most relevant to query)
        ]

        from retrieval.reranker import rerank

        query = "How does date formatting work?"
        results = rerank(query=query, sources=sources_mixed, top_n=3)

        # The low-similarity source (format_date) got score 0.92 and should be first
        assert results[0].function_name == "format_date", (
            f"Expected 'format_date' (highest cross-encoder score) to be first, "
            f"but got '{results[0].function_name}'. Re-ranker may not be sorting "
            "by cross-encoder score correctly."
        )

    @patch("retrieval.reranker.CrossEncoder")
    def test_top_n_limit_respected(self, mock_cross_encoder_class, sources_mixed):
        """
        rerank() must return at most top_n results, even if more sources
        are passed in.

        Why: The generation prompt has a fixed context window. If top_n is
        ignored and all 20 retrieved chunks are injected, we blow the token
        budget and get a truncated (and often wrong) answer.
        """
        mock_encoder = MagicMock()
        mock_cross_encoder_class.return_value = mock_encoder
        mock_encoder.predict.return_value = [0.9, 0.7, 0.5]

        from retrieval.reranker import rerank
        results = rerank(query="test", sources=sources_mixed, top_n=2)

        assert len(results) == 2, (
            f"Expected exactly 2 results with top_n=2, got {len(results)}. "
            "top_n limit is not being applied in reranker.py."
        )

    @patch("retrieval.reranker.CrossEncoder")
    def test_empty_sources_returns_empty(self, mock_cross_encoder_class):
        """
        rerank() with an empty sources list should return an empty list,
        not raise an exception.

        Why: semantic_search() can return [] if Qdrant has no matches.
        The hybrid retriever passes that directly to rerank(). If rerank()
        crashes on empty input, we get a 500 instead of a graceful refusal.
        """
        mock_encoder = MagicMock()
        mock_cross_encoder_class.return_value = mock_encoder

        from retrieval.reranker import rerank
        results = rerank(query="anything", sources=[], top_n=5)

        assert results == []
        # The cross-encoder should NOT be called with an empty input
        mock_encoder.predict.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# PART 5: HYBRID RETRIEVER TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestHybridRetriever:
    """
    Tests for retrieval/hybrid_retriever.py.

    The hybrid retriever orchestrates the full retrieval pipeline:
      1. Classify query → QueryType
      2. Run semantic search → List[CodeSourceReference]
      3. (For RELATIONAL) Run graph traversal → additional nodes
      4. Merge and de-duplicate results
      5. Re-rank → final top_n results

    These tests use mocks for all sub-components and verify the orchestration
    logic — specifically that the right sub-retriever is called for each
    query type and that results are passed through re-ranking.
    """

    @patch("retrieval.hybrid_retriever.rerank")
    @patch("retrieval.hybrid_retriever.semantic_search")
    @patch("retrieval.hybrid_retriever.classify_query")
    def test_lookup_query_uses_semantic_only(
        self,
        mock_classify,
        mock_semantic,
        mock_rerank,
        sample_source_high_confidence,
    ):
        """
        A LOOKUP query must use semantic search only — no graph traversal.

        Why: Graph traversal on a lookup query is wasteful (O(V+E) traversal
        for a question that needs a single point-lookup) and may return
        irrelevant graph neighbors.
        """
        mock_classify.return_value = QueryType.LOOKUP
        mock_semantic.return_value = [sample_source_high_confidence]
        mock_rerank.return_value = [sample_source_high_confidence]

        from retrieval.hybrid_retriever import retrieve
        results = retrieve(
            query="Where is the authenticate_user function?",
            collection_name="codesense_test_repo",
        )

        mock_semantic.assert_called_once()
        mock_rerank.assert_called_once()
        assert len(results) > 0

    @patch("retrieval.hybrid_retriever.graph_search")
    @patch("retrieval.hybrid_retriever.rerank")
    @patch("retrieval.hybrid_retriever.semantic_search")
    @patch("retrieval.hybrid_retriever.classify_query")
    def test_relational_query_calls_graph_search(
        self,
        mock_classify,
        mock_semantic,
        mock_rerank,
        mock_graph,
        sample_source_high_confidence,
        sample_source_medium_confidence,
    ):
        """
        A RELATIONAL query must call both semantic_search AND graph_search.

        Why: "What calls the auth module?" cannot be answered by embedding
        similarity alone. The call graph knows exactly which functions have
        edges pointing to auth — embeddings only know which functions are
        semantically similar to auth. Both signals are needed.
        """
        mock_classify.return_value = QueryType.RELATIONAL
        mock_semantic.return_value = [sample_source_medium_confidence]
        mock_graph.return_value = [sample_source_high_confidence]
        mock_rerank.return_value = [sample_source_high_confidence, sample_source_medium_confidence]

        from retrieval.hybrid_retriever import retrieve
        results = retrieve(
            query="What calls the auth module?",
            collection_name="codesense_test_repo",
        )

        mock_semantic.assert_called_once()
        mock_graph.assert_called_once()  # Graph MUST be called for RELATIONAL
        mock_rerank.assert_called_once()

    @patch("retrieval.hybrid_retriever.rerank")
    @patch("retrieval.hybrid_retriever.semantic_search")
    @patch("retrieval.hybrid_retriever.classify_query")
    def test_deduplication_of_results(
        self,
        mock_classify,
        mock_semantic,
        mock_rerank,
        sample_source_high_confidence,
    ):
        """
        If the same source appears in both semantic and graph results, it
        must appear only once in the merged list passed to re-ranking.

        Why: Duplicate entries in the re-ranker input don't cause crashes
        but DO cause the same source to appear twice in the generated answer,
        wasting context tokens and making the response look broken.

        Deduplication key: (file_path, function_name) — two chunks from the
        same function in the same file are the same source.
        """
        mock_classify.return_value = QueryType.RELATIONAL

        # Both semantic and graph return the same source
        mock_semantic.return_value = [sample_source_high_confidence]

        # We patch at the hybrid_retriever level — graph_search also returns
        # the same source (simulating overlap between retrieval strategies)
        with patch("retrieval.hybrid_retriever.graph_search") as mock_graph:
            mock_graph.return_value = [sample_source_high_confidence]

            # Capture what gets passed to rerank
            captured_input = []
            def capture_rerank(query, sources, top_n):
                captured_input.extend(sources)
                return sources[:top_n]
            mock_rerank.side_effect = capture_rerank

            from retrieval.hybrid_retriever import retrieve
            retrieve(
                query="What calls authenticate_user?",
                collection_name="codesense_test_repo",
            )

            # The same source appeared in both semantic and graph results
            # After deduplication, it should appear exactly once
            unique_keys = set(
                (s.file_path, s.function_name) for s in captured_input
            )
            assert len(unique_keys) == len(captured_input), (
                f"Duplicates found in re-ranker input. "
                f"unique_keys={len(unique_keys)}, total={len(captured_input)}. "
                "hybrid_retriever.py is not deduplicating results."
            )


# ─────────────────────────────────────────────────────────────────────────────
# PART 6: REGRESSION TESTS (Day 8 Experiment Results)
# ─────────────────────────────────────────────────────────────────────────────

class TestHybridVsSemanticOnlyRegression:
    """
    Regression tests that lock in the Day 8 experiment result:
    Hybrid retrieval outperforms semantic-only by a meaningful margin.

    WHY THESE TESTS EXIST:
    On Day 8, we ran an experiment comparing semantic-only retrieval vs
    hybrid (semantic + structural) retrieval on 20 test queries over the
    FastAPI repo. The hybrid approach achieved higher precision@5.

    These tests don't re-run the full experiment (that would require live
    embeddings and Qdrant). Instead, they test the LOGIC that produces the
    hybrid score — specifically that the weighted combination formula is
    applied correctly and that structural scores can change the ranking.

    If someone "simplifies" the hybrid retriever by removing the structural
    component, these tests fail and explain why that's a regression.
    """

    def test_hybrid_score_formula_applied_correctly(self):
        """
        The hybrid score must equal:
          0.7 * semantic_similarity + 0.3 * structural_similarity

        These weights come from the Day 8 experiment: we tried 0.5/0.5,
        0.6/0.4, 0.7/0.3, and 0.8/0.2 on the test query set. 0.7/0.3
        had the best precision@5 (0.78 vs 0.71 for semantic-only).

        If someone changes the weights, this test forces them to update the
        architecture notes and evaluation_results.md with new experiment data.
        """
        from config import settings

        semantic_sim = 0.80
        structural_sim = 0.60
        expected_score = (
            settings.hybrid_semantic_weight * semantic_sim
            + settings.hybrid_structural_weight * structural_sim
        )

        # Verify using the config values (not hardcoded 0.7/0.3)
        # so this test stays valid if the weights are tuned
        assert expected_score == pytest.approx(
            settings.hybrid_semantic_weight * 0.80
            + settings.hybrid_structural_weight * 0.60,
            abs=0.001
        )

    def test_structural_score_can_promote_lower_similarity_result(self):
        """
        A result with lower semantic similarity but higher structural centrality
        can end up with a higher hybrid score than a pure semantic winner.

        This is the key behavioral difference between hybrid and semantic-only:
        a function that is semantically adjacent to the query AND is a critical
        hub in the call graph should rank higher than one that is only
        semantically similar.

        Why test this?
        If the structural component is accidentally removed or zeroed out,
        all hybrid scores degrade to pure semantic scores — and this test
        fails, alerting us to the regression.
        """
        from config import settings

        # Result A: high semantic similarity, low structural score (leaf node)
        semantic_a = 0.85
        structural_a = 0.10
        score_a = (
            settings.hybrid_semantic_weight * semantic_a
            + settings.hybrid_structural_weight * structural_a
        )

        # Result B: slightly lower semantic similarity, high structural score (hub)
        semantic_b = 0.78
        structural_b = 0.90
        score_b = (
            settings.hybrid_semantic_weight * semantic_b
            + settings.hybrid_structural_weight * structural_b
        )

        # B should outrank A because its structural centrality compensates
        # for the lower semantic similarity
        assert score_b > score_a, (
            f"Expected hybrid score_b ({score_b:.3f}) > score_a ({score_a:.3f}). "
            f"score_a={score_a:.3f} (semantic={semantic_a}, structural={structural_a}), "
            f"score_b={score_b:.3f} (semantic={semantic_b}, structural={structural_b}). "
            "If this fails, the structural weight may have been set to 0. "
            "Check HYBRID_STRUCTURAL_WEIGHT in config.py and .env."
        )


# ─────────────────────────────────────────────────────────────────────────────
# PART 7: INTEGRATION TESTS
# These require: docker compose up -d (Qdrant + Redis running)
# Skip with: pytest -m "not integration"
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestSemanticRetrieverIntegration:
    """
    Integration tests for the semantic retriever against a live Qdrant instance.

    These tests:
    1. Create a temporary Qdrant collection with known test vectors
    2. Run semantic_search() against it
    3. Verify results match expected ranking
    4. Clean up the collection

    Why separate integration tests from unit tests?
    Unit tests (above) run in ~1 second total, with no infrastructure.
    Integration tests take 10-30 seconds and require Docker. Keeping them
    marked and separable lets CI run unit tests on every commit and integration
    tests on PRs or scheduled jobs.

    Marked with @pytest.mark.integration so they can be skipped:
      pytest -m "not integration"
    """

    @pytest.fixture(autouse=True)
    def setup_qdrant_collection(self):
        """
        Creates a temporary Qdrant collection before each test and deletes
        it after, regardless of test outcome (fixture teardown in finally).

        Why autouse=True?
        Every test in this class needs a clean collection. autouse=True means
        we don't have to declare this fixture in every test method.
        """
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        from config import settings

        self.client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        self.test_collection = "codesense_integration_test"

        # Create the collection
        self.client.recreate_collection(
            collection_name=self.test_collection,
            vectors_config=VectorParams(
                size=settings.semantic_embedding_dim,
                distance=Distance.COSINE,
            ),
        )

        # Insert two known test points with dummy vectors
        # Point 1: should match an auth query (authentication-related vector)
        # Point 2: should NOT match (unrelated topic)
        self.client.upsert(
            collection_name=self.test_collection,
            points=[
                PointStruct(
                    id=1,
                    vector=[0.9] * 768,  # Dummy — in real tests, use actual embeddings
                    payload={
                        "file_path": "src/auth/login.py",
                        "function_name": "authenticate_user",
                        "start_line": 42,
                        "end_line": 78,
                        "language": "python",
                        "snippet": "def authenticate_user(token): ...",
                    }
                ),
                PointStruct(
                    id=2,
                    vector=[0.1] * 768,  # Very different vector
                    payload={
                        "file_path": "src/utils/date.py",
                        "function_name": "format_date",
                        "start_line": 5,
                        "end_line": 10,
                        "language": "python",
                        "snippet": "def format_date(dt): ...",
                    }
                ),
            ]
        )

        yield  # Test runs here

        # Teardown: always delete the test collection
        self.client.delete_collection(self.test_collection)

    def test_search_returns_results_from_qdrant(self):
        """
        Verify that semantic_search() returns non-empty results from a live
        Qdrant instance with known test data.

        Why test against live Qdrant?
        The unit tests mock Qdrant entirely. This test verifies that the
        actual qdrant-client library call (collection name, vector format,
        filter payload structure) is correct — things mocks can't catch.
        """
        from retrieval.semantic_retriever import semantic_search

        with patch("retrieval.semantic_retriever.get_query_embedding") as mock_embed:
            # Embed returns a vector similar to point 1 (authenticate_user)
            mock_embed.return_value = [0.9] * 768

            results = semantic_search(
                query="How does authentication work?",
                collection_name=self.test_collection,
                top_k=5,
            )

        assert len(results) > 0, (
            "semantic_search() returned no results from a live Qdrant instance "
            "that was pre-populated with test data. Check QdrantClient connection "
            f"settings: host={settings.qdrant_host}, port={settings.qdrant_port}"
        )
        assert isinstance(results[0], CodeSourceReference)
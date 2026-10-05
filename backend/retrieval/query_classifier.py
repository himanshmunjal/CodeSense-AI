"""
retrieval/query_classifier.py
═════════════════════════════

WHY THIS FILE EXISTS
────────────────────
Not all questions about a codebase are the same kind of question. Consider:

    (A) "Where is the payment processing function?"
    (B) "What functions call the authenticate() method?"
    (C) "Find all places where error handling is done inconsistently."
    (D) "Explain what the entire data pipeline does end-to-end."

If you run all four through the same vector similarity search, you will get
poor results for B, C, and D. Question B needs graph traversal. Question C
needs clustering over many retrieved results. Question D needs multi-chunk
synthesis across an entire subsystem.

This file solves that problem: before any retrieval happens, every incoming
user query is classified into one of four types. The type determines which
retrieval strategy is invoked downstream.

This "router" pattern is what makes CodeSense feel intelligent rather than
like a keyword search box. It is the architectural decision that separates
this system from generic RAG pipelines.

QUERY TYPES
───────────
    LOOKUP        — User wants to find a specific thing.
                    Strategy: pure semantic ANN search in Qdrant.
                    Example: "Where is the JWT token validation logic?"

    RELATIONAL    — User wants to know how code elements connect.
                    Strategy: call graph traversal via networkx.
                    Example: "What calls the auth module?" / "What does X depend on?"

    ANALYTICAL    — User wants to reason across many implementations.
                    Strategy: semantic search + k-means clustering + outlier detection.
                    Example: "Find all inconsistent error handling patterns."

    SUMMARIZATION — User wants a high-level explanation of a subsystem.
                    Strategy: multi-chunk retrieval + sequential synthesis.
                    Example: "Explain what the data pipeline does."

CLASSIFICATION APPROACH
────────────────────────
The classifier calls the LLM with a strict prompt that returns a structured
JSON response. This is intentionally LLM-based rather than rule-based because:

    - Rule-based classifiers (regex, keyword matching) break on paraphrasing.
      "What invokes X?" and "Who calls X?" are the same query type — a keyword
      matcher would need exhaustive synonym lists to handle this.

    - The overhead is ~150–200ms and one small API call. Given that the total
      query latency target is 3 seconds, this is an acceptable cost for the
      accuracy gain.

    - The structured output is enforced via OpenAI's JSON mode + a Pydantic
      schema, so the response is always parseable without try/except soup.

DEPENDENCIES
────────────
    openai          — OpenAI-compatible client (pointed at Groq) for classification
    pydantic        — ClassificationResult schema + response validation
    loguru          — Structured logging
    config          — API key and model name from environment
"""

import json
from enum import Enum
from typing import Optional

from loguru import logger
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field

# Import the central settings singleton so this module never hardcodes
# API keys or model names — those live in .env only.
from config import settings


# ─────────────────────────────────────────────────────────────────────────────
# Query Type Enum
# ─────────────────────────────────────────────────────────────────────────────

class QueryType(str, Enum):
    """
    Enumeration of the four retrieval strategies supported by CodeSense.

    Using a str Enum (rather than plain int constants) means the values are
    human-readable in logs, API responses, and the frontend — no lookup table
    needed to understand what "2" means.

    These values are used as routing keys in hybrid_retriever.py to dispatch
    to the correct retrieval function.
    """

    LOOKUP        = "lookup"
    # → User wants to locate a specific function, class, or variable.
    #   Best served by pure vector similarity search.

    RELATIONAL    = "relational"
    # → User wants to understand relationships between code elements.
    #   Best served by traversing the call graph built in call_graph_builder.py.

    ANALYTICAL    = "analytical"
    # → User wants to reason across many similar implementations.
    #   Best served by broad retrieval followed by clustering + outlier detection.

    SUMMARIZATION = "summarization"
    # → User wants a prose explanation of a subsystem or module.
    #   Best served by retrieving all relevant chunks and synthesizing.


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────────────────────────────────────

class ClassificationResult(BaseModel):
    """
    The structured output returned by the LLM classifier.

    WHY PYDANTIC HERE:
    OpenAI's JSON mode guarantees the response is valid JSON, but does not
    guarantee it matches our expected schema. Wrapping the parsed JSON in a
    Pydantic model gives us:
        - Automatic type coercion (e.g., string "lookup" → QueryType.LOOKUP)
        - Field validation with clear error messages on schema mismatch
        - A typed object the rest of the codebase can import and use safely

    Fields
    ──────
    query_type      : The classified type. Drives retrieval strategy selection.
    confidence      : How confident the classifier is (0.0–1.0). Logged and
                      returned to the frontend as a signal to the user.
    reasoning       : One-sentence explanation of why this type was chosen.
                      Useful for debugging misclassifications and for showing
                      the user why their query was handled a certain way.
    suggested_filters : Optional list of metadata filters the classifier
                      inferred from the query text (e.g., a language hint,
                      a directory prefix). Passed to the retriever to narrow
                      the search space before ANN lookup.
    """

    query_type: QueryType = Field(
        ...,
        description="The classified retrieval strategy type."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Classifier confidence in the chosen query_type (0.0 to 1.0)."
    )
    reasoning: str = Field(
        ...,
        description="One-sentence explanation of the classification decision."
    )
    suggested_filters: Optional[dict] = Field(
        default=None,
        description=(
            "Optional metadata filters inferred from the query. "
            "Example: {'language': 'python', 'file_path_prefix': 'src/auth'}. "
            "Passed to Qdrant as payload filters to narrow the search space."
        )
    )


class ClassificationError(BaseModel):
    """
    Returned when classification fails (API error, parse error, etc.).

    Rather than raising an exception that crashes the request, we return
    this object and fall back to LOOKUP — the safest default strategy since
    it will at least return *something* rather than failing silently.

    The caller (hybrid_retriever.py) checks for this type and logs a warning.
    """

    error_message: str
    fallback_type: QueryType = QueryType.LOOKUP


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────────────

# This prompt is the most important string in this file. It is kept as a
# module-level constant (not inside the function) so it is easy to find,
# version, and test independently of the classification logic.
#
# Design choices in this prompt:
#   - Few-shot examples for each query type prevent the LLM from being
#     confused by edge cases (e.g., "what does X call?" looks like LOOKUP
#     but is actually RELATIONAL).
#   - Explicit instruction to return only JSON prevents preamble text that
#     would break JSON parsing.
#   - The "suggested_filters" field is described with examples so the model
#     learns to extract language/path hints from the query text.

_CLASSIFIER_SYSTEM_PROMPT = """
You are a query classifier for CodeSense, an AI-powered codebase intelligence system.

Your job is to classify a user's natural language question about a codebase into
exactly one of four retrieval strategy types. Return ONLY a valid JSON object — no
preamble, no explanation outside the JSON.

QUERY TYPES AND THEIR DEFINITIONS:

1. "lookup"
   The user wants to find a specific function, class, variable, or file.
   The answer is a single location or a small set of locations in the code.
   Examples:
     - "Where is the JWT token validation logic?"
     - "Find the database connection function."
     - "Which file handles rate limiting?"
     - "Show me the User class definition."

2. "relational"
   The user wants to understand how code elements are connected — what calls what,
   what depends on what, what would break if something changed.
   Examples:
     - "What functions call authenticate()?"
     - "What does the payment module depend on?"
     - "What would break if I changed the UserRepository class?"
     - "Who calls the send_email function?"
     - "What is the call chain from main() to the database?"

3. "analytical"
   The user wants to find patterns, inconsistencies, or quality issues across
   many parts of the codebase. The answer requires comparing multiple implementations.
   Examples:
     - "Find all places where error handling is done inconsistently."
     - "Are there any functions that don't validate input?"
     - "Which modules have the most complex functions?"
     - "Find all duplicate logic across the codebase."
     - "Are there security anti-patterns in the authentication code?"

4. "summarization"
   The user wants a high-level explanation of a module, subsystem, or concept.
   The answer requires synthesizing information from many code chunks.
   Examples:
     - "Explain what the data pipeline does."
     - "Give me an overview of how authentication works in this codebase."
     - "What is the architecture of the payment system?"
     - "How does the caching layer work?"

RESPONSE SCHEMA (return exactly this JSON structure):
{
  "query_type": "<lookup|relational|analytical|summarization>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<one sentence explaining why you chose this type>",
  "suggested_filters": <null or object with optional keys: "language", "file_path_prefix">
}

For suggested_filters:
  - If the query mentions a specific language (e.g., "in the Python files", "TypeScript"), 
    include: {"language": "python"} or {"language": "typescript"}
  - If the query mentions a specific directory or module (e.g., "in the auth module",
    "under src/payments"), include: {"file_path_prefix": "src/auth"}
  - If no filters are inferable, return null.

Return ONLY the JSON object. No markdown, no backticks, no explanation outside the JSON.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# QueryClassifier Class
# ─────────────────────────────────────────────────────────────────────────────

class QueryClassifier:
    """
    Classifies incoming user queries into one of four retrieval strategy types.

    WHY A CLASS AND NOT A FUNCTION:
    The OpenAI client is initialized once at construction time and reused for
    all subsequent classify() calls. If this were a plain function, the client
    would be re-initialized on every call, which wastes time and resources.
    Wrapping it in a class lets us share the client as instance state.

    USAGE EXAMPLE:
        classifier = QueryClassifier()

        result = classifier.classify("What calls the auth module?")
        print(result.query_type)     # QueryType.RELATIONAL
        print(result.confidence)     # 0.97
        print(result.reasoning)      # "The query asks about callers of a function..."
        print(result.suggested_filters)  # None
    """

    def __init__(self) -> None:
        """
        Initialize the classifier with a shared OpenAI client.

        The client reads the API key from settings (which reads from .env).
        We do NOT pass the key as a parameter so that test code can mock
        the settings object without touching real credentials.

        WHY groq_api_key / groq_base_url:
        Per Architecture-notes.md §3, CodeSense uses Groq (llama-3.1-70b-
        versatile) via its OpenAI-compatible endpoint, not real OpenAI.
        `settings.openai_api_key` does not exist anywhere in config.py —
        constructing the client with it was an immediate AttributeError.
        """
        self._client = OpenAI(api_key=settings.groq_api_key, base_url=settings.groq_base_url)

        # Store model name from settings so it can be changed in .env without
        # touching this file. Default is "llama-3.1-70b-versatile" (see config.py).
        self._model = settings.groq_model

        logger.debug(
            f"QueryClassifier initialized | model={self._model}"
        )

    def classify(self, query: str) -> ClassificationResult | ClassificationError:
        """
        Classify a natural language query into a retrieval strategy type.

        This is the primary public method. It is called once per user query,
        before any retrieval happens. The result is passed to hybrid_retriever.py
        which uses query_type to decide which retrieval strategy to invoke.

        HOW IT WORKS:
            1. Sends the user's query + the system prompt to the LLM.
            2. Receives a JSON string (enforced by response_format=json_object).
            3. Parses the JSON into a ClassificationResult Pydantic model.
            4. Returns the validated result, or a ClassificationError on failure.

        WHY response_format=json_object:
            OpenAI's JSON mode guarantees the response is parseable JSON.
            Without it, the model sometimes adds markdown fences or preamble
            text that breaks json.loads(). This is the safest way to get
            structured output from the chat endpoint without tool calling.

        Parameters
        ──────────
        query : str
            The raw user query string from the API request body.
            Should be non-empty. Caller is responsible for basic validation.

        Returns
        ───────
        ClassificationResult
            On success: validated Pydantic model with query_type, confidence,
            reasoning, and optional suggested_filters.

        ClassificationError
            On failure (API error, parse error, schema mismatch): a fallback
            object with fallback_type=LOOKUP and a description of what failed.
            The caller should log this and proceed with the fallback type.
        """

        if not query or not query.strip():
            logger.warning("classify() called with empty query — defaulting to LOOKUP")
            return ClassificationError(
                error_message="Empty query string provided.",
                fallback_type=QueryType.LOOKUP
            )

        logger.info(f"Classifying query | query='{query[:80]}...' " if len(query) > 80
                    else f"Classifying query | query='{query}'")

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": _CLASSIFIER_SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        # We wrap the query in a clear label so the model
                        # understands this is the text to classify, not a
                        # conversational message.
                        "content": f"Classify this query:\n\n{query}"
                    }
                ],
                # Low temperature = more deterministic classification.
                # We want the same query to produce the same type every time.
                # Classification is not a creative task.
                temperature=0.0,

                # Max tokens for the JSON response. The schema itself is
                # small, but reasoning models (e.g. openai/gpt-oss-120b on
                # Groq — see config.py's groq_model default) spend a chunk of
                # this budget on hidden reasoning tokens before the actual
                # JSON content, so 256 was cutting it close enough to
                # occasionally truncate the response before valid JSON came
                # out. 512 leaves headroom without meaningfully raising cost.
                max_tokens=512,
            )

            raw_json = response.choices[0].message.content
            logger.debug(f"Raw classifier response | json={raw_json}")

            # Parse the raw JSON string into a Python dict, then validate
            # against our Pydantic schema. Pydantic will raise ValidationError
            # if the schema doesn't match — caught below.
            parsed = json.loads(raw_json)
            result = ClassificationResult(**parsed)

            logger.info(
                f"Classification complete | "
                f"type={result.query_type} | "
                f"confidence={result.confidence:.2f} | "
                f"reasoning='{result.reasoning}'"
            )

            return result

        except OpenAIError as e:
            # Network errors, rate limits, auth failures, etc.
            # We log the full error but return a graceful fallback so the
            # user gets a response (via LOOKUP) rather than a 500 error.
            logger.error(f"OpenAI API error during classification | error={e}")
            return ClassificationError(
                error_message=f"OpenAI API error: {str(e)}",
                fallback_type=QueryType.LOOKUP
            )

        except json.JSONDecodeError as e:
            # Should be rare with response_format=json_object, but can happen
            # if the API returns an error message instead of JSON.
            logger.error(f"Failed to parse classifier JSON response | error={e}")
            return ClassificationError(
                error_message=f"JSON parse error: {str(e)}",
                fallback_type=QueryType.LOOKUP
            )

        except Exception as e:
            # Catch-all for Pydantic ValidationError, unexpected API changes, etc.
            logger.error(f"Unexpected error in QueryClassifier.classify() | error={e}")
            return ClassificationError(
                error_message=f"Unexpected error: {str(e)}",
                fallback_type=QueryType.LOOKUP
            )

    def classify_batch(
        self, queries: list[str]
    ) -> list[ClassificationResult | ClassificationError]:
        """
        Classify multiple queries sequentially.

        WHY NOT ASYNC/PARALLEL:
        Classification is called once per user request, so there is never
        a natural batch of queries from a single user. This method exists for
        the evaluation pipeline (evaluation/run_eval.py) which needs to
        classify 30–40 eval queries. Sequential is fine there — no need for
        the complexity of async gather.

        Parameters
        ──────────
        queries : list[str]
            List of raw query strings to classify.

        Returns
        ───────
        list[ClassificationResult | ClassificationError]
            One result per input query, in the same order.
            Failed classifications return ClassificationError objects, not
            exceptions, so the batch does not abort on a single failure.
        """
        results = []
        for i, query in enumerate(queries):
            logger.debug(f"Batch classification | {i+1}/{len(queries)}")
            results.append(self.classify(query))
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

# A single shared instance for use across the application.
# hybrid_retriever.py imports this directly:
#   from retrieval.query_classifier import classifier
#
# Using a module-level singleton avoids re-initializing the OpenAI client
# on every request. The client is thread-safe so this is safe under FastAPI's
# async workers.

classifier = QueryClassifier()
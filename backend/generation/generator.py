"""
generation/generator.py
──────────────────────────────────────────────────────────────────────────────
WHY THIS FILE EXISTS
──────────────────────────────────────────────────────────────────────────────
This file is the last mile of the CodeSense pipeline. By the time execution
reaches this file, the retrieval layer has already:
  - Classified the query (query_classifier.py)
  - Fetched the most relevant code chunks (semantic + graph retrieval)
  - Re-ranked them with the cross-encoder (reranker.py)

generator.py takes those ranked chunks and does two things:
  1. Builds a prompt that grounds LLM STRICTLY in the retrieved code.
  2. Calls the LLM, validates the output against response_schema.py, and
     returns a structured CodeSenseResponse.

The key design constraint this file enforces: the LLM is NEVER allowed to
use its pre-trained knowledge about common coding patterns. It may ONLY
reference what is in the retrieved code. This is what makes CodeSense a
code intelligence tool rather than a code-flavored chatbot.

Key responsibilities of this file:
  - Confidence gating: refuse to generate if max similarity is below
    settings.retrieval_confidence_threshold
  - Prompt construction with inline citations
  - Structured output parsing and Pydantic validation
  - Retry logic for transient API failures
  - Latency measurement for every call (p50/p90/p99 monitoring)
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import time
from typing import List, Optional

from loguru import logger
from openai import OpenAI, APITimeoutError, APIConnectionError, RateLimitError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging

from config import settings
from generation.response_schema import (
    CodeSenseResponse,
    CodeSourceReference,
    ConfidenceLevel,
    ImpactAnalysisResponse,
    ImpactNode,
    QueryType,
    SummaryGenerationResponse,
    make_refusal_response,
)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL CONSTANTS
# These live here (not in config.py) because they are generation-specific
# tuning knobs that only this module cares about.
# ─────────────────────────────────────────────────────────────────────────────

# Maximum tokens for the full prompt (context + retrieved code chunks).
# The default model (gpt-oss-120b on Groq) supports ~128k tokens. We leave generous headroom for the response.
MAX_PROMPT_TOKENS = 24_000

# Maximum number of source chunks we inject into a single prompt.
# Even if the re-ranker returns 5, we cap here to avoid overloading context
# with low-value chunks when the top 3 are sufficient.
MAX_SOURCES_IN_PROMPT = 5

# Maximum tokens the LLM is allowed to use for its response.
# 1500 is enough for a thorough answer + 3 follow-up queries.
MAX_RESPONSE_TOKENS = 1_500

# Temperature for generation. Set to 0.0 intentionally.
# We want deterministic, grounded answers — not creative ones.
# Higher temperature produces varied but less reliable citations.
GENERATION_TEMPERATURE = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# LLM CLIENT
# Instantiated once at module load time. This is safe because the OpenAI SDK
# client is thread-safe and reusing it avoids repeated TCP connection setup
# overhead.
#
# WHY THIS IS THE OpenAI SDK CLASS BUT NOT REAL OPENAI:
# Per Architecture-notes.md §3, the LLM provider is Groq (llama-3.1-70b-
# versatile), not OpenAI. Groq exposes an OpenAI-compatible /chat/completions
# endpoint, so we keep using the `openai` package's client class and just
# point `base_url` at Groq instead of api.openai.com. settings.openai_api_key
# does not exist in config.py (there is no OpenAI key anywhere in this
# project) — using it here previously was a straight AttributeError at
# import time.
# ─────────────────────────────────────────────────────────────────────────────

_client = OpenAI(api_key=settings.groq_api_key, base_url=settings.groq_base_url)


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

# The system prompt is a hard constraint, not a soft suggestion.
# It tells the LLM exactly three things:
#   1. What it IS (a code analysis assistant).
#   2. What it MUST do (ground every answer in the provided code).
#   3. What it MUST NOT do (use prior knowledge, guess, or hallucinate paths).
#
# The explicit "DO NOT" instructions are load-bearing. Without them, the LLM
# will confidently hallucinate file paths and function names that don't exist
# in the actual repo — which is worse than no answer at all.
_SYSTEM_PROMPT = """You are CodeSense, a code intelligence assistant. Your job is to answer
developer questions about a specific codebase using ONLY the code excerpts provided to you.

RULES YOU MUST FOLLOW WITHOUT EXCEPTION:
1. Answer based EXCLUSIVELY on the CODE EXCERPTS provided below. Do not use any prior
   knowledge about common patterns, libraries, or how code "usually" works.
2. Every claim in your answer MUST be attributable to a specific code excerpt.
3. Always cite your sources inline using the format: [file_path:function_name:Lstart-Lend]
4. If the provided excerpts do not contain enough information to answer the question,
   say so explicitly. Do not guess, infer, or fill gaps with general knowledge.
5. Never invent file paths, function names, or line numbers that are not in the excerpts.
6. Return your response as a valid JSON object matching this exact schema:
   {
     "answer": "<your answer with inline citations>",
     "followup_queries": ["<query 1>", "<query 2>", "<query 3>"]
   }
   The followup_queries array must contain 3 suggested follow-up questions
   that would help the developer understand this part of the codebase better.
   Every follow-up must be about code that exists in the excerpts. The one exception:
   if the excerpts do not contain what the developer asked about at all, do NOT
   suggest follow-ups that assume it exists — return fewer, or an empty array.
7. Do not include markdown formatting, code fences, or any text outside the JSON object."""


# Per-query-type answer shape, appended to _SYSTEM_PROMPT.
#
# Without these, the model defaults to one or two sentences for every query,
# which is right for "what calls X?" but drops half the flow for "how does X
# handle Y?". Length should track the question, not be uniformly short or long.
_TYPE_INSTRUCTIONS = {
    QueryType.LOOKUP: (
        "ANSWER SHAPE — LOOKUP:\n"
        "Name the exact location that answers the question first. If the question "
        "asks where something happens (is called, pushed, registered, raised), cite the "
        "call site that does it, not only the definition of the method involved. Then "
        "add 1-3 sentences of context: what triggers it and what it sets up. Do not "
        "claim one function calls another unless the excerpt shows that call."
    ),
    QueryType.RELATIONAL: (
        "ANSWER SHAPE — RELATIONAL:\n"
        "List each relationship with direction (calls / is called by / imports) and a "
        "citation for both sides. Keep it tight; one line per relationship plus one "
        "sentence on what the caller does with the result is enough."
    ),
    QueryType.ANALYTICAL: (
        "ANSWER SHAPE — ANALYTICAL:\n"
        "Synthesize across excerpts: group similar implementations, call out any that "
        "deviate, and cite each observation. Use as many short paragraphs as the "
        "comparison needs."
    ),
    QueryType.SUMMARIZATION: (
        "ANSWER SHAPE — EXPLANATION:\n"
        "Walk through the flow step by step in 2-4 short paragraphs, citing each step. "
        "Cover the normal path, any branches or short-circuits (e.g. an early return "
        "that skips a later step), and for error-handling questions, where an error "
        "goes if the first handler does not resolve it — follow it to the outer caller "
        "if that caller is in the excerpts. Do not pad with information that is not in "
        "the excerpts."
    ),
}


def _system_prompt_for(query_type: QueryType) -> str:
    instructions = _TYPE_INSTRUCTIONS.get(query_type)
    return f"{_SYSTEM_PROMPT}\n\n{instructions}" if instructions else _SYSTEM_PROMPT


def _build_context_block(sources: List[CodeSourceReference]) -> str:
    """
    Builds the "CODE EXCERPTS" section of the prompt from retrieved sources.

    Why a separate function?
    The prompt has two distinct parts: the fixed instructions (system prompt)
    and the dynamic context (this function's output). Separating them makes
    it easy to log, test, and swap out the context independently.

    Format chosen:
    We use a numbered, clearly delimited format for each source because the LLM
    is more reliable about citing sources when they have unambiguous labels
    (EXCERPT 1, EXCERPT 2) than when they are presented as a continuous block.
    The explicit metadata header (path, function, lines, language) is repeated
    for each excerpt so the model doesn't have to remember which excerpt had
    which path — it's always right there.

    Args:
        sources — The re-ranked list of CodeSourceReference objects, ordered
                  by relevance (most relevant first).

    Returns:
        A formatted string ready to inject into the user message.
    """
    # Cap at MAX_SOURCES_IN_PROMPT even if the caller passed more.
    # We slice here (not at the call site) to keep this function self-contained.
    sources = sources[:MAX_SOURCES_IN_PROMPT]

    blocks = []
    for i, src in enumerate(sources, start=1):
        block = (
            f"--- EXCERPT {i} ---\n"
            f"File:     {src.file_path}\n"
            f"Function: {src.function_name}\n"
            f"Lines:    {src.start_line}–{src.end_line}\n"
            f"Language: {src.language}\n"
            f"Similarity: {src.similarity:.3f}\n\n"
            f"{src.snippet}\n"
            f"--- END EXCERPT {i} ---"
        )
        blocks.append(block)

    return "\n\n".join(blocks)


def _build_user_message(query: str, sources: List[CodeSourceReference]) -> str:
    """
    Assembles the full user message: context block + the developer's question.

    Why combine context and query into the user message (not system)?
    The system prompt contains static instructions that never change between
    calls. The context (retrieved code) changes with every query. Keeping
    them separate:
      - Allows OpenAI to cache the system prompt (reduces latency + cost).
      - Makes the conversation structure clean and auditable in logs.
      - Lets us modify retrieval strategy without touching the instructions.

    Args:
        query   — The original developer question.
        sources — Retrieved, re-ranked code chunks to inject as context.

    Returns:
        A formatted string for the "user" role of the OpenAI messages array.
    """
    context_block = _build_context_block(sources)

    return (
        f"CODE EXCERPTS FROM THE REPOSITORY:\n\n"
        f"{context_block}\n\n"
        f"──────────────────────────────────────\n"
        f"DEVELOPER QUESTION:\n{query}\n\n"
        f"Answer the question using ONLY the excerpts above. "
        f"Cite sources inline as [file_path:function_name:Lstart-Lend]."
    )


# ─────────────────────────────────────────────────────────────────────────────
# RETRY DECORATOR
# ─────────────────────────────────────────────────────────────────────────────

def _make_retry_decorator():
    """
    Creates a tenacity retry decorator for OpenAI API calls.

    Why retry at all?
    OpenAI's API has occasional transient failures: rate limit 429s during
    traffic spikes, timeout errors on slow queries, and connection resets.
    A single API failure should not fail the entire developer query.

    Why tenacity instead of a manual try/except loop?
    Tenacity gives us exponential backoff (2s, 4s, 8s) with jitter, proper
    logging of retry attempts, and clean exception type filtering — all in
    a single decorator. Manual retry loops are error-prone and harder to test.

    Retry policy:
      - Max 3 attempts (1 original + 2 retries)
      - Exponential backoff: 2s → 4s between attempts
      - Only retry on transient errors (timeout, connection, rate limit)
      - Do NOT retry on 400 (bad request) or 401 (auth) — those are bugs
    """
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type((APITimeoutError, APIConnectionError, RateLimitError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,  # If all retries fail, re-raise the original exception
    )


_retry = _make_retry_decorator()


# ─────────────────────────────────────────────────────────────────────────────
# CORE GENERATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

@_retry
def _call_openai(messages: list[dict]) -> str:
    """
    Makes the actual API call to the LLM and returns the raw response string.

    Why is this a separate function from generate()?
    Isolating the API call makes it independently retryable (the retry
    decorator is applied here, not on the full generate() flow), independently
    mockable in tests (mock this function, not the OpenAI SDK directly), and
    easier to swap out if we add other LLM providers later.

    The response_format={"type": "json_object"} parameter activates OpenAI's
    JSON mode, which guarantees the response is valid JSON. Without this,
    the LLM sometimes wraps its JSON in markdown code fences or adds preamble
    text — both of which break json.loads().

    Args:
        messages — The full messages array in OpenAI chat format.

    Returns:
        The raw content string from the model's first choice.
        Guaranteed to be valid JSON if JSON mode is active.

    Raises:
        APITimeoutError      — Request timed out (will be retried)
        APIConnectionError   — Network error (will be retried)
        RateLimitError       — Rate limited (will be retried with backoff)
        Any other exception  — Not retried, propagates immediately
    """
    response = _client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        max_tokens=MAX_RESPONSE_TOKENS,
        temperature=GENERATION_TEMPERATURE,
        response_format={"type": "json_object"},  # Guarantees valid JSON output
    )
    return response.choices[0].message.content


def _parse_llm_output(
    raw_json: str,
    query: str,
    query_type: QueryType,
    sources: List[CodeSourceReference],
    max_similarity: float,
    latency_ms: float,
) -> CodeSenseResponse:
    """
    Parses the raw JSON string from the LLM into a validated CodeSenseResponse.

    Why is parsing separated from the API call?
    Parsing is pure data transformation — no I/O, no side effects. Separating
    it from _call_openai() means:
      - The retry decorator on _call_openai() doesn't retry on parse failures
        (parse failures are bugs, not transient errors — retrying won't help).
      - Tests can unit-test parsing independently of the API call.
      - If parsing fails, we have the raw JSON in the exception for debugging.

    This function also "fills in" the Pydantic model with data the LLM did NOT
    return (sources, similarity scores, latency) — the LLM only returns
    `answer` and `followup_queries`. Everything else is injected here from
    the retrieval layer's output.

    Args:
        raw_json      — Raw JSON string from the LLM.
        query         — Original developer query (echoed in the response).
        query_type    — Classified query type.
        sources       — Re-ranked sources (from retrieval layer).
        max_similarity — Highest similarity score among retrieved chunks.
        latency_ms    — Measured end-to-end latency.

    Returns:
        A fully validated CodeSenseResponse.

    Raises:
        ValueError — If the JSON is malformed or missing required fields.
        ValidationError — If Pydantic validation fails (schema mismatch).
    """
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"The LLM returned invalid JSON despite JSON mode being active. "
            f"Raw response: {raw_json[:500]}. Error: {e}"
        ) from e

    # Validate required fields are present in the parsed dict
    if "answer" not in parsed:
        raise ValueError(
            f"The LLM response missing required 'answer' field. Got keys: {list(parsed.keys())}"
        )

    # followup_queries is required by schema but we default gracefully
    # if the model omits it (shouldn't happen with JSON mode, but defensive)
    followup_queries = parsed.get("followup_queries", [])
    if not isinstance(followup_queries, list):
        logger.warning("followup_queries was not a list; defaulting to empty.")
        followup_queries = []

    # Cap to 3 follow-up queries (the system prompt asks for at most 3 but
    # the LLM occasionally returns 4 or 5 — clamp defensively)
    followup_queries = followup_queries[:3]

    return CodeSenseResponse(
        query=query,
        query_type=query_type,
        answer=parsed["answer"],
        sources=sources,
        max_similarity=max_similarity,
        overall_confidence=CodeSourceReference.confidence_from_similarity(max_similarity),
        refused=False,
        refusal_reason=None,
        followup_queries=followup_queries,
        latency_ms=latency_ms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

def generate(
    query: str,
    query_type: QueryType,
    sources: List[CodeSourceReference],
) -> CodeSenseResponse:
    """
    Main entry point for the generation layer. Takes a query + retrieved sources
    and returns a validated, grounded CodeSenseResponse.

    This is the function called by api/routes/query.py and api/routes/summarize.py.
    It is the only public function in this module — everything else is internal.

    FLOW:
      1. Confidence gate: if max similarity < threshold, refuse immediately.
         No LLM call is made. This saves latency AND prevents hallucination.
      2. Build prompt: assemble system prompt + context block + user question.
      3. Call the LLM with retry logic.
      4. Parse and validate the response against CodeSenseResponse schema.
      5. Return the validated response with latency measurement.

    Args:
        query      — The original developer question as typed.
        query_type — The QueryType assigned by query_classifier.py.
                     Used for routing decisions and logged in the response.
        sources    — Re-ranked list of CodeSourceReference objects from the
                     retrieval layer (reranker.py output). Must not be empty
                     if we expect to generate an answer.

    Returns:
        CodeSenseResponse — always. On refusal or error, a refused=True
        response is returned rather than raising an exception to the caller.
        Errors are logged internally.

    Why return-on-error instead of raise-on-error?
    The API route layer (query.py) benefits from always receiving a
    CodeSenseResponse — it can serialize it directly without wrapping
    everything in try/except. Internal errors are logged with full context.
    The route layer also adds a secondary error handler for truly unexpected
    failures.
    """
    start_time = time.perf_counter()

    # ── Step 1: Confidence Gate ──────────────────────────────────────────────
    # Calculate the maximum similarity from all retrieved sources.
    # If no sources were retrieved, max_similarity is 0.0 — always a refusal.
    if not sources:
        logger.warning(
            f"generate() called with empty sources for query: '{query[:80]}...'"
        )
        latency_ms = (time.perf_counter() - start_time) * 1000
        return make_refusal_response(
            query=query,
            query_type=query_type,
            max_similarity=0.0,
            latency_ms=latency_ms,
        )

    max_similarity = max(src.similarity for src in sources)

    if max_similarity < settings.retrieval_confidence_threshold:
        # Hard refusal — do not call the LLM.
        # Returning a grounded "I don't know" is always better than a confident
        # wrong answer. This is the most important safety gate in the system.
        logger.info(
            f"Refusing query '{query[:60]}' — max similarity {max_similarity:.3f} "
            f"below threshold {settings.retrieval_confidence_threshold}"
        )
        latency_ms = (time.perf_counter() - start_time) * 1000
        return make_refusal_response(
            query=query,
            query_type=query_type,
            max_similarity=max_similarity,
            latency_ms=latency_ms,
        )

    # ── Step 2: Build Prompt ─────────────────────────────────────────────────
    user_message = _build_user_message(query, sources)

    messages = [
        {"role": "system", "content": _system_prompt_for(query_type)},
        {"role": "user",   "content": user_message},
    ]

    logger.debug(
        f"Calling the LLM for query type={query_type.value}, "
        f"sources={len(sources)}, max_similarity={max_similarity:.3f}"
    )

    # ── Step 3 & 4: Call + Parse ─────────────────────────────────────────────
    try:
        raw_json = _call_openai(messages)
        latency_ms = (time.perf_counter() - start_time) * 1000

        response = _parse_llm_output(
            raw_json=raw_json,
            query=query,
            query_type=query_type,
            sources=sources,
            max_similarity=max_similarity,
            latency_ms=latency_ms,
        )

        logger.info(
            f"Generated response for query='{query[:60]}' | "
            f"confidence={response.overall_confidence.value} | "
            f"latency={latency_ms:.0f}ms | sources={len(response.sources)}"
        )
        return response

    except Exception as e:
        # All retries exhausted or non-retryable error.
        # Log the full error context and return a graceful refusal rather
        # than propagating the exception to the route handler.
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            f"Generation failed after retries for query='{query[:60]}'. "
            f"Error: {type(e).__name__}: {e}. Returning refusal response."
        )
        # Return a refusal that signals an internal error (not a retrieval miss)
        # so that monitoring can distinguish "no relevant code" from "API error".
        return CodeSenseResponse(
            query=query,
            query_type=query_type,
            answer="An error occurred while generating the response. Please try again.",
            sources=[],
            max_similarity=max_similarity,
            overall_confidence=ConfidenceLevel.LOW,
            refused=True,
            refusal_reason=f"Internal generation error: {type(e).__name__}. Check server logs.",
            followup_queries=[],
            latency_ms=latency_ms,
        )


def generate_impact_summary(
    changed_function: str,
    changed_file: str,
    blast_radius: List[ImpactNode],
) -> str:
    """
    Generates a plain-English summary of an impact analysis result.

    Why a separate function from generate()?
    Impact analysis doesn't involve retrieving code chunks — it's a graph
    traversal operation. There are no CodeSourceReference objects to pass in.
    This function takes the pre-computed blast radius (from impact_analysis.py)
    and asks the LLM to write a clear, developer-friendly summary of the risk.

    It does NOT go through the confidence gate (there's no similarity score
    for graph results). Instead, it asks the LLM to summarize structured data
    that we computed deterministically — so hallucination risk is much lower.

    Args:
        changed_function — The function the developer is thinking of changing.
        changed_file     — The file it lives in.
        blast_radius     — Pre-computed list of ImpactNode objects from
                           features/impact_analysis.py.

    Returns:
        A plain English string summarizing the blast radius risk.
        This is displayed in the ImpactGraph.jsx panel alongside the graph.
    """
    if not blast_radius:
        return (
            f"No other functions in the codebase directly or indirectly call "
            f"`{changed_function}`. Changing it should have no ripple effects."
        )

    # Serialize the blast radius into a compact, readable format for the prompt.
    # We don't pass full source snippets here — just the structural metadata.
    blast_summary_lines = []
    for node in blast_radius[:20]:  # Cap at 20 to avoid overflowing context
        blast_summary_lines.append(
            f"  - {node.function_name} ({node.file_path}) "
            f"| depth={node.depth} | risk={node.risk_level.value} "
            f"| centrality={node.centrality:.3f}"
        )
    blast_summary = "\n".join(blast_summary_lines)

    high_count   = sum(1 for n in blast_radius if n.risk_level.value == "high")
    medium_count = sum(1 for n in blast_radius if n.risk_level.value == "medium")
    low_count    = sum(1 for n in blast_radius if n.risk_level.value == "low")

    prompt = (
        f"A developer is considering changing the function `{changed_function}` "
        f"in file `{changed_file}`.\n\n"
        f"The call graph analysis found {len(blast_radius)} functions that would "
        f"be affected: {high_count} HIGH risk, {medium_count} MEDIUM risk, "
        f"{low_count} LOW risk.\n\n"
        f"Affected functions (top 20):\n{blast_summary}\n\n"
        f"Write a concise 3-4 sentence developer-focused summary of the change "
        f"risk. Mention the highest-risk functions by name. Be direct and practical."
    )

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a code review assistant. Write concise, accurate "
                    "summaries of code change risk based only on the data provided. "
                    "Do not add general advice. Focus on the specific functions listed."
                )
            },
            {"role": "user", "content": prompt},
        ]
        # Note: JSON mode is NOT used here — we want a plain string, not JSON.
        response = _client.chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            max_tokens=300,
            temperature=0.1,  # Slight temperature for more natural prose
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Failed to generate impact summary: {e}")
        return (
            f"Change to `{changed_function}` affects {len(blast_radius)} functions "
            f"({high_count} high risk). Review the graph for details."
        )


def generate_summary(
    scope_label: str,
    chunks: List[dict],
    focus_hint: Optional[str] = None,
    settings=None,
) -> SummaryGenerationResponse:
    """
    Produces a structured summary (overview / key_components / data_flow /
    entry_points) for a file, directory, or whole repository.

    Called by api/routes/summarizer.py, which does exhaustive scroll-based
    retrieval (not top-K search — see _retrieve_chunks_in_scope) over the
    requested scope and hands us the raw chunk payload dicts to synthesize.

    Why a separate function from generate():
    generate() answers one question grounded in a handful of re-ranked
    chunks with a single free-text answer + per-source similarity scores.
    Summarization has no question and no similarity scores — it's a
    synthesis over potentially dozens of chunks at once, producing a fixed
    structured shape instead. Sharing generate()'s confidence-gate/
    CodeSourceReference machinery would not fit either need.

    Args:
        scope_label — Human-readable scope, e.g. "auth/jwt.py" or "entire repository".
        chunks      — Raw Qdrant payload dicts (CodeChunk.to_qdrant_payload() shape:
                      entity_name, fully_qualified_name, file_path, source_code,
                      docstring, start_line, end_line, language, in_degree, ...).
        focus_hint  — Optional user-supplied hint narrowing what to emphasize
                      (e.g. "focus on error handling").

    Returns:
        SummaryGenerationResponse — on any failure, a best-effort fallback
        built from chunk metadata alone (mirrors generate()'s pattern of
        never propagating LLM errors to the route layer).
    """
    # Cap how much source goes into the prompt — chunks here are already
    # capped by MAX_CHUNKS_FOR_SUMMARY / cluster sampling upstream, but each
    # chunk's source can still be long, so truncate per-chunk too.
    chunk_lines = []
    for c in chunks:
        name = c.get("fully_qualified_name") or c.get("entity_name", "unknown")
        snippet = (c.get("source_code") or "")[:300]
        chunk_lines.append(
            f"### {c.get('file_path', 'unknown')} :: {name} "
            f"(lines {c.get('start_line', 0)}-{c.get('end_line', 0)}, "
            f"callers={c.get('in_degree', 0)})\n"
            f"{c.get('docstring') or ''}\n{snippet}"
        )
    chunks_block = "\n\n".join(chunk_lines)

    focus_line = f"\nFocus specifically on: {focus_hint}\n" if focus_hint else ""

    prompt = (
        f"Summarize the following code from '{scope_label}'.{focus_line}\n\n"
        f"{chunks_block}\n\n"
        f'Respond with JSON: {{"overview": "2-4 sentence summary", '
        f'"key_components": [{{"name": str, "file_path": str, "start_line": int, "role": str}}], '
        f'"data_flow": "string or null", "entry_points": ["function names likely called from outside this scope"]}}'
    )

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a code documentation assistant. Summarize only the "
                    "code provided — never invent functions, files, or behavior "
                    "not present in the given context. Respond with valid JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        response = _client.chat.completions.create(
            model=settings.groq_model if settings else globals()["settings"].groq_model,
            messages=messages,
            max_tokens=800,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.choices[0].message.content)
        return SummaryGenerationResponse(
            overview=parsed.get("overview", ""),
            key_components=parsed.get("key_components", []),
            data_flow=parsed.get("data_flow"),
            entry_points=parsed.get("entry_points", []),
        )

    except Exception as e:
        logger.error(f"Failed to generate summary for '{scope_label}': {e}")
        # Fallback: a minimal but honest summary built from metadata alone,
        # rather than surfacing an internal error to the frontend.
        return SummaryGenerationResponse(
            overview=(
                f"Summary generation failed for '{scope_label}'. "
                f"{len(chunks)} code chunks were retrieved but could not be synthesized."
            ),
            key_components=[
                {
                    "name": c.get("fully_qualified_name") or c.get("entity_name", "unknown"),
                    "file_path": c.get("file_path", "unknown"),
                    "start_line": c.get("start_line", 0),
                    "role": "Unable to summarize — see server logs.",
                }
                for c in chunks[:5]
            ],
            data_flow=None,
            entry_points=[],
        )
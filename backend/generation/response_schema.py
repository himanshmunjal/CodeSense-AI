"""
generation/response_schema.py
──────────────────────────────────────────────────────────────────────────────
WHY THIS FILE EXISTS
──────────────────────────────────────────────────────────────────────────────
When the LLM generates a response, we get back a raw string. That raw string
is dangerous — it can hallucinate citations, return inconsistent formats, or
silently omit source references that we promised developers would always be
present. This file solves that problem by defining EXACTLY what shape every
response must take, using Pydantic models.

Think of this file as the "contract" between the LLM and the rest of the
system. generator.py calls the OpenAI API and then validates the output
against these schemas. If the LLM returns something that doesn't match, it
fails loudly instead of silently passing garbage downstream.

This file is also what gets serialized over the FastAPI layer to the frontend.
The same schemas are used for:
  - Validating LLM output           (generation layer)
  - Serializing API responses        (routes layer)
  - Type-checking in tests           (test layer)

Having one canonical schema in one place means all three layers stay in sync.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class QueryType(str, Enum):
    """
    Represents the four types of queries the system understands.

    Why an Enum instead of a plain string?
    - Prevents typos: QueryType.LOOKUP is safer than "lookup".
    - The classifier in retrieval/query_classifier.py returns one of these.
    - FastAPI serializes Enums as strings automatically in JSON responses.
    - Makes routing logic in hybrid_retriever.py explicit and readable.
    """
    LOOKUP        = "lookup"         # "Where is the payment function?"
    RELATIONAL    = "relational"     # "What calls the auth module?"
    ANALYTICAL    = "analytical"     # "Find inconsistent error handling"
    SUMMARIZATION = "summarization"  # "Explain what the data pipeline does"


class ConfidenceLevel(str, Enum):
    """
    Human-readable confidence band derived from the raw similarity score.

    Why convert a float score to a band?
    - The frontend ConfidenceBadge.jsx shows HIGH / MEDIUM / LOW, not 0.73.
    - Different bands trigger different UI behaviors (LOW shows a warning).
    - Bands are more robust to minor model changes than raw score thresholds.

    Mapping (applied in CodeSourceReference.confidence_from_similarity):
      >= 0.80  → HIGH
      >= 0.65  → MEDIUM
      <  0.65  → LOW  (still answered if above RETRIEVAL_CONFIDENCE_THRESHOLD,
                     0.5 by default — the UI shows a warning)
    """
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE REFERENCE
# ─────────────────────────────────────────────────────────────────────────────

class CodeSourceReference(BaseModel):
    """
    A single pointer back to the exact location in the codebase that was
    used to generate part of the answer.

    Why this model exists:
    The core promise of CodeSense is that EVERY answer is grounded in real
    code with exact file + line citations. This model enforces that promise.
    Without it, the LLM may say "in the auth module" without telling the
    developer which file, which function, or which lines.

    The frontend SourceViewer.jsx uses `file_path` + `start_line` to open
    and highlight the exact code the user asked about.

    Fields:
      file_path     — Relative path from repo root. e.g. "src/auth/login.py"
      function_name — The specific function or class method. e.g. "authenticate_user"
      start_line    — First line of the retrieved chunk (1-indexed).
      end_line      — Last line of the retrieved chunk (1-indexed).
      language      — Programming language; used by SourceViewer for syntax highlighting.
      similarity    — Raw cosine similarity score from Qdrant (0.0 to 1.0).
      confidence    — Bucketed confidence band derived from similarity.
      snippet       — The actual source code of the retrieved chunk, trimmed to fit context.
    """

    file_path: str = Field(
        ...,
        description="Relative path from repo root to the source file.",
        examples=["src/auth/login.py"]
    )
    function_name: str = Field(
        ...,
        description="Name of the function or method this chunk belongs to.",
        examples=["authenticate_user"]
    )
    start_line: int = Field(
        ...,
        ge=1,
        description="First line of this chunk in the source file (1-indexed)."
    )
    end_line: int = Field(
        ...,
        ge=1,
        description="Last line of this chunk in the source file (1-indexed)."
    )
    language: str = Field(
        ...,
        description="Programming language for syntax highlighting.",
        examples=["python", "typescript", "java"]
    )
    similarity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Raw cosine similarity score from vector search (0.0 to 1.0)."
    )
    confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.MEDIUM,
        description="Human-readable confidence band derived from similarity score."
    )
    snippet: str = Field(
        ...,
        description="The actual source code of the retrieved chunk (trimmed to fit context)."
    )

    @field_validator("end_line")
    @classmethod
    def end_line_must_be_gte_start(cls, end_line: int, info) -> int:
        """
        Sanity check: end_line must always be >= start_line.

        Why this validator?
        Tree-sitter occasionally returns inverted line ranges for single-line
        functions, especially in minified or oddly formatted files. This catches
        that early rather than letting SourceViewer.jsx try to highlight an
        invalid or backwards range — which would silently show nothing.
        """
        start = info.data.get("start_line")
        if start is not None and end_line < start:
            raise ValueError(
                f"end_line ({end_line}) must be >= start_line ({start}). "
                "This usually indicates a tree-sitter parsing error on a "
                "single-line or minified function."
            )
        return end_line

    @classmethod
    def confidence_from_similarity(cls, similarity: float) -> ConfidenceLevel:
        """
        Convert a raw cosine similarity score to a ConfidenceLevel band.

        Why a classmethod and not a property?
        We need to compute this BEFORE the model is instantiated, because
        `confidence` is set at construction time. A classmethod lets the
        caller compute it and pass it in during object creation.

        Thresholds:
          >= 0.80 → HIGH    (strong match — trust the answer fully)
          >= 0.65 → MEDIUM  (reasonable match — answer is likely correct)
          <  0.65 → LOW     (weak match — below RETRIEVAL_CONFIDENCE_THRESHOLD
                              the generator refuses outright)
        """
        if similarity >= 0.80:
            return ConfidenceLevel.HIGH
        elif similarity >= 0.65:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    @property
    def location_string(self) -> str:
        """
        Returns a compact, human-readable citation string.

        Example output: "src/auth/login.py:authenticate_user:L42-L78"

        Why a property?
        Both prompt_builder.py (injecting inline citations into the LLM prompt)
        and the API response formatter need this format. A single property
        guarantees the format stays identical in both places.
        """
        return (
            f"{self.file_path}:{self.function_name}"
            f":L{self.start_line}-L{self.end_line}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# IMPACT ANALYSIS SCHEMA
# Used by: features/impact_analysis.py, api/routes/impact.py
# ─────────────────────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    """
    Risk classification for nodes in the blast radius of a change.

    Why three tiers?
    HIGH:   Direct callers (depth 1). If you change the function signature,
            these break immediately and with certainty.
    MEDIUM: Callers of callers (depth 2-3). Indirectly affected — likely need
            review but may not break depending on the change.
    LOW:    Everything else in the blast radius. Aware but probably safe.

    This mirrors how senior engineers actually think about change risk at FAANG
    scale — the reviewers you're interviewing with think in these tiers.
    """
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"


class ImpactNode(BaseModel):
    """
    Represents a single function in the blast radius of a proposed change.

    Used by the /impact-analysis endpoint. Each node tells the developer:
    "if you change X, this function is affected, at this risk level."

    Fields:
      function_name — Name of the affected function.
      file_path     — Where it lives in the repo.
      depth         — Number of hops from the changed function in the call
                      graph. Depth 1 means it directly calls the changed function.
      risk_level    — Derived from depth + centrality score.
      centrality    — networkx betweenness centrality (0.0 to 1.0). A high
                      centrality function is a hub — changing something it
                      depends on has outsized, cascading effects.
    """

    function_name: str = Field(..., description="Name of the affected function.")
    file_path: str = Field(..., description="File path of the affected function.")
    depth: int = Field(
        ...,
        ge=1,
        description="Number of hops from the changed node in the call graph."
    )
    risk_level: RiskLevel = Field(
        ...,
        description="Risk classification: high (depth 1), medium (2-3), low (4+)."
    )
    centrality: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Betweenness centrality of this node in the full call graph."
    )

    @classmethod
    def risk_from_depth(cls, depth: int) -> RiskLevel:
        """
        Derive a RiskLevel from call graph traversal depth.

        This is the primary heuristic for blast radius risk. The depth-based
        classification is intentionally simple — it does not need ML. Engineers
        understand and trust depth-based risk reasoning intuitively.

        Centrality is stored separately as a secondary signal that the frontend
        can use to sort within a risk tier (e.g. two HIGH-risk functions, show
        the higher-centrality one first).
        """
        if depth == 1:
            return RiskLevel.HIGH
        elif depth <= 3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW


# ─────────────────────────────────────────────────────────────────────────────
# INCONSISTENCY DETECTION SCHEMA
# Used by: features/inconsistency_detector.py, api/routes/query.py (ANALYTICAL)
# ─────────────────────────────────────────────────────────────────────────────

class InconsistencyFlag(BaseModel):
    """
    Represents a single function that deviates from the majority pattern
    found by k-means clustering in inconsistency_detector.py.

    Why a separate model from CodeSourceReference?
    CodeSourceReference answers "where did the answer come from."
    InconsistencyFlag answers "this code looks wrong compared to its peers."
    They serve different purposes — conflating them would muddy the intent
    of both.

    Fields:
      source        — The exact code location of the flagged function.
      deviation     — Natural language description of what differs.
                      Generated by the LLM after seeing the majority pattern
                      and the outlier side by side.
      cluster_label — k-means cluster ID this function was assigned to.
                      Outliers appear in singleton clusters or as distant
                      members of a cluster dominated by other functions.
      distance      — Distance from the cluster centroid in embedding space.
                      Higher distance = more anomalous.
    """

    source: CodeSourceReference = Field(
        ...,
        description="Location of the function that deviates from the majority pattern."
    )
    deviation: str = Field(
        ...,
        description="Human-readable description of what differs from the majority pattern."
    )
    cluster_label: int = Field(
        ...,
        description="k-means cluster ID this function was assigned to."
    )
    distance: float = Field(
        ...,
        ge=0.0,
        description="Distance from cluster centroid. Higher means more anomalous."
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class CodeSenseResponse(BaseModel):
    """
    The primary response schema for all /query and /summarize requests.

    WHY THIS IS THE MOST IMPORTANT MODEL IN THE FILE:
    This is what the API returns to the frontend for every developer query.
    It enforces three guarantees that make CodeSense trustworthy:

    1. GROUNDING:
       Every answer comes with `sources` — a list of exact file+line
       references. There is no non-refused answer without sources.
       The model_validator below enforces this at the type level.

    2. TRANSPARENCY:
       `max_similarity` and `overall_confidence` tell the developer how
       much to trust the answer. A LOW confidence answer is visually
       flagged differently in ConfidenceBadge.jsx.

    3. REFUSAL:
       If `refused` is True, the system declined to answer because retrieval
       similarity was below the threshold. The developer sees a clear
       explanation rather than a hallucination.

    Fields:
      query              — Echo of the original query (for logging + display).
      query_type         — Which query class the classifier assigned.
      answer             — The generated answer, grounded in retrieved code only.
      sources            — List of exact code locations used to generate the answer.
      max_similarity     — The highest similarity score among retrieved chunks.
      overall_confidence — Confidence band for the entire response.
      refused            — True if the system declined to answer.
      refusal_reason     — Why the system refused, if applicable.
      followup_queries   — Up to 3 suggested follow-up questions from the LLM.
      latency_ms         — End-to-end response time in ms (for monitoring).
    """

    query: str = Field(..., description="The original developer query as submitted.")
    query_type: QueryType = Field(
        ...,
        description="The classified query type that determined retrieval strategy."
    )
    answer: str = Field(
        ...,
        description="The generated answer, grounded only in retrieved code context."
    )
    sources: List[CodeSourceReference] = Field(
        default_factory=list,
        description="Ordered list of code locations used to generate this answer."
    )
    max_similarity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Highest cosine similarity score among all retrieved chunks."
    )
    overall_confidence: ConfidenceLevel = Field(
        ...,
        description="Confidence band for the whole response, derived from max_similarity."
    )
    refused: bool = Field(
        default=False,
        description="True if the system refused to answer due to low retrieval confidence."
    )
    refusal_reason: Optional[str] = Field(
        default=None,
        description="Explanation for refusal, present only when refused=True."
    )
    followup_queries: List[str] = Field(
        default_factory=list,
        description="Up to 3 suggested follow-up questions, generated by the LLM."
    )
    latency_ms: Optional[float] = Field(
        default=None,
        description="Total end-to-end response time in milliseconds."
    )

    @model_validator(mode="after")
    def sources_required_unless_refused(self) -> "CodeSenseResponse":
        """
        Enforces the core grounding guarantee at the type level:
        A non-refused answer MUST contain at least one source.

        Why a model_validator instead of a field_validator?
        We need access to BOTH `refused` and `sources` simultaneously, which
        requires the full model to be constructed first. field_validators run
        field-by-field before the model exists. model_validator(mode="after")
        runs once all fields have been validated and set.

        This validator is intentionally strict. An answer without sources
        means generator.py broke the grounding constraint — that's a bug to
        be caught immediately, not degraded behavior to be tolerated silently.
        """
        if not self.refused and len(self.sources) == 0:
            raise ValueError(
                "A non-refused CodeSenseResponse must include at least one source. "
                "If no relevant code was found, set refused=True and use "
                "make_refusal_response() instead of constructing this directly."
            )
        return self

    @property
    def primary_source(self) -> Optional[CodeSourceReference]:
        """
        Returns the highest-similarity source from the sources list.

        Why sort by similarity here instead of trusting insertion order?
        Sources come from the re-ranker, so insertion order reflects re-rank
        score — not raw similarity. The primary source for display purposes
        should be the one with the strongest raw vector match, which the
        SourceViewer highlights first when the answer loads.
        """
        if not self.sources:
            return None
        return max(self.sources, key=lambda s: s.similarity)


class ImpactAnalysisResponse(BaseModel):
    """
    Response schema for the /impact-analysis endpoint.

    Answers: "What would break if I changed this function?"

    Returns the full blast radius — every function that directly or indirectly
    calls the changed function — sorted by risk level (HIGH first).

    Fields:
      changed_function — The function the developer asked about.
      changed_file     — File path of that function.
      blast_radius     — All affected functions, HIGH risk first.
      total_affected   — Count of affected nodes (auto-set by model_validator).
      graph_depth      — Maximum traversal depth reached in the call graph.
      latency_ms       — Response time for p50/p90/p99 monitoring.
    """

    changed_function: str = Field(..., description="The function name being analyzed.")
    changed_file: str = Field(..., description="File path of the function being changed.")
    blast_radius: List[ImpactNode] = Field(
        default_factory=list,
        description="All affected functions, sorted by risk level (HIGH first)."
    )
    total_affected: int = Field(
        default=0,
        description="Total number of functions in the blast radius. Auto-computed."
    )
    graph_depth: int = Field(
        default=0,
        description="Maximum depth reached during reverse call graph traversal."
    )
    latency_ms: Optional[float] = Field(
        default=None,
        description="End-to-end response time in milliseconds."
    )

    @model_validator(mode="after")
    def set_total_affected(self) -> "ImpactAnalysisResponse":
        """
        Auto-computes total_affected from the blast_radius list length.

        Why not just call len(blast_radius) in the route handler?
        We want total_affected as a top-level JSON field so the frontend
        can show the count (e.g. "23 functions affected") without deserializing
        the full blast_radius array. This validator keeps it in sync
        automatically — callers never have to set it manually and risk it
        going stale.
        """
        self.total_affected = len(self.blast_radius)
        return self


class InconsistencyResponse(BaseModel):
    """
    Response schema for analytical/inconsistency detection queries.

    Returned when the system finds that some functions implement a pattern
    differently from the majority. For example: 8 of 10 error-handling
    functions use structured logging, but 2 use print() — those 2 are flagged.

    Fields:
      pattern_query    — What the developer asked about (e.g. "error handling").
      total_retrieved  — How many functions were compared.
      num_clusters     — Number of k-means clusters found in embedding space.
      flagged          — Functions that deviate from the majority pattern.
      majority_pattern — the LLM's description of the dominant implementation style.
      latency_ms       — Response time for monitoring.
    """

    pattern_query: str = Field(
        ...,
        description="The original pattern query (e.g. 'how is error handling done?')."
    )
    total_retrieved: int = Field(..., description="Number of code chunks retrieved and clustered.")
    num_clusters: int = Field(..., description="Number of k-means clusters found.")
    flagged: List[InconsistencyFlag] = Field(
        default_factory=list,
        description="Functions that deviate from the majority implementation pattern."
    )
    majority_pattern: str = Field(
        ...,
        description="LLM description of the dominant implementation pattern."
    )
    latency_ms: Optional[float] = Field(
        default=None,
        description="End-to-end response time in milliseconds."
    )


# ─────────────────────────────────────────────────────────────────────────────
# REFUSAL CONVENIENCE CONSTRUCTOR
# ─────────────────────────────────────────────────────────────────────────────

def make_refusal_response(
    query: str,
    query_type: QueryType,
    max_similarity: float,
    latency_ms: Optional[float] = None,
) -> CodeSenseResponse:
    """
    Constructs a standardized refusal response when retrieval confidence is too low.

    WHY THIS FUNCTION EXISTS AS A STANDALONE FUNCTION (not a classmethod):
    The system has one hard rule: if max cosine similarity is below
    settings.retrieval_confidence_threshold (0.5 by default), the system
    must NOT attempt to generate an answer. It must return a clear, helpful
    explanation of why it's declining.

    This function is the SINGLE place in the codebase that constructs that
    refusal. Without it, generator.py and the route handlers might each build
    their own refusal format — leading to inconsistent messages, harder testing,
    and diverging UX behavior over time.

    It also bypasses the model_validator that normally requires sources,
    because a refusal legitimately has no sources. The `refused=True` flag
    signals to the validator that an empty sources list is acceptable here.

    Args:
        query          — The original developer query.
        query_type     — The classified query type (preserved for logging + analytics).
        max_similarity — The actual similarity score that triggered the refusal.
                         Included in the refusal_reason for transparency.
        latency_ms     — Optional: total response time to include in the response.

    Returns:
        A fully valid CodeSenseResponse with refused=True and no sources.

    Usage:
        from generation.response_schema import make_refusal_response, QueryType
        response = make_refusal_response(
            query="Where is the billing logic?",
            query_type=QueryType.LOOKUP,
            max_similarity=0.41,
            latency_ms=312.5,
        )
    """
    return CodeSenseResponse(
        query=query,
        query_type=query_type,
        answer=(
            "I couldn't find relevant code in this repository for your query. "
            "The indexed codebase does not appear to contain a strong match. "
            "Try rephrasing with more specific function or module names."
        ),
        sources=[],  # Allowed here because refused=True bypasses the validator
        max_similarity=max_similarity,
        overall_confidence=ConfidenceLevel.LOW,
        refused=True,
        refusal_reason=(
            f"Maximum retrieval similarity ({max_similarity:.3f}) was below the "
            f"confidence threshold (0.65). No code chunk in the indexed repository "
            f"was a strong enough match for this query to generate a trustworthy answer."
        ),
        followup_queries=[],
        latency_ms=latency_ms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARIZATION SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

class SummaryGenerationResponse(BaseModel):
    """
    Structured LLM output for POST /api/v1/summarize, produced by
    generation/generator.py's generate_summary() and consumed by
    api/routes/summarizer.py, which maps this onto its own SummarizeResponse
    (adding scope/files_analyzed/latency metadata that the LLM doesn't see).

    Kept separate from CodeSenseResponse because a summary has no single
    "answer" string or per-source similarity score — it's a synthesis over
    many chunks at once, not a grounded citation-per-claim response.
    """
    overview: str = Field(..., description="2-4 sentence high-level explanation of this scope.")
    key_components: list[dict] = Field(
        default_factory=list,
        description="Notable functions/classes, each {name, file_path, start_line, role}.",
    )
    data_flow: Optional[str] = Field(default=None)
    entry_points: list[str] = Field(default_factory=list)
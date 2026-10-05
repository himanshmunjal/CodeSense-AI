"""
generation/prompt_builder.py
─────────────────────────────────────────────────────────────────────────────
WHY THIS FILE EXISTS
─────────────────────────────────────────────────────────────────────────────
The retrieval pipeline produces ranked code chunks. The generator (LLM)
consumes a structured text prompt. This file is the bridge between them.

Prompt engineering is not a cosmetic concern — it directly controls:

  1. GROUNDING: Whether the model answers from the code or from its own
     pre-training knowledge. A loose prompt lets the model "help" with what
     it knows about common Python patterns. We want it to refuse to do that.

  2. CITATION ACCURACY: The model must include file paths and line numbers in
     its answer. If the prompt does not specify this format precisely, the
     model will sometimes omit citations, sometimes hallucinate them.

  3. CONFIDENCE CALIBRATION: The model must say "I don't know" when the
     retrieved code does not contain a clear answer. This requires an explicit
     instruction and example of what "I don't know" looks like.

  4. QUERY-TYPE ADAPTATION: A lookup query ("where is X?") needs a different
     response style than a summarization query ("explain what this pipeline
     does"). This file manages those variations.

Keeping all prompt logic here (not scattered across generator.py, routes,
or utils) means prompt changes are localized, testable, and versioned in one
place.

POSITION IN PIPELINE
─────────────────────────────────────────────────────────────────────────────
reranker.py → [this file] → generation/generator.py
"""

from enum import Enum
from typing import Optional

from loguru import logger
from retrieval.reranker import RankedResult


# ─────────────────────────────────────────────────────────────────────────────
# Query Type Enum
# ─────────────────────────────────────────────────────────────────────────────

class QueryType(str, Enum):
    """
    The four query types CodeSense supports.

    Defined here (not just in query_classifier.py) because the prompt builder
    needs to know the query type to choose the right instructions and response
    format. Using a shared Enum prevents string mismatches between the
    classifier and the builder.

    LOOKUP       — "Where is X defined?" / "Find the function that does Y"
                   Expected response: a direct pointer to a file + function.

    RELATIONAL   — "What calls X?" / "What does X depend on?"
                   Expected response: a structured list of relationships.

    ANALYTICAL   — "Find all inconsistent error handling" / "How is auth done?"
                   Expected response: a synthesis across multiple code locations.

    SUMMARIZATION — "Explain what the data pipeline does"
                   Expected response: a narrative explanation, not a code pointer.
    """
    LOOKUP = "lookup"
    RELATIONAL = "relational"
    ANALYTICAL = "analytical"
    SUMMARIZATION = "summarization"


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# The system prompt is the most important prompt in the system.
# It runs before every single query and establishes the core contract
# with the model: answer only from code, cite everything, refuse if uncertain.
#
# Key design decisions:
#   - "ONLY use the retrieved code context" is explicit because the LLM will
#     otherwise blend in its knowledge of common Python/JS patterns.
#   - The citation format is spelled out character-by-character to prevent
#     variations like "token.py line 42" vs "auth/token.py:42".
#   - The refusal instruction includes the exact output string so the
#     generator.py can detect and handle it programmatically.
#   - "Do not guess" is repeated because it is the #1 failure mode.
BASE_SYSTEM_PROMPT = """You are CodeSense, an AI assistant that answers questions about codebases.

CRITICAL RULES — follow these without exception:

1. ONLY use the retrieved code context provided in the user message to answer.
   Do NOT use your own knowledge about programming patterns, common libraries,
   or how code "typically" works. If the answer is not in the retrieved context,
   say so explicitly.

2. Every specific claim you make must cite its source using this exact format:
   `file_path:function_name:start_line-end_line`
   Example: `src/auth/token.py:validate_token:42-67`

3. When including code in your answer, use the exact code from the retrieved
   context. Do not paraphrase, clean up, or reformat the code.

4. If the retrieved context does not contain enough information to answer the
   question confidently, respond with exactly:
   "INSUFFICIENT_CONTEXT: [brief explanation of what is missing]"
   Do NOT attempt a partial answer or guess. Do not say "it might be" or
   "it's likely that". Do not guess.

5. If multiple retrieved chunks contain contradictory information, point out
   the contradiction explicitly rather than resolving it yourself.

6. Do not reference files, functions, or line numbers that are not in the
   retrieved context, even if you "know" they exist from your training.
"""

# Token budget for the context window.
# The default model has a ~128k token context window. We reserve:
#   - ~500 tokens for the system prompt
#   - ~300 tokens for query-type instructions
#   - ~1000 tokens for the model's output (max_tokens in generator.py)
#   - Remainder (~126,200) for the code context
# In practice we use a much smaller budget here as a safety margin.
# A 5-result reranked output is typically 1,000–3,000 tokens of code.
MAX_CONTEXT_TOKENS = 12_000  # Conservative limit; increase if needed

# Separator used between code chunks in the context block.
# Visually distinct so the model can clearly identify chunk boundaries.
CHUNK_SEPARATOR = "\n" + "─" * 60 + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Main Class
# ─────────────────────────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Constructs the system prompt and user message for the LLM from ranked
    retrieval results.

    There are two outputs:
      1. system_prompt — sent as the "system" role message. Establishes
         rules and persona. Built once per query.
      2. user_message — sent as the "user" role message. Contains the
         query, the retrieved code context, and query-type-specific
         instructions.

    Keeping system and user messages separate (rather than combining them)
    is deliberate: OpenAI's models treat system messages as stronger
    constraints than user messages. Putting the grounding rules in the
    system message makes them harder for the model to "override" through
    reasoning.
    """

    def build_system_prompt(
        self,
        query_type: QueryType,
        repo_name: str,
    ) -> str:
        """
        Build the system prompt for a given query type.

        The base system prompt (grounding rules, citation format, refusal
        instruction) is always included. Query-type-specific instructions
        are appended to tune the response style.

        Parameters
        ----------
        query_type : QueryType
            Determines which response format instructions are appended.

        repo_name : str
            Injected into the prompt so the model knows what codebase it
            is analyzing. Prevents confusion in multi-repo scenarios.

        Returns
        -------
        str
            The complete system prompt string.
        """
        # Start with the base rules (always present)
        prompt_parts = [BASE_SYSTEM_PROMPT]

        # Add repository context
        prompt_parts.append(
            f"\nYou are analyzing the repository: {repo_name}\n"
        )

        # Append query-type-specific response format instructions
        type_instructions = self._get_type_instructions(query_type)
        prompt_parts.append(type_instructions)

        system_prompt = "\n".join(prompt_parts)

        logger.debug(
            f"Built system prompt | query_type={query_type} | "
            f"chars={len(system_prompt)}"
        )
        return system_prompt

    def build_user_message(
        self,
        query: str,
        ranked_results: list[RankedResult],
        query_type: QueryType,
        max_context_tokens: int = MAX_CONTEXT_TOKENS,
    ) -> str:
        """
        Build the user message containing the query and retrieved code context.

        Structure
        ─────────
            RETRIEVED CODE CONTEXT
            ──────────────────────
            [Chunk 1 header and code]
            ─────────────────────── (separator)
            [Chunk 2 header and code]
            ...

            USER QUESTION
            ─────────────
            <query>

            [Query-type-specific instruction reminder]

        The code context comes BEFORE the question. This is intentional —
        research on LLM attention patterns suggests that models pay more
        attention to content at the beginning and end of their context window
        ("lost in the middle" problem). By placing the most relevant code
        (rank 1) first and the question last, we maximize the probability
        that both are attended to strongly.

        Parameters
        ----------
        query : str
            The user's original question.

        ranked_results : list[RankedResult]
            Re-ranked results from the reranker. Typically 3–5 results.

        query_type : QueryType
            Used to add a reminder of the expected response format at the
            end of the user message.

        max_context_tokens : int
            Approximate token budget for the context block. If the ranked
            results exceed this, lower-ranked results are dropped until
            the budget is met.

        Returns
        -------
        str
            The complete user message string.
        """
        if not ranked_results:
            # This should not happen in practice — the caller checks for
            # empty results before calling build_user_message. But we handle
            # it gracefully rather than crashing.
            logger.warning("build_user_message called with no ranked results")
            return f"USER QUESTION\n─────────────\n{query}"

        # ── Build context block ───────────────────────────────────────────
        context_block = self._build_context_block(
            ranked_results, max_context_tokens
        )

        # ── Build response format reminder ────────────────────────────────
        # A brief reminder of the expected format at the end of the user
        # message. This acts as a "closer" that primes the model to produce
        # the right output structure immediately.
        format_reminder = self._get_format_reminder(query_type)

        # ── Assemble the full user message ────────────────────────────────
        user_message = (
            f"RETRIEVED CODE CONTEXT\n"
            f"──────────────────────\n"
            f"{context_block}\n\n"
            f"USER QUESTION\n"
            f"─────────────\n"
            f"{query}\n\n"
            f"{format_reminder}"
        )

        logger.debug(
            f"Built user message | "
            f"context_chunks={len(ranked_results)} | "
            f"total_chars={len(user_message)}"
        )
        return user_message

    # ─────────────────────────────────────────────────────────────────────
    # Private Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _build_context_block(
        self,
        ranked_results: list[RankedResult],
        max_context_tokens: int,
    ) -> str:
        """
        Format ranked results into a structured context block for the prompt.

        Each chunk is formatted as:
            [CHUNK 1] (relevance: 0.923)
            File: src/auth/token.py
            Function: validate_token
            Lines: 42-67
            Language: python

            ```python
            def validate_token(token: str) -> bool:
                ...
            ```

            Docstring: Validates a JWT token and returns True if valid.

        The relevance score is included so the model can factor in retrieval
        confidence when synthesizing an answer.

        Token budgeting: We estimate tokens as len(text) / 4 (rough approximation
        for English + code). If adding the next chunk would exceed the budget,
        we stop adding chunks. This prevents prompt truncation surprises at
        the OpenAI API layer, which silently truncates rather than erroring.

        Parameters
        ----------
        ranked_results : list[RankedResult]
            Results in rerank_score order (rank 1 = most relevant = first).

        max_context_tokens : int
            Token budget. Chunks are included in rank order until budget
            is exhausted.
        """
        chunk_texts = []
        estimated_tokens_used = 0

        for result in ranked_results:
            chunk_text = self._format_single_chunk(result)
            estimated_chunk_tokens = len(chunk_text) // 4  # Rough token estimate

            # Check if adding this chunk would exceed the budget.
            # We always include at least one chunk (even if it exceeds budget)
            # so the model has something to work with.
            if chunk_texts and (estimated_tokens_used + estimated_chunk_tokens > max_context_tokens):
                logger.warning(
                    f"Context budget reached at chunk {result.rank}. "
                    f"Dropping remaining {len(ranked_results) - result.rank + 1} chunks."
                )
                break

            chunk_texts.append(chunk_text)
            estimated_tokens_used += estimated_chunk_tokens

        return CHUNK_SEPARATOR.join(chunk_texts)

    def _format_single_chunk(self, result: RankedResult) -> str:
        """
        Format a single RankedResult into a labeled text block.

        The format is designed to be:
          a) Readable by the LLM (clear headers for each metadata field)
          b) Parseable by the response_schema validator (consistent citation format)
          c) Displayable in the frontend source viewer (file + line range)

        Parameters
        ----------
        result : RankedResult
            A single re-ranked result with code, metadata, and scores.
        """
        lines = []

        # Chunk header — rank and relevance score
        lines.append(
            f"[CHUNK {result.rank}] (relevance: {result.rerank_score:.3f} | "
            f"hybrid: {result.hybrid_score:.3f})"
        )

        # Source citation metadata — these exact fields are what the model
        # must reproduce in its answer. Showing them in the context teaches
        # the model the format through in-context demonstration.
        lines.append(f"File: {result.file_path}")
        lines.append(f"Function: {result.function_name}")
        lines.append(f"Lines: {result.start_line}-{result.end_line}")
        lines.append(f"Language: {result.language}")
        lines.append("")  # Blank line before code block

        # Code block with language tag for syntax highlighting
        lines.append(f"```{result.language}")
        lines.append(result.code.strip())
        lines.append("```")

        # Docstring as a separate section (not inside the code block)
        # because it provides natural language context that may be
        # more useful to the model than the implementation itself
        if result.docstring and result.docstring.strip():
            lines.append("")
            lines.append(f"Docstring: {result.docstring.strip()}")

        return "\n".join(lines)

    def _get_type_instructions(self, query_type: QueryType) -> str:
        """
        Return query-type-specific response format instructions.

        These instructions are appended to the system prompt and control
        the structure of the model's output. Each query type has a different
        expected answer shape:

          LOOKUP → single source citation, short answer
          RELATIONAL → list of relationships with citations
          ANALYTICAL → multi-location synthesis with comparison
          SUMMARIZATION → narrative explanation, fewer citations needed

        These are not separate prompts — they are additive instructions
        on top of the base grounding rules.
        """
        instructions = {
            QueryType.LOOKUP: (
                "\nRESPONSE FORMAT — LOOKUP QUERY\n"
                "Answer with: the exact file, function, and line range where "
                "the answer is found. Keep the answer concise. If multiple "
                "locations are relevant, list them in order of relevance."
            ),
            QueryType.RELATIONAL: (
                "\nRESPONSE FORMAT — RELATIONAL QUERY\n"
                "Answer with a structured list of relationships. For each "
                "relationship, cite both the caller and callee with full "
                "file:function:line format. Indicate direction (calls / "
                "is called by / imports / is imported by)."
            ),
            QueryType.ANALYTICAL: (
                "\nRESPONSE FORMAT — ANALYTICAL QUERY\n"
                "Synthesize patterns across the retrieved code chunks. "
                "Group similar implementations together. Explicitly flag "
                "any chunks that deviate from the majority pattern. "
                "Every observation must cite its source."
            ),
            QueryType.SUMMARIZATION: (
                "\nRESPONSE FORMAT — SUMMARIZATION QUERY\n"
                "Write a clear explanation of what the code does. Structure "
                "it as: (1) purpose, (2) inputs/outputs, (3) key steps. "
                "Cite the most relevant chunks for each point. Avoid "
                "implementation detail that doesn't serve understanding."
            ),
        }
        return instructions.get(query_type, "")

    def _get_format_reminder(self, query_type: QueryType) -> str:
        """
        A short reminder appended at the end of the user message.

        Positioned at the end of the context window (where LLM attention
        is strong) to prime the model to produce the right output format
        immediately. Kept brief — the full instructions are in the system
        prompt.
        """
        reminders = {
            QueryType.LOOKUP: (
                "Reminder: Provide the exact file:function:line citation. "
                "If not found in context, respond with INSUFFICIENT_CONTEXT."
            ),
            QueryType.RELATIONAL: (
                "Reminder: List all relationships with full citations. "
                "If not found in context, respond with INSUFFICIENT_CONTEXT."
            ),
            QueryType.ANALYTICAL: (
                "Reminder: Compare patterns across chunks. Flag outliers. "
                "Cite every observation."
            ),
            QueryType.SUMMARIZATION: (
                "Reminder: Explain clearly and cite key chunks. "
                "Stay grounded in the retrieved code."
            ),
        }
        return reminders.get(query_type, "")
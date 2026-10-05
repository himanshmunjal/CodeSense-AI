"""
complexity_analyzer.py
───────────────────────
PURPOSE
-------
Computes code complexity metrics for every function/method extracted from
the AST.  The primary metric is Cyclomatic Complexity (CC), but the module
also computes cognitive complexity and a simple lines-of-code count.

WHY THIS FILE EXISTS
--------------------
Complexity scores serve two distinct purposes in CodeSense:

1. RETRIEVAL QUALITY
   Complexity is stored as metadata on every Qdrant chunk.  This lets the
   retrieval layer prefer simpler, more canonical implementations when
   answering "how does X work?" queries — complex, edge-case-handling
   functions are often not the best explanation of a concept.

2. INCONSISTENCY DETECTION SIGNAL
   features/inconsistency_detector.py uses complexity as one signal when
   flagging outliers.  A function implementing a pattern with CC=15 when
   all similar functions have CC=3 is a strong inconsistency candidate.

WHAT IS CYCLOMATIC COMPLEXITY?
--------------------------------
Cyclomatic Complexity (McCabe, 1976) counts the number of linearly
independent paths through a function.  It is computed as:

    CC = number_of_decision_points + 1

Decision points are: if, elif, else, for, while, except, with, assert,
boolean operators (and/or), ternary expressions.

Interpretation:
    CC 1–5   → simple, easy to test
    CC 6–10  → moderate complexity, acceptable
    CC 11–20 → high complexity, consider refactoring
    CC 21+   → very high, hard to test, flag for review

USED BY
-------
- indexing/metadata_builder.py         stored as chunk metadata in Qdrant
- features/inconsistency_detector.py   outlier detection signal
- evaluation/run_eval.py               complexity distribution in eval reports
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

class ComplexityRating(str, Enum):
    """
    Human-readable complexity rating derived from the raw CC score.

    Using an Enum (not plain strings) means misspellings are caught at
    import time rather than silently producing wrong filter results in Qdrant.
    """
    SIMPLE = "simple"          # CC 1–5
    MODERATE = "moderate"      # CC 6–10
    HIGH = "high"              # CC 11–20
    VERY_HIGH = "very_high"    # CC 21+


@dataclass
class ComplexityResult:
    """
    Full complexity report for a single function or method.

    Attributes
    ----------
    function_name : str
        Fully-qualified name of the analyzed function.
    file_path : str
        Source file containing the function.
    start_line : int
        First line of the function definition.
    end_line : int
        Last line of the function definition.
    cyclomatic_complexity : int
        McCabe cyclomatic complexity score.
    cognitive_complexity : int
        Cognitive complexity score (penalizes nesting depth more heavily
        than CC — better proxy for human readability).
    lines_of_code : int
        Physical lines of code (blank lines and comments excluded).
    decision_point_count : int
        Raw count of branching constructs (if/for/while/try/etc.).
        Stored separately so downstream code can audit the CC calculation.
    rating : ComplexityRating
        Categorical rating derived from cyclomatic_complexity.
    language : str
        Source language: "python" | "javascript" | "java" | "typescript" | "go".
    """
    function_name: str
    file_path: str
    start_line: int
    end_line: int
    cyclomatic_complexity: int
    cognitive_complexity: int
    lines_of_code: int
    decision_point_count: int
    rating: ComplexityRating
    language: str

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a plain dictionary for storage as Qdrant chunk metadata.

        All values must be JSON-serializable primitives because Qdrant
        metadata is stored as a JSON payload.
        """
        return {
            "function_name": self.function_name,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "cognitive_complexity": self.cognitive_complexity,
            "lines_of_code": self.lines_of_code,
            "decision_point_count": self.decision_point_count,
            "complexity_rating": self.rating.value,
            "language": self.language,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Language-specific Decision Point Definitions
# ─────────────────────────────────────────────────────────────────────────────

# Maps tree-sitter node types to a complexity increment for each language.
# Each entry is: { language: { node_type: increment } }
#
# WHY A TABLE INSTEAD OF IF/ELIF CHAINS?
# Adding a new language only requires a new dictionary entry here —
# no changes to the analysis logic itself.  Open/closed principle.
#
# The increment is 1 for most constructs.  Boolean operators (and/or)
# get 1 because each one adds an independent logical path.

DECISION_POINT_NODE_TYPES: dict[str, dict[str, int]] = {
    "python": {
        "if_statement": 1,
        "elif_clause": 1,
        "else_clause": 0,           # else does not add a new path
        "for_statement": 1,
        "while_statement": 1,
        "try_statement": 0,       # each except/catch adds the path, not the try
        "except_clause": 1,
        "with_statement": 0,        # context managers don't branch
        "assert_statement": 1,
        "boolean_operator": 1,      # and / or
        "conditional_expression": 1,  # ternary: a if cond else b
        "match_statement": 1,
        "case_clause": 1,
    },
    "javascript": {
        "if_statement": 1,
        "else_clause": 0,
        "for_statement": 1,
        "for_in_statement": 1,
        "while_statement": 1,
        "do_statement": 1,
        "switch_statement": 0,           # each case adds the path, not the switch
        "case": 1,
        "try_statement": 0,       # each except/catch adds the path, not the try
        "catch_clause": 1,
        "logical_expression": 1,    # && / ||
        "ternary_expression": 1,
        "optional_chain": 0,        # ?. is not a branch
    },
    "typescript": {
        # TypeScript is a superset of JavaScript — same decision points.
        "if_statement": 1,
        "else_clause": 0,
        "for_statement": 1,
        "for_in_statement": 1,
        "while_statement": 1,
        "do_statement": 1,
        "switch_statement": 0,           # each case adds the path, not the switch
        "case": 1,
        "try_statement": 0,       # each except/catch adds the path, not the try
        "catch_clause": 1,
        "logical_expression": 1,
        "ternary_expression": 1,
        "optional_chain": 0,
    },
    "java": {
        "if_statement": 1,
        "else": 0,
        "for_statement": 1,
        "enhanced_for_statement": 1,
        "while_statement": 1,
        "do_statement": 1,
        "switch_expression": 0,          # each case adds the path, not the switch
        "switch_label": 1,
        "try_statement": 0,       # each except/catch adds the path, not the try
        "catch_clause": 1,
        "binary_expression": 1,     # && / ||
        "ternary_expression": 1,
    },
    "go": {
        "if_statement": 1,
        "for_statement": 1,         # covers classic, condition-only, and range for
        "expression_switch_statement": 0, # each case adds the path, not the switch
        "expression_case": 1,
        "type_switch_statement": 0,      # each case adds the path, not the switch
        "type_case": 1,
        "default_case": 0,          # default does not add a new path
        "select_statement": 0,           # each case adds the path, not the switch
        "communication_case": 1,
        "binary_expression": 1,     # && / || (Go has one generic binary_expression
                                     # node for all operators, same overcounting
                                     # tradeoff already accepted for Java above)
    },
}

# Nesting depth penalty multipliers for cognitive complexity.
# Each level of nesting adds this bonus to a decision point's increment.
NESTING_DEPTH_PENALTY = 1


# ─────────────────────────────────────────────────────────────────────────────
# Core Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class ComplexityAnalyzer:
    """
    Computes cyclomatic and cognitive complexity from a tree-sitter AST node.

    This class is instantiated once per repository session and reused across
    all parsed files to avoid re-loading the node-type tables.

    Parameters
    ----------
    language : str
        Source language for this analyzer instance.
        One of: "python", "javascript", "typescript", "java", "go".
    """

    def __init__(self, language: str) -> None:
        if language not in DECISION_POINT_NODE_TYPES:
            raise ValueError(
                f"Unsupported language '{language}'. "
                f"Supported: {list(DECISION_POINT_NODE_TYPES.keys())}"
            )
        self.language = language
        self._decision_points = DECISION_POINT_NODE_TYPES[language]
        logger.debug("ComplexityAnalyzer initialized for language '%s'", language)

    # ── Public Interface ──────────────────────────────────────────────────────

    def analyze_function(
        self,
        ast_node: Any,
        function_name: str,
        file_path: str,
        source_code: str,
    ) -> ComplexityResult:
        """
        Compute all complexity metrics for a single function AST node.

        This is the main entry point called by entity_extractor.py after
        it has isolated the AST node for each function definition.

        Parameters
        ----------
        ast_node : Any
            A tree-sitter Node object representing the full function body.
            Must include the function signature and all nested statements.
        function_name : str
            Fully-qualified name used in the result and for logging.
        file_path : str
            Source file path, stored in the result for traceability.
        source_code : str
            Full source text of the file.  Used to count physical LOC
            by slicing the relevant line range.

        Returns
        -------
        ComplexityResult
            Complete complexity report ready to be stored as chunk metadata.
        """
        start_line = ast_node.start_point[0] + 1   # tree-sitter is 0-indexed
        end_line = ast_node.end_point[0] + 1

        decision_points, cyclomatic_cc = self._compute_cyclomatic(ast_node)
        cognitive_cc = self._compute_cognitive(ast_node, nesting_depth=0)
        loc = self._count_lines_of_code(source_code, start_line, end_line)
        rating = self._rate(cyclomatic_cc)

        result = ComplexityResult(
            function_name=function_name,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            cyclomatic_complexity=cyclomatic_cc,
            cognitive_complexity=cognitive_cc,
            lines_of_code=loc,
            decision_point_count=decision_points,
            rating=rating,
            language=self.language,
        )

        if cyclomatic_cc >= 11:
            logger.warning(
                "High complexity detected: %s (CC=%d, file=%s)",
                function_name, cyclomatic_cc, file_path,
            )

        return result

    def analyze_file(
        self,
        function_nodes: list[tuple[Any, str, int, int]],
        file_path: str,
        source_code: str,
    ) -> list[ComplexityResult]:
        """
        Analyze all functions in a single file in one pass.

        Called by entity_extractor.py after it has collected all function
        nodes from a parsed file.  Returns one ComplexityResult per function.

        Parameters
        ----------
        function_nodes : list[tuple[Any, str, int, int]]
            Each tuple is (ast_node, qualified_name, start_line, end_line).
        file_path : str
            Source file path for the whole batch.
        source_code : str
            Full source text of the file.

        Returns
        -------
        list[ComplexityResult]
            One result per function, in the same order as `function_nodes`.
        """
        results = []
        for ast_node, func_name, _, _ in function_nodes:
            result = self.analyze_function(ast_node, func_name, file_path, source_code)
            results.append(result)
        logger.info("Analyzed %d functions in %s", len(results), file_path)
        return results

    # ── Cyclomatic Complexity ─────────────────────────────────────────────────

    def _compute_cyclomatic(self, node: Any) -> tuple[int, int]:
        """
        Recursively count decision points and derive cyclomatic complexity.

        Algorithm:
          CC = sum(increment for each decision-point node in subtree) + 1

        The +1 accounts for the single path that exists even in a function
        with no branches.

        Parameters
        ----------
        node : Any
            Root tree-sitter Node (the function body).

        Returns
        -------
        tuple[int, int]
            (total_decision_points, cyclomatic_complexity_score)
        """
        decision_count = self._count_decision_points(node)
        return decision_count, decision_count + 1

    def _count_decision_points(self, node: Any) -> int:
        """
        Walk the AST subtree and sum all decision-point increments.

        Uses an iterative stack-based traversal (not Python recursion) to
        avoid hitting Python's default recursion limit on deeply nested code.

        Parameters
        ----------
        node : Any
            Any tree-sitter Node — the traversal descends into all children.

        Returns
        -------
        int
            Total decision-point count for the subtree.
        """
        total = 0
        stack = [node]

        while stack:
            current = stack.pop()
            node_type = current.type
            increment = self._decision_points.get(node_type, 0)
            total += increment

            # Push children onto the stack for continued traversal.
            # We reverse() so that the first child is processed first
            # (stack is LIFO, so we want to push last-child first).
            stack.extend(reversed(current.children))

        return total

    # ── Cognitive Complexity ──────────────────────────────────────────────────

    def _compute_cognitive(self, node: Any, nesting_depth: int) -> int:
        """
        Compute cognitive complexity using the SonarSource specification.

        Cognitive Complexity differs from Cyclomatic Complexity in two ways:
          1. Each decision point is penalized MORE at deeper nesting levels.
             An `if` inside three nested `for` loops is harder to understand
             than a top-level `if`.
          2. Structural increments (breaks in linear flow) are counted
             separately from nesting increments.

        This recursive implementation tracks nesting_depth as a parameter
        so each recursive call automatically inherits the correct depth.

        Parameters
        ----------
        node : Any
            Current tree-sitter Node.
        nesting_depth : int
            Current nesting depth (0 = function body level).

        Returns
        -------
        int
            Cognitive complexity score for this subtree.
        """
        score = 0
        node_type = node.type
        base_increment = self._decision_points.get(node_type, 0)

        if base_increment > 0:
            # Structural increment: +1 for the construct itself.
            score += base_increment
            # Nesting increment: +N for each nesting level above the first.
            if nesting_depth > 0:
                score += nesting_depth * NESTING_DEPTH_PENALTY

        # Constructs that increase nesting for their children.
        nesting_constructs = {
            "if_statement", "for_statement", "while_statement",
            "do_statement", "try_statement", "switch_statement",
            "with_statement",
        }
        child_depth = nesting_depth + 1 if node_type in nesting_constructs else nesting_depth

        for child in node.children:
            score += self._compute_cognitive(child, child_depth)

        return score

    # ── Lines of Code ─────────────────────────────────────────────────────────

    def _count_lines_of_code(
        self,
        source_code: str,
        start_line: int,
        end_line: int,
    ) -> int:
        """
        Count non-blank, non-comment physical lines within a line range.

        'Lines of code' here means lines that contain actual executable
        statements — blank lines and pure-comment lines are excluded.

        This is a lightweight approximation, not a full tokenizer pass.
        It uses simple string heuristics per language, which is accurate
        enough for the metadata use case (we do not need exact SLOC).

        Parameters
        ----------
        source_code : str
            Full source text of the file (all lines).
        start_line : int
            1-indexed first line of the function (inclusive).
        end_line : int
            1-indexed last line of the function (inclusive).

        Returns
        -------
        int
            Count of non-blank, non-comment lines in the range.
        """
        all_lines = source_code.splitlines()
        # Clamp to valid range — tree-sitter line numbers should always be
        # in bounds, but defensive slicing avoids IndexError on edge cases.
        func_lines = all_lines[start_line - 1: end_line]

        comment_prefixes = self._get_comment_prefixes()
        loc = 0

        for line in func_lines:
            stripped = line.strip()
            if not stripped:
                continue
            if any(stripped.startswith(prefix) for prefix in comment_prefixes):
                continue
            loc += 1

        return loc

    def _get_comment_prefixes(self) -> list[str]:
        """
        Return the single-line comment prefixes for the current language.

        Used by _count_lines_of_code to skip comment-only lines.
        Block comment detection (/* ... */) is not implemented here because
        block comments rarely appear inside function bodies and the added
        complexity is not worth it for metadata purposes.

        Returns
        -------
        list[str]
            List of comment-start strings, e.g. ["#"] for Python.
        """
        prefixes = {
            "python": ["#"],
            "javascript": ["//", "/*", "*"],
            "typescript": ["//", "/*", "*"],
            "java": ["//", "/*", "*"],
            "go": ["//", "/*", "*"],
        }
        return prefixes.get(self.language, ["#", "//"])

    # ── Rating ────────────────────────────────────────────────────────────────

    @staticmethod
    def _rate(cyclomatic_complexity: int) -> ComplexityRating:
        """
        Convert a raw CC score to a categorical ComplexityRating.

        Thresholds follow the standard McCabe complexity interpretation.
        These categories are surfaced in the frontend as color-coded badges
        (green / yellow / orange / red) so developers can quickly spot
        high-complexity functions in the file tree.

        Parameters
        ----------
        cyclomatic_complexity : int
            Raw cyclomatic complexity score.

        Returns
        -------
        ComplexityRating
            Categorical rating.
        """
        if cyclomatic_complexity <= 5:
            return ComplexityRating.SIMPLE
        elif cyclomatic_complexity <= 10:
            return ComplexityRating.MODERATE
        elif cyclomatic_complexity <= 20:
            return ComplexityRating.HIGH
        else:
            return ComplexityRating.VERY_HIGH

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def summarize(results: list[ComplexityResult]) -> dict[str, Any]:
        """
        Produce aggregate complexity statistics for a full file or module.

        This summary is logged after each file is processed and is included
        in the evaluation report to show the complexity distribution of the
        indexed codebase.

        Parameters
        ----------
        results : list[ComplexityResult]
            All complexity results from a single file or module.

        Returns
        -------
        dict[str, Any]
            Summary statistics: mean CC, max CC, rating distribution, etc.
        """
        if not results:
            return {}

        cc_scores = [r.cyclomatic_complexity for r in results]
        ratings = [r.rating.value for r in results]

        return {
            "function_count": len(results),
            "mean_cyclomatic_complexity": round(sum(cc_scores) / len(cc_scores), 2),
            "max_cyclomatic_complexity": max(cc_scores),
            "min_cyclomatic_complexity": min(cc_scores),
            "total_loc": sum(r.lines_of_code for r in results),
            "rating_distribution": {
                rating.value: ratings.count(rating.value)
                for rating in ComplexityRating
            },
            "high_complexity_functions": [
                r.function_name
                for r in results
                if r.rating in (ComplexityRating.HIGH, ComplexityRating.VERY_HIGH)
            ],
        }
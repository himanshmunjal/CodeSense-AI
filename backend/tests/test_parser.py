"""
================================================================================
tests/test_parser.py — Parsing & Entity Extraction Unit Tests
================================================================================

WHY THIS FILE EXISTS:
    parsing/tree_sitter_parser.py, parsing/entity_extractor.py,
    parsing/call_graph_builder.py, and parsing/complexity_analyzer.py form
    the foundation the ENTIRE rest of CodeSense is built on. If a function
    is mis-extracted here (wrong line numbers, missed docstring, dropped
    call edge), every downstream layer — embeddings, retrieval, generation,
    impact analysis — silently inherits that error. There is no way to
    "fix it later" at the retrieval layer if the parsing layer handed it
    bad data. This makes parser correctness the single highest-priority
    thing to test in this codebase.

WHY THESE ARE TRUE UNIT TESTS, NOT MOCKED:
    Unlike tests/test_api.py (which mocks Qdrant/Redis/OpenAI because those
    are external services), these tests run REAL tree-sitter parsing against
    small, hand-written code snippets. There is no meaningful way to "mock"
    an AST parser and still test anything useful — the entire point of using
    tree-sitter (as documented in Architecture-notes.md, Section 4) is
    that it produces a correct, real AST. Mocking it would mean testing
    nothing. These tests ARE slower than the mocked API tests (tree-sitter
    parsing isn't free), but they're still well under a second total — far
    from the territory where that matters.

WHY WE TEST WITH SMALL, SYNTHETIC SNIPPETS INSTEAD OF REAL REPO FILES:
    Real-world files (like FastAPI's actual oauth2.py) are large, change
    over time, and would make tests brittle (a test asserting "exactly 4
    functions extracted" breaks the moment upstream FastAPI adds a 5th).
    Small, hand-written snippets give us precise control over exactly what
    edge case each test is targeting — multiple inheritance, decorators,
    nested functions, etc. — and they never change underneath us.
    The evaluation suite (evaluation/run_eval.py) is where we test
    against real, large repositories; this file tests parsing CORRECTNESS
    in isolation.

WHAT "GOOD" LOOKS LIKE FOR THIS TEST FILE:
    - Every supported language gets at least one parsing smoke test.
    - Every extracted entity type (function, class, import, docstring, call
      edge) has a dedicated test with a known, hand-verified expected output.
    - Edge cases that the architecture notes explicitly flag as known
      limitations (syntactically broken code, deeply nested classes) are
      tested to confirm they degrade GRACEFULLY rather than crashing the
      whole ingestion pipeline for one bad file.

HOW TO RUN:
    pytest tests/test_parser.py -v
================================================================================
"""

from pathlib import Path

import pytest

from ingestion.file_walker import SourceFile
from parsing.call_graph_builder import CallGraphBuilder, FunctionNode
from parsing.complexity_analyzer import ComplexityAnalyzer
from parsing.entity_extractor import FileEntities, extract_entities
from parsing.tree_sitter_parser import (
    _get_parser,
    find_nodes_by_type,
    parse_file,
    parse_source_string,
)


_EXTENSIONS = {"python": ".py", "javascript": ".js", "go": ".go"}


def _extract(source: str, language: str, tmp_path: Path) -> FileEntities:
    """Run the real parse → extract pipeline on a snippet written to disk."""
    path = tmp_path / f"sample{_EXTENSIONS[language]}"
    path.write_text(source)
    source_file = SourceFile(
        absolute_path=path,
        relative_path=path.name,
        language=language,
        size_bytes=len(source.encode()),
        extension=path.suffix,
    )
    parsed = parse_file(source_file)
    assert parsed is not None
    return extract_entities(parsed)


def _first_function_node(source: str):
    tree = parse_source_string(source, "python")
    return find_nodes_by_type(tree.root_node, "function_definition")[0]


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES — SAMPLE CODE SNIPPETS
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def python_sample() -> str:
    """
    WHY THIS SPECIFIC SNIPPET:
        Deliberately packs several extraction targets into one small file:
        a module-level import, a base class, a derived class (to test
        inheritance extraction), a docstring on both a class and a method,
        and a function that calls another function (to test call-edge
        resolution in call_graph_builder.py). Every line earns its place —
        nothing here is decorative.
    """
    return '''
import hashlib
from typing import Optional


class BaseAuthenticator:
    """Base class for all authentication strategies."""

    def authenticate(self, token: str) -> bool:
        """Validates a token. Subclasses must override this."""
        raise NotImplementedError


class TokenAuthenticator(BaseAuthenticator):
    """Authenticates users via a hashed bearer token."""

    def authenticate(self, token: str) -> bool:
        """Hashes the token and checks it against the stored value."""
        hashed = hash_token(token)
        return verify_hash(hashed)


def hash_token(token: str) -> str:
    """Returns a SHA-256 hash of the given token."""
    return hashlib.sha256(token.encode()).hexdigest()


def verify_hash(hashed: str) -> bool:
    """Checks a hashed token against the database. Stubbed for the test."""
    return hashed is not None
'''


@pytest.fixture
def javascript_sample() -> str:
    """
    WHY THIS SPECIFIC SNIPPET:
        JavaScript's function syntax has more variety than Python's (function
        declarations, arrow functions, class methods) — this snippet covers
        two of those forms specifically to confirm tree-sitter-javascript's
        grammar differences from tree-sitter-python are handled correctly by
        our extractor, not just incidentally working because Python and JS
        happen to look similar in this one case.
    """
    return '''
function fetchUser(userId) {
    return database.find(userId);
}

const formatUserName = (user) => {
    return user.firstName + " " + user.lastName;
};

class UserService {
    getActiveUsers() {
        return fetchUser(this.currentUserId);
    }
}
'''


@pytest.fixture
def syntactically_broken_python() -> str:
    """
    WHY THIS SNIPPET EXISTS:
        docs/architecture_notes.md lists tree-sitter's error recovery as a
        deliberate advantage over alternatives. This snippet has an
        intentionally unclosed function definition (missing body) to verify
        the parser doesn't crash on broken code — it should still extract
        whatever entities ARE well-formed (here, hash_token) rather than
        failing the entire file.
    """
    return '''
def broken_function(

def hash_token(token):
    """A valid function after a broken one."""
    return token.upper()
'''


@pytest.fixture
def high_complexity_function() -> str:
    """
    WHY THIS SNIPPET EXISTS:
        Cyclomatic complexity counts decision points (if/elif/for/while/and/or/
        except). This function has a deliberately countable number of branches
        so the expected complexity score can be hand-verified rather than
        guessed at. Base complexity is 1, plus one for each of: if, elif,
        for, the `and` in the if condition, and the except clause — 5 decision
        points total, giving an expected cyclomatic complexity of 6.
    """
    return '''
def process_payment(amount, currency, user):
    if amount > 0 and currency == "USD":
        result = charge_card(user, amount)
    elif currency == "EUR":
        result = charge_card(user, amount * 1.1)
    else:
        result = None

    for attempt in range(3):
        if result:
            break

    try:
        log_transaction(result)
    except ConnectionError:
        pass

    return result
'''


@pytest.fixture
def low_complexity_function() -> str:
    """
    WHY THIS SNIPPET EXISTS:
        A straight-line function with zero branches should score the
        minimum possible cyclomatic complexity of 1 — this is the baseline
        every other complexity test is measured against.
    """
    return '''
def add(a, b):
    return a + b
'''


# ─────────────────────────────────────────────────────────────────────────────
# TREE-SITTER PARSER
# ─────────────────────────────────────────────────────────────────────────────

class TestTreeSitterParser:
    """Smoke tests: every supported grammar loads and parses real code."""

    def test_parses_python_without_error(self, python_sample):
        tree = parse_source_string(python_sample, "python")
        assert tree is not None
        assert not tree.root_node.has_error

    def test_parses_javascript_without_error(self, javascript_sample):
        tree = parse_source_string(javascript_sample, "javascript")
        assert tree is not None
        assert not tree.root_node.has_error

    def test_parses_go_without_error(self):
        tree = parse_source_string("package main\n\nfunc main() {}\n", "go")
        assert not tree.root_node.has_error

    def test_error_recovery_on_broken_syntax(self, syntactically_broken_python):
        """
        WHY THIS MATTERS:
            One malformed file in a 2,000-file repo must not abort ingestion.
            tree-sitter's error recovery should still hand back a tree, just
            with has_error set so callers can log it.
        """
        tree = parse_source_string(syntactically_broken_python, "python")
        assert tree is not None
        assert tree.root_node.has_error

    def test_raises_clear_error_for_unsupported_language(self):
        with pytest.raises(ValueError, match="[Uu]nsupported language"):
            _get_parser("cobol")


# ─────────────────────────────────────────────────────────────────────────────
# ENTITY EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

class TestEntityExtractor:
    """Functions, classes, docstrings, imports and line numbers from real ASTs."""

    def test_extracts_all_function_names(self, python_sample, tmp_path):
        entities = _extract(python_sample, "python", tmp_path)
        assert sorted(f.name for f in entities.functions) == [
            "authenticate",     # base method
            "authenticate",     # override — see next test
            "hash_token",
            "verify_hash",
        ]

    def test_distinguishes_methods_with_same_name_in_different_classes(
        self, python_sample, tmp_path
    ):
        """
        WHY THIS MATTERS:
            Two `authenticate` methods must stay two entities tagged with
            their own class — collapsing them would drop one from the index
            and merge their call-graph edges.
        """
        entities = _extract(python_sample, "python", tmp_path)
        authenticate_methods = [f for f in entities.functions if f.name == "authenticate"]
        assert {f.class_name for f in authenticate_methods} == {
            "BaseAuthenticator", "TokenAuthenticator",
        }
        assert all(f.is_method for f in authenticate_methods)

    def test_extracts_class_inheritance(self, python_sample, tmp_path):
        entities = _extract(python_sample, "python", tmp_path)
        classes = {c.name: c for c in entities.classes}
        assert classes["TokenAuthenticator"].base_classes == ["BaseAuthenticator"]
        assert classes["BaseAuthenticator"].base_classes == []

    def test_extracts_docstrings_for_functions_and_classes(self, python_sample, tmp_path):
        entities = _extract(python_sample, "python", tmp_path)
        hash_token_fn = next(f for f in entities.functions if f.name == "hash_token")
        assert hash_token_fn.docstring == "Returns a SHA-256 hash of the given token."
        base_auth = next(c for c in entities.classes if c.name == "BaseAuthenticator")
        assert base_auth.docstring == "Base class for all authentication strategies."

    def test_extracts_imports(self, python_sample, tmp_path):
        entities = _extract(python_sample, "python", tmp_path)
        assert {imp.module_path for imp in entities.imports} >= {"hashlib", "typing"}

    def test_records_correct_line_numbers(self, python_sample, tmp_path):
        """
        Line numbers are 1-based, and the fixture starts with a blank line,
        so `def hash_token` sits on line 23. Off-by-one errors here show up
        as citations pointing at the wrong code in every answer.
        """
        entities = _extract(python_sample, "python", tmp_path)
        hash_token_fn = next(f for f in entities.functions if f.name == "hash_token")
        assert (hash_token_fn.start_line, hash_token_fn.end_line) == (23, 25)

    def test_javascript_extracts_function_declarations_and_arrow_functions(
        self, javascript_sample, tmp_path
    ):
        entities = _extract(javascript_sample, "javascript", tmp_path)
        functions = {f.name: f for f in entities.functions}
        assert {"fetchUser", "formatUserName", "getActiveUsers"} <= set(functions)
        assert functions["getActiveUsers"].class_name == "UserService"

    def test_extraction_does_not_crash_on_broken_file(
        self, syntactically_broken_python, tmp_path
    ):
        """A broken file degrades to partial results instead of raising."""
        entities = _extract(syntactically_broken_python, "python", tmp_path)
        assert isinstance(entities, FileEntities)


# ─────────────────────────────────────────────────────────────────────────────
# CALL GRAPH
# ─────────────────────────────────────────────────────────────────────────────

class TestCallGraph:
    """
    Call edges are resolved by tasks/celery_worker.py from extracted function
    bodies, using "file::Class.method" node ids that match the indexed chunks'
    fully_qualified_name.
    """

    @staticmethod
    def _graph(entities: FileEntities):
        from tasks.celery_worker import _resolve_call_edges

        builder = CallGraphBuilder(repo_name="test/repo")
        all_functions = {}
        for fn in entities.functions:
            qualified = f"{fn.class_name}.{fn.name}" if fn.class_name else fn.name
            node_id = f"{fn.relative_file_path}::{qualified}"
            all_functions[node_id] = (fn, node_id)
            builder.add_function(FunctionNode(
                qualified_name=node_id,
                file_path=fn.relative_file_path,
                start_line=fn.start_line,
                end_line=fn.end_line,
                language=fn.language,
            ))
        builder.add_calls_bulk(_resolve_call_edges(all_functions))
        return builder.graph

    def test_builds_correct_call_edges(self, python_sample, tmp_path):
        graph = self._graph(_extract(python_sample, "python", tmp_path))
        caller = "sample.py::TokenAuthenticator.authenticate"
        assert graph.has_edge(caller, "sample.py::hash_token")
        assert graph.has_edge(caller, "sample.py::verify_hash")

    def test_same_named_methods_are_separate_nodes(self, python_sample, tmp_path):
        graph = self._graph(_extract(python_sample, "python", tmp_path))
        assert "sample.py::BaseAuthenticator.authenticate" in graph.nodes
        assert "sample.py::TokenAuthenticator.authenticate" in graph.nodes

    def test_graph_has_no_edge_for_unrelated_functions(self, python_sample, tmp_path):
        graph = self._graph(_extract(python_sample, "python", tmp_path))
        assert not graph.has_edge("sample.py::hash_token", "sample.py::verify_hash")

    def test_nested_definitions_are_not_calls(self, tmp_path):
        """
        A nested `def __call__(` is a definition, not a call. Real case
        (pallets/flask): test helpers defining WSGI middleware with their own
        `__call__` were linked as callers of Flask.__call__.
        """
        source = (
            "class Flask:\n"
            "    def __call__(self, environ):\n"
            "        return environ\n"
            "\n"
            "def make_middleware(app):\n"
            "    class Middleware:\n"
            "        def __call__(self, environ):\n"
            "            return app(environ)\n"
            "    return Middleware()\n"
        )
        graph = self._graph(_extract(source, "python", tmp_path))
        assert not graph.has_edge("sample.py::make_middleware", "sample.py::Flask.__call__")

    def test_reverse_traversal_finds_all_callers(self, python_sample, tmp_path):
        graph = self._graph(_extract(python_sample, "python", tmp_path))
        callers = set(graph.predecessors("sample.py::hash_token"))
        assert callers == {"sample.py::TokenAuthenticator.authenticate"}


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY
# ─────────────────────────────────────────────────────────────────────────────

class TestComplexityAnalyzer:
    """McCabe cyclomatic complexity: decision points + 1."""

    def test_straight_line_function_has_complexity_one(self, low_complexity_function):
        result = ComplexityAnalyzer("python").analyze_function(
            _first_function_node(low_complexity_function), "add", "x.py",
            low_complexity_function,
        )
        assert result.cyclomatic_complexity == 1

    def test_branching_function_has_expected_complexity(self, high_complexity_function):
        """
        Hand count for process_payment: if, `and`, elif, for, inner if,
        except → 6 decision points → CC 7. `else` and the bare `try` add no
        independent path.
        """
        result = ComplexityAnalyzer("python").analyze_function(
            _first_function_node(high_complexity_function), "process_payment", "x.py",
            high_complexity_function,
        )
        assert result.decision_point_count == 6
        assert result.cyclomatic_complexity == 7

    def test_complexity_is_always_at_least_one(self):
        source = "def noop():\n    pass\n"
        result = ComplexityAnalyzer("python").analyze_function(
            _first_function_node(source), "noop", "x.py", source,
        )
        assert result.cyclomatic_complexity >= 1

    def test_unsupported_language_raises(self):
        with pytest.raises(ValueError):
            ComplexityAnalyzer("cobol")

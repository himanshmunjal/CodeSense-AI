"""
tests/test_snippet_budget.py — prompt source ordering and snippet budgeting.

Regression coverage for "Where is the request context pushed?" against
pallets/flask: the reranker put the 9.9K-char AppContext class chunk above
AppContext.push, the class head-cut consumed the whole snippet budget before
`def push`, and push() itself was dropped — so the LLM answered that no push
operation was visible.
"""

from api.routes.query import (
    _TOTAL_SNIPPET_CHAR_BUDGET,
    _apply_snippet_budget,
    _order_specific_before_containers,
)
from generation.response_schema import CodeSourceReference
from retrieval.hybrid_retriever import HybridResult
from retrieval.reranker import RankedResult


def _ranked(name: str, start: int, end: int, file_path: str = "src/ctx.py") -> RankedResult:
    return RankedResult(
        hybrid_result=HybridResult(
            chunk_id=name, file_path=file_path, function_name=name,
            start_line=start, end_line=end, language="python", code="",
            docstring="", semantic_score=0.8, structural_score=0.0,
            hybrid_score=0.8, graph_distance=None, metadata={},
        ),
        rerank_score=1.0,
        rank=1,
    )


def _source(snippet: str, start: int = 260, name: str = "AppContext") -> CodeSourceReference:
    end = start + snippet.count("\n")
    return CodeSourceReference(
        file_path="src/ctx.py", function_name=name, start_line=start,
        end_line=end, language="python", similarity=0.8,
        confidence=CodeSourceReference.confidence_from_similarity(0.8),
        snippet=snippet,
    )


def _big_class() -> str:
    """A class whose last method starts well past the snippet budget."""
    filler = "        x = 1\n" * (_TOTAL_SNIPPET_CHAR_BUDGET // 14 + 50)
    return (
        "class AppContext:\n"
        "    def __init__(self, app):\n"
        f"{filler}"
        "    def push(self) -> None:\n"
        "        _cv_app.set(self)\n"
    )


class TestOrderSpecificBeforeContainers:
    def test_method_moves_ahead_of_its_class(self):
        cls, push, other = _ranked("AppContext", 260, 525), _ranked("push", 416, 444), _ranked("x", 1, 5, "app.py")
        ordered = _order_specific_before_containers([cls, push, other])
        assert [r.function_name for r in ordered] == ["push", "AppContext", "x"]

    def test_nested_containers_follow_all_descendants(self):
        module, cls, push = _ranked("mod", 1, 600), _ranked("AppContext", 260, 525), _ranked("push", 416, 444)
        ordered = _order_specific_before_containers([module, cls, push])
        assert [r.function_name for r in ordered] == ["push", "AppContext", "mod"]

    def test_unrelated_order_and_identical_ranges_preserved(self):
        a, b = _ranked("a", 10, 20), _ranked("b", 10, 20)
        c = _ranked("c", 30, 40, "other.py")
        assert _order_specific_before_containers([a, b, c]) == [a, b, c]

    def test_same_range_in_different_files_is_not_containment(self):
        cls, push = _ranked("AppContext", 260, 525, "a.py"), _ranked("push", 416, 444, "b.py")
        assert _order_specific_before_containers([cls, push]) == [cls, push]


class TestSnippetBudgetOutline:
    def test_over_budget_class_is_outlined_with_absolute_lines(self):
        src = _source(_big_class(), start=260)
        push_line = 260 + _big_class().split("\n").index("    def push(self) -> None:")

        [out] = _apply_snippet_budget([src])

        assert len(out.snippet) <= _TOTAL_SNIPPET_CHAR_BUDGET
        assert out.snippet.startswith("class AppContext:")
        assert f"L{push_line}: def push(self) -> None:" in out.snippet
        assert "L261: def __init__(self, app):" in out.snippet
        assert "x = 1" not in out.snippet

    def test_fitting_snippets_are_untouched(self):
        src = _source("def f():\n    return 1\n", name="f")
        assert _apply_snippet_budget([src])[0].snippet == src.snippet

    def test_no_nested_definitions_falls_back_to_head_cut(self):
        body = "def f():\n" + "    x = 1\n" * (_TOTAL_SNIPPET_CHAR_BUDGET // 10 + 50)
        [out] = _apply_snippet_budget([_source(body, name="f")])
        assert out.snippet.endswith("... [truncated]")

    def test_unsupported_language_falls_back_to_head_cut(self):
        src = _source(_big_class()).model_copy(update={"language": "unknown"})
        [out] = _apply_snippet_budget([src])
        assert out.snippet.endswith("... [truncated]")

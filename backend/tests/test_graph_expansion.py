"""
tests/test_graph_expansion.py — call-graph expansion for LOOKUP/SUMMARIZATION.

Regression coverage for two pallets/flask answers that were wrong because the
answering code was one call-graph hop from every semantic hit:
  - "Where is the request context pushed?" matched AppContext.push, but the
    push happens in its caller Flask.wsgi_app, which was never retrieved.
  - "How does Flask handle errors in full_dispatch_request?" never saw
    handle_user_exception / finalize_request (callees) or wsgi_app (caller).
"""

import pytest

from api.routes import query as q
from retrieval.graph_retriever import GraphNode, GraphSearchResult
from retrieval.hybrid_retriever import HybridResult
from retrieval.query_classifier import QueryType


def _hit(file_path: str, name: str, score: float = 0.8) -> HybridResult:
    return HybridResult(
        chunk_id=name, file_path=file_path, function_name=name,
        start_line=1, end_line=2, language="python", code="", docstring="",
        semantic_score=score, structural_score=0.0, hybrid_score=score,
        graph_distance=None, metadata={},
    )


def _node(file_path: str, name: str, rel: str) -> GraphNode:
    return GraphNode(
        node_id=f"{file_path}::{name}", file_path=file_path, function_name=name,
        language="python", start_line=1, end_line=2, complexity=1,
        relationship_type=rel, distance=1,
    )


class _FakeGraph:
    """Graph keyed by exact node id; unknown ids fuzzy-resolve elsewhere."""

    def __init__(self, callers: dict, callees: dict):
        self.callers, self.callees = callers, callees

    def load_graph(self, collection_name):
        return True

    def _result(self, table, query, kind):
        if query not in table:
            # Mimics _resolve_node's fuzzy fallback landing on another node.
            return GraphSearchResult(
                nodes=[_node("src/x.py", "Unrelated.method", "direct_caller")],
                query_node_id="src/x.py::Unrelated.method", traversal_type=kind,
            )
        return GraphSearchResult(nodes=table[query], query_node_id=query, traversal_type=kind)

    def get_callers(self, query, collection_name, max_depth=1):
        return self._result(self.callers, query, "callers")

    def get_callees(self, query, collection_name, max_depth=1):
        return self._result(self.callees, query, "callees")


@pytest.fixture
def flask_graph(monkeypatch):
    graph = _FakeGraph(
        callers={
            "src/flask/ctx.py::AppContext.push": [
                _node("tests/test_basic.py", "test_context_test", "direct_caller"),
                _node("tests/test_basic.py", "test_teardown_on_pop", "direct_caller"),
                _node("tests/test_basic.py", "test_manual_context_binding", "direct_caller"),
                _node("src/flask/app.py", "Flask.wsgi_app", "direct_caller"),
            ],
            "src/flask/app.py::Flask.full_dispatch_request": [
                _node("src/flask/app.py", "Flask.wsgi_app", "direct_caller"),
            ],
        },
        callees={
            "src/flask/app.py::Flask.full_dispatch_request": [
                _node("src/flask/views.py", "View.dispatch_request", "direct_callee"),
                _node("src/flask/views.py", "MethodView.dispatch_request", "direct_callee"),
                _node("src/flask/app.py", "Flask.preprocess_request", "direct_callee"),
                _node("src/flask/app.py", "Flask.finalize_request", "direct_callee"),
                _node("src/flask/app.py", "Flask.handle_user_exception", "direct_callee"),
            ],
        },
    )
    monkeypatch.setattr(q, "graph_retriever", graph)
    monkeypatch.setattr(q, "_fetch_snippet_for_node", lambda *a: ("def f(): ...", ""))


def _names(results):
    return [r.function_name for r in results]


def test_lookup_pulls_in_caller_ahead_of_tests(flask_graph):
    added = q._expand_with_graph_neighbours(
        [_hit("src/flask/ctx.py", "AppContext.push")], QueryType.LOOKUP, "c", None,
    )
    assert _names(added)[0] == "Flask.wsgi_app"


def test_lookup_does_not_expand_callees(flask_graph):
    added = q._expand_with_graph_neighbours(
        [_hit("src/flask/app.py", "Flask.full_dispatch_request")], QueryType.LOOKUP, "c", None,
    )
    assert _names(added) == ["Flask.wsgi_app"]


def test_summarization_prefers_same_file_callees(flask_graph):
    added = q._expand_with_graph_neighbours(
        [_hit("src/flask/app.py", "Flask.full_dispatch_request")],
        QueryType.SUMMARIZATION, "c", None,
    )
    names = _names(added)
    assert "Flask.handle_user_exception" in names
    assert "Flask.finalize_request" in names
    # Same-file callees take the cap's slots before name-collision fan-out.
    assert "MethodView.dispatch_request" not in names
    assert names.index("Flask.handle_user_exception") < names.index("View.dispatch_request")


def test_fuzzy_resolved_seed_is_not_expanded(flask_graph):
    # A class chunk isn't a call-graph node; its fuzzy match must be ignored.
    added = q._expand_with_graph_neighbours(
        [_hit("src/flask/ctx.py", "AppContext")], QueryType.LOOKUP, "c", None,
    )
    assert added == []


def test_neighbours_score_below_their_seed_and_skip_duplicates(flask_graph):
    seeds = [
        _hit("src/flask/app.py", "Flask.full_dispatch_request", score=0.7),
        _hit("src/flask/app.py", "Flask.wsgi_app", score=0.6),
    ]
    added = q._expand_with_graph_neighbours(seeds, QueryType.SUMMARIZATION, "c", None)
    assert "Flask.wsgi_app" not in _names(added)
    assert all(a.hybrid_score < 0.7 for a in added)


# ── LOOKUP reserved caller slot ──────────────────────────────────────────────

from retrieval.reranker import RankedResult


def _ranked(file_path: str, name: str, score: float, **meta) -> RankedResult:
    h = _hit(file_path, name)
    h.metadata.update(meta)
    return RankedResult(hybrid_result=h, rerank_score=score, rank=0)


PUSH = "src/flask/ctx.py::AppContext.push"


def _push_ranking():
    top = [
        _ranked("src/flask/ctx.py", "AppContext.push", 5.6),
        _ranked("src/flask/ctx.py", "AppContext", 5.3),
        _ranked("src/flask/app.py", "Flask.app_context", 5.1),
        _ranked("src/flask/app.py", "Flask.request_context", 4.9),
        _ranked("tests/test_testing.py", "test_client_pop_all_preserved", 4.8),
    ]
    rest = [
        _ranked("tests/test_basic.py", "test_context_test", 1.0,
                caller_of=[PUSH]),
        _ranked("src/flask/app.py", "Flask.wsgi_app", -2.9,
                caller_of=[PUSH]),
        _ranked("src/flask/ctx.py", "AppContext.__enter__", -10.6,
                caller_of=[PUSH]),
    ]
    return top, top + rest


def test_reserved_slot_takes_best_non_test_caller_and_drops_last():
    top, all_ranked = _push_ranking()
    result = q._reserve_caller_slot(top, all_ranked, max_results=5)
    assert _names(result) == [
        "AppContext.push", "AppContext", "Flask.app_context",
        "Flask.request_context", "Flask.wsgi_app",
    ]


def test_reserved_slot_appends_when_under_limit():
    top, all_ranked = _push_ranking()
    result = q._reserve_caller_slot(top[:3], all_ranked, max_results=5)
    assert _names(result)[-1] == "Flask.wsgi_app" and len(result) == 4


def test_reserved_slot_noop_when_caller_present_or_single_result():
    top, all_ranked = _push_ranking()
    with_caller = top[:4] + [all_ranked[6]]
    assert q._reserve_caller_slot(with_caller, all_ranked, 5) == with_caller
    assert q._reserve_caller_slot(top[:1], all_ranked, 1) == top[:1]


def test_reserved_slot_ignores_callers_of_other_hits():
    top, all_ranked = _push_ranking()
    for r in all_ranked[5:]:
        r.hybrid_result.metadata["caller_of"] = ["src/flask/app.py::Flask.app_context"]
    assert q._reserve_caller_slot(top, all_ranked, 5) == top


def test_caller_already_in_semantic_hits_is_marked(flask_graph):
    # Live regression: Flask.wsgi_app was already a semantic hit, so expansion
    # skipped it as a duplicate and the reserved slot never saw the call edge.
    push = _hit("src/flask/ctx.py", "AppContext.push")
    wsgi = _hit("src/flask/app.py", "Flask.wsgi_app", score=0.5)
    added = q._expand_with_graph_neighbours([push, wsgi], QueryType.LOOKUP, "c", None)
    assert "Flask.wsgi_app" not in _names(added)
    assert wsgi.metadata["caller_of"] == [PUSH]


def test_reserved_slot_anchors_on_method_when_class_chunk_ranks_first():
    # Live regression: the AppContext class chunk (260-525) out-scored
    # AppContext.push (416-444), so the slot looked for callers of the class.
    def ranked(file_path, name, score, start, end, **meta):
        r = _ranked(file_path, name, score, **meta)
        r.hybrid_result.start_line, r.hybrid_result.end_line = start, end
        return r

    top = [
        ranked("src/flask/ctx.py", "AppContext", 5.59, 260, 525),
        ranked("src/flask/ctx.py", "AppContext.push", 5.36, 416, 444),
        ranked("src/flask/app.py", "Flask.app_context", 5.09, 1484, 1502),
        ranked("src/flask/app.py", "Flask.request_context", 4.79, 1504, 1518),
        ranked("tests/test_testing.py", "test_client_pop_all_preserved", 4.76, 384, 398),
    ]
    wsgi = ranked("src/flask/app.py", "Flask.wsgi_app", -2.9, 1569, 1619, caller_of=[PUSH])
    result = q._reserve_caller_slot(top, top + [wsgi], 5)
    assert _names(result)[-1] == "Flask.wsgi_app"
    assert "test_client_pop_all_preserved" not in _names(result)

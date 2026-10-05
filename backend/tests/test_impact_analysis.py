"""
tests/test_impact_analysis.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY THIS FILE EXISTS
────────────────────
Impact analysis is the most algorithmically complex feature in
CodeSense. It depends on graph construction, traversal ordering, and
risk classification all working correctly together. A bug here doesn't
produce an obviously wrong answer — it silently returns the wrong
functions, the wrong risk levels, or the wrong paths, and the developer
trusts it.

That's the worst kind of bug: one that looks correct but misleads.

These tests exist to:

  1. VERIFY CORRECTNESS of the BFS traversal on known graphs where we
     can reason about the expected output by hand.

  2. PREVENT REGRESSIONS when the graph structure changes (e.g. if
     call_graph_builder.py changes its node ID format, or if we add
     edge weights, the tests will catch it immediately).

  3. DOCUMENT BEHAVIOR — each test is also a specification. Reading
     the tests tells you exactly what analyze_impact() is supposed to
     do in each edge case, without having to trace through the code.

  4. DEMONSTRATE ENGINEERING DISCIPLINE in interviews. FAANG
     interviewers look for candidates who write tests as part of their
     workflow, not as an afterthought. Having tests here signals that.

TEST STRATEGY
─────────────
We build synthetic call graphs using networkx directly in each test.
We do NOT load real repositories or call the parsing layer. This keeps
tests:
  - Fast (no I/O, no network)
  - Deterministic (no flakiness from real codebases changing)
  - Focused (testing the algorithm, not the integration)

Graph shape naming convention used in this file:
  - "linear chain":  A → B → C → D  (simple pipeline)
  - "diamond":       A → B → D, A → C → D  (multiple paths to same node)
  - "star":          Many nodes → B  (high-fanin function)
  - "isolated":      A node with no edges at all
  - "deep chain":    10+ nodes in a single chain (tests max_depth)

Run these tests with:
    pytest tests/test_impact_analysis.py -v
"""

import pytest
import networkx as nx

from features.impact_analysis import (
    analyze_impact,
    summarize_impact,
    RiskLevel,
    ImpactReport,
    AffectedNode,
    _classify_risk,
    _reverse_bfs,
    _count_dependents,
)


# ─────────────────────────────────────────────────────────────────────
# Fixtures — Reusable Graph Shapes
#
# Why fixtures instead of building graphs in every test?
# - DRY: graph construction logic is written once.
# - Clarity: each test focuses on assertions, not setup.
# - Composability: tests can combine fixtures if needed.
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def linear_chain_graph() -> nx.DiGraph:
    """
    A simple linear call chain: A → B → C → D

    In plain English: A calls B, B calls C, C calls D.
    If we change D:
      - C is a direct caller (distance 1, HIGH)
      - B calls C which calls D (distance 2, MEDIUM)
      - A calls B which calls C which calls D (distance 3, LOW)

    This is the simplest possible non-trivial graph and validates
    the basic BFS and risk classification pipeline end-to-end.
    """
    g = nx.DiGraph()
    g.add_edge("module/a.py::func_a", "module/b.py::func_b")
    g.add_edge("module/b.py::func_b", "module/c.py::func_c")
    g.add_edge("module/c.py::func_c", "module/d.py::func_d")
    return g


@pytest.fixture
def diamond_graph() -> nx.DiGraph:
    """
    Diamond-shaped dependency: two paths converge on func_d.

        func_a
       ↙     ↘
    func_b   func_c
       ↘     ↙
        func_d

    If we change func_d, both func_b and func_c are direct callers
    (distance 1, HIGH), and func_a is distance 2 (MEDIUM) via either path.

    This tests that BFS finds the SHORTEST path, not just any path.
    func_a should have distance 2, not distance 3 or 4.
    """
    g = nx.DiGraph()
    g.add_edge("mod/a.py::func_a", "mod/b.py::func_b")
    g.add_edge("mod/a.py::func_a", "mod/c.py::func_c")
    g.add_edge("mod/b.py::func_b", "mod/d.py::func_d")
    g.add_edge("mod/c.py::func_c", "mod/d.py::func_d")
    return g


@pytest.fixture
def star_graph() -> nx.DiGraph:
    """
    High-fanin graph: five functions all call func_center.

    func_a ──┐
    func_b ──┤
    func_c ──┼──→ func_center
    func_d ──┤
    func_e ──┘

    If we change func_center, all 5 outer nodes are direct callers
    (distance 1, HIGH). This tests that we correctly identify all
    callers when a function is heavily used.

    In real codebases, these are the most dangerous functions to
    change — utility functions called everywhere.
    """
    g = nx.DiGraph()
    center = "utils/core.py::func_center"
    for letter in "abcde":
        g.add_edge(f"services/s_{letter}.py::func_{letter}", center)
    return g


@pytest.fixture
def isolated_node_graph() -> nx.DiGraph:
    """
    A graph where func_leaf exists but nobody calls it.

    This is common for: main() functions, CLI entry points, test
    helper functions, or recently added functions not yet called.

    Expected behavior: analyze_impact returns an empty blast radius,
    not an error — the function IS in the graph, just has no callers.
    """
    g = nx.DiGraph()
    g.add_node("main.py::func_leaf")
    # func_leaf calls func_other, but nobody calls func_leaf
    g.add_edge("main.py::func_leaf", "utils/helper.py::func_other")
    return g


@pytest.fixture
def deep_chain_graph() -> nx.DiGraph:
    """
    A chain of 15 nodes: node_0 → node_1 → ... → node_14

    Used to test max_depth enforcement. If we change node_14 with
    max_depth=5, only nodes 13 down to 9 should appear in results.
    Nodes 8 and above should be cut off.
    """
    g = nx.DiGraph()
    nodes = [f"pkg/mod.py::func_{i}" for i in range(15)]
    for i in range(len(nodes) - 1):
        g.add_edge(nodes[i], nodes[i + 1])
    return g


# ─────────────────────────────────────────────────────────────────────
# Unit Tests: Helper Functions
#
# Testing helpers in isolation first makes failures easier to locate.
# If _classify_risk() is broken, all impact tests will fail — but
# knowing _classify_risk() specifically is wrong narrows the fix.
# ─────────────────────────────────────────────────────────────────────

class TestClassifyRisk:
    """Tests for the _classify_risk() helper function."""

    def test_distance_1_is_high(self):
        """Direct callers are always HIGH risk."""
        assert _classify_risk(1) == RiskLevel.HIGH

    def test_distance_2_is_medium(self):
        """Two hops away is MEDIUM risk."""
        assert _classify_risk(2) == RiskLevel.MEDIUM

    def test_distance_3_is_low(self):
        """Three hops away is LOW risk."""
        assert _classify_risk(3) == RiskLevel.LOW

    def test_distance_10_is_low(self):
        """Any distance >= 3 is LOW, no matter how large."""
        assert _classify_risk(10) == RiskLevel.LOW

    def test_distance_100_is_low(self):
        """Even very large distances are LOW, not some new tier."""
        assert _classify_risk(100) == RiskLevel.LOW


class TestCountDependents:
    """Tests for the _count_dependents() helper function."""

    def test_no_callers_returns_zero(self, isolated_node_graph):
        """A node with no incoming edges has 0 dependents."""
        count = _count_dependents(
            isolated_node_graph,
            "main.py::func_leaf"
        )
        assert count == 0

    def test_single_caller(self, linear_chain_graph):
        """func_c has exactly one caller (func_b) in the linear chain."""
        count = _count_dependents(
            linear_chain_graph,
            "module/c.py::func_c"
        )
        assert count == 1

    def test_multiple_callers(self, star_graph):
        """func_center is called by 5 functions in the star graph."""
        count = _count_dependents(
            star_graph,
            "utils/core.py::func_center"
        )
        assert count == 5

    def test_diamond_converge_node(self, diamond_graph):
        """func_d is called by both func_b and func_c in the diamond."""
        count = _count_dependents(
            diamond_graph,
            "mod/d.py::func_d"
        )
        assert count == 2


class TestReverseBFS:
    """Tests for the _reverse_bfs() internal traversal function."""

    def test_finds_all_callers_in_chain(self, linear_chain_graph):
        """
        Starting from func_d, reverse BFS should find func_c, func_b, func_a.
        """
        result = _reverse_bfs(
            linear_chain_graph,
            "module/d.py::func_d",
            max_depth=10
        )
        assert "module/c.py::func_c" in result
        assert "module/b.py::func_b" in result
        assert "module/a.py::func_a" in result

    def test_start_node_not_in_results(self, linear_chain_graph):
        """
        The changed function itself should NOT appear in the BFS results.
        We want functions AFFECTED BY the change, not the change itself.
        """
        result = _reverse_bfs(
            linear_chain_graph,
            "module/d.py::func_d",
            max_depth=10
        )
        assert "module/d.py::func_d" not in result

    def test_correct_distances_in_chain(self, linear_chain_graph):
        """
        In a linear chain A→B→C→D, changing D:
          C is distance 1, B is distance 2, A is distance 3.
        """
        result = _reverse_bfs(
            linear_chain_graph,
            "module/d.py::func_d",
            max_depth=10
        )
        assert result["module/c.py::func_c"]["distance"] == 1
        assert result["module/b.py::func_b"]["distance"] == 2
        assert result["module/a.py::func_a"]["distance"] == 3

    def test_diamond_shortest_path_distance(self, diamond_graph):
        """
        In the diamond graph, func_a can reach func_d via two paths
        (through func_b or through func_c), both of length 2.
        BFS must record distance 2, not 3 or 4.
        """
        result = _reverse_bfs(
            diamond_graph,
            "mod/d.py::func_d",
            max_depth=10
        )
        assert result["mod/a.py::func_a"]["distance"] == 2

    def test_max_depth_cuts_traversal(self, deep_chain_graph):
        """
        With max_depth=3, only the 3 nodes immediately above func_14
        should be returned (func_13, func_12, func_11).
        Nodes 10 and above should NOT appear.
        """
        result = _reverse_bfs(
            deep_chain_graph,
            "pkg/mod.py::func_14",
            max_depth=3
        )
        assert "pkg/mod.py::func_13" in result
        assert "pkg/mod.py::func_12" in result
        assert "pkg/mod.py::func_11" in result
        assert "pkg/mod.py::func_10" not in result
        assert "pkg/mod.py::func_0" not in result

    def test_empty_result_for_isolated_node(self, isolated_node_graph):
        """
        func_leaf has no callers, so reverse BFS from it returns nothing.
        """
        result = _reverse_bfs(
            isolated_node_graph,
            "main.py::func_leaf",
            max_depth=10
        )
        assert result == {}


# ─────────────────────────────────────────────────────────────────────
# Integration Tests: analyze_impact()
#
# These tests exercise the full public API end-to-end, treating the
# module as a black box. They validate that the output ImpactReport
# is correct in structure, content, and ordering.
# ─────────────────────────────────────────────────────────────────────

class TestAnalyzeImpact:
    """Integration tests for the main analyze_impact() function."""

    def test_returns_impact_report(self, linear_chain_graph):
        """analyze_impact() always returns an ImpactReport instance."""
        report = analyze_impact(
            linear_chain_graph,
            "module/d.py::func_d"
        )
        assert isinstance(report, ImpactReport)

    def test_correct_changed_function_recorded(self, linear_chain_graph):
        """The report records which function was analyzed."""
        report = analyze_impact(
            linear_chain_graph,
            "module/d.py::func_d"
        )
        assert report.changed_function == "module/d.py::func_d"

    def test_total_affected_count(self, linear_chain_graph):
        """
        In a 4-node linear chain, changing the last node affects 3 others.
        """
        report = analyze_impact(
            linear_chain_graph,
            "module/d.py::func_d"
        )
        assert report.total_affected == 3

    def test_high_risk_is_direct_caller(self, linear_chain_graph):
        """
        func_c is the direct caller of func_d and must be HIGH risk.
        """
        report = analyze_impact(
            linear_chain_graph,
            "module/d.py::func_d"
        )
        high_risk_ids = [
            n.function_id
            for n in report.affected_nodes
            if n.risk_level == RiskLevel.HIGH
        ]
        assert "module/c.py::func_c" in high_risk_ids

    def test_sorting_high_before_medium_before_low(self, linear_chain_graph):
        """
        The affected_nodes list must be sorted: HIGH first, then MEDIUM, then LOW.
        If the sort is broken, the UI will show unimportant functions at the top.
        """
        report = analyze_impact(
            linear_chain_graph,
            "module/d.py::func_d"
        )
        risk_order = {RiskLevel.HIGH: 0, RiskLevel.MEDIUM: 1, RiskLevel.LOW: 2}
        levels = [risk_order[n.risk_level] for n in report.affected_nodes]
        assert levels == sorted(levels), (
            "affected_nodes is not sorted by risk level (HIGH → MEDIUM → LOW)"
        )

    def test_call_path_starts_at_affected_node(self, linear_chain_graph):
        """
        Each node's call_path should start at that node and end at the
        changed function, showing exactly how the dependency flows.
        """
        report = analyze_impact(
            linear_chain_graph,
            "module/d.py::func_d"
        )
        for node in report.affected_nodes:
            assert node.call_path[0] == "module/d.py::func_d", (
                f"call_path for {node.function_id} should start at the "
                f"changed function (the BFS root), not at the affected node."
            )
            assert node.function_id in node.call_path, (
                f"The affected node itself ({node.function_id}) must appear "
                f"in its own call_path."
            )

    def test_unknown_function_returns_unreachable_reason(self, linear_chain_graph):
        """
        If the changed_function doesn't exist in the graph, the report
        must explain why rather than crashing or returning empty silently.
        """
        report = analyze_impact(
            linear_chain_graph,
            "nonexistent/file.py::ghost_function"
        )
        assert report.unreachable_reason is not None
        assert len(report.affected_nodes) == 0

    def test_isolated_node_returns_empty_blast_radius(self, isolated_node_graph):
        """
        func_leaf exists in the graph but has no callers.
        The report should have total_affected=0 with no error.
        """
        report = analyze_impact(
            isolated_node_graph,
            "main.py::func_leaf"
        )
        assert report.total_affected == 0
        assert report.unreachable_reason is None
        assert report.affected_nodes == []

    def test_diamond_all_callers_found(self, diamond_graph):
        """
        Changing func_d in the diamond should affect func_b, func_c (HIGH)
        and func_a (MEDIUM). All 3 must appear in the report.
        """
        report = analyze_impact(
            diamond_graph,
            "mod/d.py::func_d"
        )
        affected_ids = {n.function_id for n in report.affected_nodes}
        assert "mod/b.py::func_b" in affected_ids
        assert "mod/c.py::func_c" in affected_ids
        assert "mod/a.py::func_a" in affected_ids
        assert report.total_affected == 3

    def test_diamond_func_a_is_medium_not_high(self, diamond_graph):
        """
        func_a is 2 hops from func_d (via either path), so it must be
        MEDIUM risk, not HIGH. Both BFS paths lead to distance 2.
        """
        report = analyze_impact(
            diamond_graph,
            "mod/d.py::func_d"
        )
        func_a = next(
            n for n in report.affected_nodes
            if n.function_id == "mod/a.py::func_a"
        )
        assert func_a.risk_level == RiskLevel.MEDIUM
        assert func_a.distance == 2

    def test_max_depth_limits_results(self, deep_chain_graph):
        """
        With max_depth=3, only the 3 nearest callers should appear.
        The rest of the chain (nodes 0-10) should not be in the report.
        """
        report = analyze_impact(
            deep_chain_graph,
            "pkg/mod.py::func_14",
            max_depth=3
        )
        assert report.total_affected == 3
        affected_ids = {n.function_id for n in report.affected_nodes}
        assert "pkg/mod.py::func_13" in affected_ids
        assert "pkg/mod.py::func_10" not in affected_ids

    def test_star_all_callers_are_high_risk(self, star_graph):
        """
        In the star graph, all 5 outer nodes directly call func_center,
        so all of them must be HIGH risk with distance 1.
        """
        report = analyze_impact(
            star_graph,
            "utils/core.py::func_center"
        )
        assert report.total_affected == 5
        for node in report.affected_nodes:
            assert node.risk_level == RiskLevel.HIGH
            assert node.distance == 1

    def test_max_depth_reached_is_correct(self, linear_chain_graph):
        """
        In the linear chain (4 nodes), changing the last node (func_d)
        affects 3 others at distances 1, 2, 3.
        max_depth_reached should be 3.
        """
        report = analyze_impact(
            linear_chain_graph,
            "module/d.py::func_d"
        )
        assert report.max_depth_reached == 3

    def test_dependents_count_populated(self, star_graph):
        """
        For func_center (called by 5 nodes), each affected AffectedNode's
        dependents_count should reflect the in-degree of that node.
        The outer nodes (func_a through func_e) call nobody in this graph,
        so their dependents_count is 0.
        """
        report = analyze_impact(
            star_graph,
            "utils/core.py::func_center"
        )
        for node in report.affected_nodes:
            # In the star graph, none of the outer nodes have callers
            assert node.dependents_count == 0


# ─────────────────────────────────────────────────────────────────────
# Tests: summarize_impact()
# ─────────────────────────────────────────────────────────────────────

class TestSummarizeImpact:
    """Tests for the plain-English summarizer."""

    def test_summary_contains_changed_function(self, linear_chain_graph):
        """The summary must mention the function that was changed."""
        report = analyze_impact(
            linear_chain_graph,
            "module/d.py::func_d"
        )
        summary = summarize_impact(report)
        assert "module/d.py::func_d" in summary

    def test_summary_contains_total_count(self, linear_chain_graph):
        """The summary must state how many functions are affected."""
        report = analyze_impact(
            linear_chain_graph,
            "module/d.py::func_d"
        )
        summary = summarize_impact(report)
        assert "3" in summary  # 3 affected functions

    def test_unreachable_summary(self, linear_chain_graph):
        """If analysis failed, the summary must explain why."""
        report = analyze_impact(
            linear_chain_graph,
            "doesnt/exist.py::ghost"
        )
        summary = summarize_impact(report)
        assert "unavailable" in summary.lower() or "not found" in summary.lower()

    def test_zero_blast_radius_summary(self, isolated_node_graph):
        """If nothing is affected, the summary should say so clearly."""
        report = analyze_impact(
            isolated_node_graph,
            "main.py::func_leaf"
        )
        summary = summarize_impact(report)
        assert "no detected blast radius" in summary.lower() or "no other" in summary.lower()

    def test_summary_is_string(self, linear_chain_graph):
        """summarize_impact() must always return a string, never None or an object."""
        report = analyze_impact(
            linear_chain_graph,
            "module/d.py::func_d"
        )
        assert isinstance(summarize_impact(report), str)


# ─────────────────────────────────────────────────────────────────────
# Edge Case Tests
# ─────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """
    Tests for unusual or extreme inputs that real codebases will produce.
    These are the cases most likely to cause silent failures.
    """

    def test_empty_graph(self):
        """
        An empty graph with no nodes. analyze_impact should not crash
        and should return a meaningful unreachable_reason.
        """
        empty_graph = nx.DiGraph()
        report = analyze_impact(empty_graph, "any/file.py::any_func")
        assert report.unreachable_reason is not None
        assert report.total_affected == 0

    def test_single_node_no_edges(self):
        """A graph with exactly one node and no edges — isolated function."""
        g = nx.DiGraph()
        g.add_node("solo/file.py::solo_func")
        report = analyze_impact(g, "solo/file.py::solo_func")
        assert report.total_affected == 0
        assert report.unreachable_reason is None

    def test_self_loop_does_not_cause_infinite_loop(self):
        """
        A function that calls itself (recursion) creates a self-loop in the
        call graph. BFS must not get stuck in an infinite loop.
        The self-loop should be harmlessly skipped by the "already visited" check.
        """
        g = nx.DiGraph()
        g.add_edge("module/a.py::recursive_func", "module/a.py::recursive_func")
        g.add_edge("module/b.py::caller", "module/a.py::recursive_func")
        # Should complete without hanging
        report = analyze_impact(g, "module/a.py::recursive_func")
        assert report.total_affected == 1
        assert report.affected_nodes[0].function_id == "module/b.py::caller"

    def test_max_depth_zero_returns_empty(self, linear_chain_graph):
        """
        max_depth=0 means don't traverse at all. Should return empty
        blast radius without crashing.
        """
        report = analyze_impact(
            linear_chain_graph,
            "module/d.py::func_d",
            max_depth=0
        )
        assert report.total_affected == 0

    def test_function_id_with_special_characters(self):
        """
        Real codebases have function names with underscores, numbers,
        and occasionally special characters. The graph traversal must
        handle these correctly.
        """
        g = nx.DiGraph()
        g.add_edge(
            "src/auth/__init__.py::_validate_jwt_token_v2",
            "src/db/models.py::User__get_by_email"
        )
        report = analyze_impact(g, "src/db/models.py::User__get_by_email")
        assert report.total_affected == 1
        assert report.affected_nodes[0].function_id == \
               "src/auth/__init__.py::_validate_jwt_token_v2"
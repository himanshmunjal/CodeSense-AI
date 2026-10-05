"""
features/impact_analysis.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY THIS FILE EXISTS
────────────────────
One of the two "differentiator features" described in the architecture.
Every developer has experienced this problem: you need to change a
function but you're not sure what else will break. In a small codebase
you can grep and reason about it manually. In a 200,000-line monorepo
with layered abstractions, that's impossible.

This module answers the question: "If I change function X, what is the
full blast radius?" It does so by traversing the call graph — the
directed graph built in parsing/call_graph_builder.py — in REVERSE. If
function A calls B which calls C, and you're changing C, then B and A
are both at risk. The deeper A is from C, the lower its risk, but it's
still in the blast radius.

This is what sets CodeSense apart from generic RAG chatbots. A chatbot
can describe what a function does. This module can tell you what breaks
if you touch it — and rank the results by risk so an engineer knows
where to focus their testing effort.

WHY THIS MATTERS IN FAANG INTERVIEWS
─────────────────────────────────────
Impact analysis is a real problem that internal tools at Google (Kythe),
Meta (Glean), and Amazon (Brazil build system) solve at scale. Building
an open-source version of this, even at smaller scale, demonstrates:
  1. You understand the problem beyond just "I built an AI chatbot"
  2. You can reason about graph algorithms (BFS, reverse traversal)
  3. You understand software engineering concerns like dependency risk

DEPENDENCIES
────────────
- networkx: the call graph built by parsing/call_graph_builder.py is a
  networkx DiGraph where nodes are function identifiers and directed
  edges represent "caller → callee" relationships.
- The graph is passed in at call time — this module is stateless and
  does not manage persistence or loading.
"""

from collections import deque

import networkx as nx
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from loguru import logger


# ─────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    """
    Risk classification for each node in the blast radius.

    Risk is determined by how far a node is from the changed function
    in the reverse call graph. Direct callers are HIGH risk because
    they call the changed function directly and will be immediately
    affected by signature or behavior changes. Indirect callers are
    MEDIUM or LOW depending on how many hops away they are.

    Why use an Enum instead of a plain string?
    - Prevents typos in risk level names throughout the codebase.
    - Makes it easy to sort/compare risk levels programmatically.
    - Serializes cleanly to JSON via FastAPI's response models.
    """
    HIGH   = "HIGH"    # Distance 1 — direct caller of the changed function
    MEDIUM = "MEDIUM"  # Distance 2 — calls a direct caller
    LOW    = "LOW"     # Distance 3+ — transitively affected


@dataclass
class AffectedNode:
    """
    Represents a single function in the blast radius of a change.

    Each AffectedNode captures not just which function is affected,
    but WHY it's affected (the call path from it to the changed
    function) and how SEVERELY (risk level based on distance).

    Fields
    ──────
    function_id : str
        Unique identifier for this function in the call graph. Format:
        "{file_path}::{function_name}", e.g. "auth/login.py::verify_token".
        This format matches what call_graph_builder.py assigns as node IDs.

    distance : int
        Number of hops from this node to the changed function in the
        reverse call graph. Distance 1 = direct caller.

    risk_level : RiskLevel
        Derived from distance. HIGH for distance 1, MEDIUM for 2, LOW for 3+.

    call_path : list[str]
        The exact chain of function calls from this node down to the
        changed function. Useful for displaying "why is this affected?"
        in the UI. Example: ["api/router.py::handle_request",
        "auth/middleware.py::require_auth", "auth/login.py::verify_token"]

    dependents_count : int
        How many OTHER functions in the entire graph depend on this node.
        A node with high dependents_count that is already in the blast
        radius is especially dangerous — changing it has a cascading effect.
        This helps the engineer prioritize which affected nodes to test first.
    """
    function_id:      str
    distance:         int
    risk_level:       RiskLevel
    call_path:        list[str]      = field(default_factory=list)
    dependents_count: int            = 0


@dataclass
class ImpactReport:
    """
    The full output of an impact analysis run.

    This is what gets serialized and returned by the /impact-analysis
    API endpoint. It contains everything an engineer needs to understand
    the consequences of changing a function.

    Fields
    ──────
    changed_function : str
        The function ID that was analyzed (the "source" of the change).

    affected_nodes : list[AffectedNode]
        All functions in the blast radius, sorted by risk (HIGH first)
        and then by distance (closest first within same risk tier).

    total_affected : int
        Total count of affected functions. Quick summary metric for the UI.

    max_depth_reached : int
        How deep the traversal went before finding no more callers. Gives
        a sense of how interconnected the changed function is.

    unreachable_reason : str | None
        If the changed_function was not found in the graph, this field
        explains why (e.g. "function not indexed", "isolated node").
        None when analysis succeeded normally.
    """
    changed_function:   str
    affected_nodes:     list[AffectedNode] = field(default_factory=list)
    total_affected:     int                = 0
    max_depth_reached:  int                = 0
    unreachable_reason: Optional[str]      = None


# ─────────────────────────────────────────────────────────────────────
# Risk Classification Helper
# ─────────────────────────────────────────────────────────────────────

def _classify_risk(distance: int) -> RiskLevel:
    """
    Converts a graph distance into a human-readable risk level.

    Why these thresholds?
    - Distance 1 (HIGH): This function directly calls the changed one.
      If the signature changes or the return value changes, this breaks
      immediately and obviously.
    - Distance 2 (MEDIUM): This function calls a direct caller. It is
      indirectly affected — it may still break if the behavior change
      propagates upward through the call stack.
    - Distance 3+ (LOW): Far enough away that the impact is speculative.
      Still worth knowing about, but lower priority for testing.

    These thresholds are intentionally simple. You could make them
    configurable via settings, but for v1 these are sensible defaults
    that match how senior engineers intuitively reason about risk.

    Args:
        distance: Number of hops in the reverse call graph.

    Returns:
        RiskLevel enum value.
    """
    if distance == 1:
        return RiskLevel.HIGH
    elif distance == 2:
        return RiskLevel.MEDIUM
    else:
        return RiskLevel.LOW


# ─────────────────────────────────────────────────────────────────────
# Core Traversal Logic
# ─────────────────────────────────────────────────────────────────────

def _reverse_bfs(
    graph:            nx.DiGraph,
    start_node:       str,
    max_depth:        int,
) -> dict[str, dict]:
    """
    Performs a Breadth-First Search on the REVERSED call graph.

    Why BFS instead of DFS?
    - BFS naturally gives us the shortest path distance from each node
      to the start_node. This is exactly what we want for risk
      classification — we care about the MINIMUM distance (the most
      direct dependency path), not just any path.
    - DFS would give us paths but not minimum distances efficiently.

    Why reverse the graph?
    - The call graph has edges pointing from caller → callee.
      (A calls B means there is an edge A → B.)
    - To find "who is affected if B changes", we need to traverse
      BACKWARDS: find all nodes that have B as a callee, i.e. all A
      where edge A → B exists.
    - networkx.reverse() flips all edge directions, turning callee → caller
      edges so we can do a normal forward BFS to find all callers.

    Args:
        graph:      The original call graph (caller → callee direction).
        start_node: The function being changed. BFS starts here on the
                    reversed graph.
        max_depth:  Maximum number of hops to traverse. Prevents
                    runaway traversal on deeply interconnected graphs.
                    Default in the public API is 10.

    Returns:
        A dict mapping each visited node_id →
            { "distance": int, "path": list[str] }
        The start_node itself is NOT included in the results (we only
        want nodes AFFECTED BY the change, not the changed node itself).
    """
    reversed_graph = graph.reverse(copy=False)
    # copy=False avoids duplicating the graph in memory — safe here
    # because we only read from reversed_graph, never modify it.

    visited: dict[str, dict] = {}
    # Queue entries are tuples of (node_id, current_depth, path_so_far)
    queue: deque[tuple[str, int, list[str]]] = deque([(start_node, 0, [start_node])])

    while queue:
        current_node, depth, path = queue.popleft()

        # Stop expanding beyond the max depth limit
        if depth >= max_depth:
            continue

        for neighbor in reversed_graph.neighbors(current_node):
            # Skip if already visited — BFS guarantees the first visit
            # is the shortest path, so we never need to revisit. The start
            # node itself is never "affected" (recursion / call cycles lead
            # back to it).
            if neighbor in visited or neighbor == start_node:
                continue

            neighbor_depth = depth + 1
            neighbor_path  = path + [neighbor]

            visited[neighbor] = {
                "distance": neighbor_depth,
                "path":     neighbor_path,
            }

            queue.append((neighbor, neighbor_depth, neighbor_path))

    return visited


def _count_dependents(graph: nx.DiGraph, node_id: str) -> int:
    """
    Returns how many other nodes in the graph have an incoming edge
    pointing TO node_id — i.e., how many functions call this node.

    This is the in-degree of the node in the REVERSED graph, which
    equals the in-degree of the node in the original graph.

    Why do we compute this?
    When ranking blast radius nodes, raw distance is not enough. A
    function that is 3 hops away but is called by 50 other functions
    is more dangerous to leave untested than a function 1 hop away
    called by nobody else. dependents_count gives the engineer this
    signal without them having to investigate manually.

    Args:
        graph:   The original call graph.
        node_id: The function to count dependents for.

    Returns:
        Integer count of direct callers of this node.
    """
    return graph.in_degree(node_id)


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

def analyze_impact(
    call_graph:       nx.DiGraph,
    changed_function: str,
    max_depth:        int = 10,
) -> ImpactReport:
    """
    Main entry point. Given a call graph and a function that is about
    to change, returns a ranked ImpactReport of everything in the
    blast radius.

    This is called by api/routes/impact.py when the frontend submits
    an impact analysis request.

    Algorithm
    ─────────
    1. Validate that the changed function exists in the graph.
    2. Run reverse BFS from the changed function up to max_depth hops.
    3. For each affected node, compute risk level and dependents count.
    4. Sort results: HIGH risk first, then MEDIUM, then LOW. Within
       the same risk tier, sort by dependents_count descending so the
       most critical nodes appear first.
    5. Package everything into an ImpactReport and return.

    Args:
        call_graph:       networkx DiGraph from call_graph_builder.py.
                          Nodes are function IDs, edges are caller→callee.
        changed_function: The function ID being analyzed. Must match
                          the node ID format used in call_graph_builder:
                          "{file_path}::{function_name}"
        max_depth:        How many hops to traverse. Default 10 is
                          sufficient for most real codebases without
                          causing performance issues. Set lower for
                          faster responses on large graphs.

    Returns:
        ImpactReport dataclass. If the function is not in the graph,
        returns a report with unreachable_reason set and empty
        affected_nodes.

    Example
    ───────
        graph = build_call_graph(parsed_repo)  # from call_graph_builder
        report = analyze_impact(graph, "auth/login.py::verify_token")

        for node in report.affected_nodes:
            print(f"{node.risk_level}: {node.function_id}")
            print(f"  Path: {' → '.join(node.call_path)}")
    """
    logger.info(
        f"Starting impact analysis for '{changed_function}' "
        f"(max_depth={max_depth}, graph_nodes={call_graph.number_of_nodes()})"
    )

    # ── Step 1: Validate the changed function exists ──────────────
    if changed_function not in call_graph.nodes:
        logger.warning(
            f"Function '{changed_function}' not found in call graph. "
            f"It may not have been indexed or may be a leaf with no calls."
        )
        return ImpactReport(
            changed_function=changed_function,
            unreachable_reason=(
                f"'{changed_function}' was not found in the indexed call graph. "
                f"Possible reasons: the function was not parsed (unsupported language, "
                f"syntax error), the repo has not been re-indexed after this function "
                f"was added, or the function ID format is incorrect."
            ),
        )

    # ── Step 2: Run reverse BFS ───────────────────────────────────
    visited = _reverse_bfs(
        graph=call_graph,
        start_node=changed_function,
        max_depth=max_depth,
    )

    if not visited:
        # The function exists but nothing calls it — it's a root or
        # an entry point (e.g. a main() function or a CLI command).
        logger.info(
            f"'{changed_function}' has no callers. "
            f"It is either a root node or an isolated entry point."
        )
        return ImpactReport(
            changed_function=changed_function,
            affected_nodes=[],
            total_affected=0,
            max_depth_reached=0,
            unreachable_reason=None,
        )

    # ── Step 3: Build AffectedNode objects ───────────────────────
    affected_nodes: list[AffectedNode] = []

    for node_id, info in visited.items():
        distance         = info["distance"]
        path             = info["path"]
        risk_level       = _classify_risk(distance)
        dependents_count = _count_dependents(call_graph, node_id)

        affected_nodes.append(AffectedNode(
            function_id=node_id,
            distance=distance,
            risk_level=risk_level,
            call_path=path,
            dependents_count=dependents_count,
        ))

    # ── Step 4: Sort the results ──────────────────────────────────
    # Primary sort: risk level order (HIGH → MEDIUM → LOW)
    # Secondary sort: dependents_count descending (more callers = more danger)
    # Tertiary sort: distance ascending (closer = more direct impact)
    risk_order = {RiskLevel.HIGH: 0, RiskLevel.MEDIUM: 1, RiskLevel.LOW: 2}

    affected_nodes.sort(key=lambda n: (
        risk_order[n.risk_level],
        -n.dependents_count,
        n.distance,
    ))

    max_depth_reached = max(n.distance for n in affected_nodes)

    logger.info(
        f"Impact analysis complete. "
        f"Affected: {len(affected_nodes)} functions, "
        f"Max depth: {max_depth_reached}, "
        f"HIGH: {sum(1 for n in affected_nodes if n.risk_level == RiskLevel.HIGH)}, "
        f"MEDIUM: {sum(1 for n in affected_nodes if n.risk_level == RiskLevel.MEDIUM)}, "
        f"LOW: {sum(1 for n in affected_nodes if n.risk_level == RiskLevel.LOW)}"
    )

    return ImpactReport(
        changed_function=changed_function,
        affected_nodes=affected_nodes,
        total_affected=len(affected_nodes),
        max_depth_reached=max_depth_reached,
        unreachable_reason=None,
    )


def summarize_impact(report: ImpactReport) -> str:
    """
    Returns a plain-English summary of an ImpactReport for use in
    the generation layer.

    Why does this exist separately from analyze_impact?
    The generation layer (generation/generator.py) needs to inject
    impact analysis results into prompts alongside code context. A
    structured dataclass is hard to inject into a prompt. This function
    converts the report into a concise, human-readable string that
    the LLM can reason about alongside the retrieved code chunks.

    Args:
        report: An ImpactReport returned by analyze_impact().

    Returns:
        A multi-line string summarizing the impact in plain English.

    Example output:
        "Changing 'auth/login.py::verify_token' affects 12 functions.
         HIGH risk (3): api/router.py::handle_request, ...
         MEDIUM risk (5): services/user.py::get_user, ...
         LOW risk (4): utils/logging.py::log_event, ..."
    """
    if report.unreachable_reason:
        return f"Impact analysis unavailable: {report.unreachable_reason}"

    if report.total_affected == 0:
        return (
            f"Changing '{report.changed_function}' has no detected blast radius. "
            f"No other functions in the indexed codebase call this function directly "
            f"or transitively."
        )

    # Group by risk level for a clean summary
    by_risk: dict[RiskLevel, list[str]] = {
        RiskLevel.HIGH:   [],
        RiskLevel.MEDIUM: [],
        RiskLevel.LOW:    [],
    }
    for node in report.affected_nodes:
        by_risk[node.risk_level].append(node.function_id)

    lines = [
        f"Changing '{report.changed_function}' affects "
        f"{report.total_affected} function(s) across {report.max_depth_reached} "
        f"level(s) of the call stack.",
    ]
    for risk, nodes in by_risk.items():
        if nodes:
            sample = ", ".join(nodes[:3])
            suffix = f" (and {len(nodes) - 3} more)" if len(nodes) > 3 else ""
            lines.append(f"  {risk.value} risk ({len(nodes)}): {sample}{suffix}")

    return "\n".join(lines)
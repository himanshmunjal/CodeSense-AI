"""
call_graph_builder.py
─────────────────────
PURPOSE
-------
Builds a directed call graph from the structured entities extracted by
entity_extractor.py.  A call graph is a data structure where every node
is a function/method and every directed edge A → B means "A calls B".

WHY THIS FILE EXISTS
--------------------
Plain semantic search cannot answer questions like:
  • "What breaks if I rename `authenticate_user`?"
  • "How many hops away is the database layer from the API surface?"
  • "Which functions are never called (dead code)?"

These are graph problems, not text-search problems.  By representing the
codebase as a NetworkX DiGraph we can run standard graph algorithms
(BFS, DFS, PageRank, topological sort) to answer them cheaply and
accurately — without touching the LLM at all.

This graph is also the backbone of the Impact Analysis feature
(features/impact_analysis.py), which traverses it in reverse to produce
the "blast radius" of any change.

USED BY
-------
- features/impact_analysis.py          reverse traversal for blast-radius
- retrieval/graph_retriever.py          graph-aware retrieval queries
- embeddings/graph_embedder.py          node2vec needs this graph as input
- indexing/metadata_builder.py          centrality scores stored as metadata
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CallEdge:
    """
    Represents a single directed call relationship between two functions.

    Attributes
    ----------
    caller : str
        Fully-qualified name of the calling function, e.g. "auth.login.validate_token".
    callee : str
        Fully-qualified name of the called function.
    file_path : str
        Source file where the call expression appears (useful for filtering
        the graph to a single module during impact analysis).
    line_number : int
        Line in `file_path` where the call occurs (used to jump to source
        in the frontend SourceViewer).
    call_type : str
        One of "direct", "conditional", "async", "recursive".
        Helps rank edges during blast-radius calculations — a conditional
        call is lower-risk than an unconditional direct call.
    """
    caller: str
    callee: str
    file_path: str
    line_number: int
    call_type: str = "direct"


@dataclass
class FunctionNode:
    """
    Metadata stored on each node of the call graph.

    Attributes
    ----------
    qualified_name : str
        Unique identifier for the function: "<module>.<class>.<function>".
    file_path : str
        Absolute path of the source file.
    start_line : int
        First line of the function definition.
    end_line : int
        Last line of the function definition.
    language : str
        Source language: "python" | "javascript" | "java" | "typescript".
    is_public : bool
        Whether the function is part of the module's public API.
        Derived from naming conventions (leading underscore = private in Python)
        or explicit access modifiers (public/private/protected in Java).
    """
    qualified_name: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    is_public: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Core Builder
# ─────────────────────────────────────────────────────────────────────────────

class CallGraphBuilder:
    """
    Constructs and manages the directed call graph for an entire repository.

    The graph is stored as a NetworkX DiGraph so that every standard graph
    algorithm (shortest path, connected components, PageRank, etc.) is
    immediately available without custom implementation.

    Parameters
    ----------
    repo_name : str
        Human-readable label for the repository.  Used as a graph attribute
        and in log messages.
    """

    def __init__(self, repo_name: str) -> None:
        self.repo_name = repo_name
        # DiGraph because calls are directional: A → B ≠ B → A.
        # This asymmetry is critical for reverse-traversal in impact analysis.
        self._graph: nx.DiGraph = nx.DiGraph(repo=repo_name)
        logger.info("Initialized empty call graph for repo '%s'", repo_name)

    # ── Node Operations ───────────────────────────────────────────────────────

    def add_function(self, node: FunctionNode) -> None:
        """
        Add a function as a node in the call graph.

        Node attributes are stored directly on the NetworkX node so they
        are available during graph traversal without a separate lookup.

        Parameters
        ----------
        node : FunctionNode
            Metadata about the function to register.
        """
        self._graph.add_node(
            node.qualified_name,
            file_path=node.file_path,
            start_line=node.start_line,
            end_line=node.end_line,
            language=node.language,
            is_public=node.is_public,
            **node.extra,
        )
        logger.debug("Added node: %s", node.qualified_name)

    def add_functions_bulk(self, nodes: list[FunctionNode]) -> None:
        """
        Batch-insert multiple function nodes.

        Prefer this over repeated add_function() calls for large repos —
        NetworkX batch operations avoid repeated internal bookkeeping overhead.

        Parameters
        ----------
        nodes : list[FunctionNode]
            All function nodes extracted from a parsed file or module.
        """
        nx_nodes = [
            (
                n.qualified_name,
                {
                    "file_path": n.file_path,
                    "start_line": n.start_line,
                    "end_line": n.end_line,
                    "language": n.language,
                    "is_public": n.is_public,
                    **n.extra,
                },
            )
            for n in nodes
        ]
        self._graph.add_nodes_from(nx_nodes)
        logger.debug("Bulk-added %d function nodes", len(nodes))

    # ── Edge Operations ───────────────────────────────────────────────────────

    def add_call(self, edge: CallEdge) -> None:
        """
        Record a single call relationship as a directed edge.

        If either the caller or callee node does not yet exist in the graph
        (e.g. it belongs to an external library), a stub node is created
        automatically so the graph stays structurally consistent.

        Parameters
        ----------
        edge : CallEdge
            The call relationship to record.
        """
        # Auto-create stub nodes for external/unresolved functions.
        # These stubs are marked so downstream code can filter them out.
        for name in (edge.caller, edge.callee):
            if name not in self._graph:
                self._graph.add_node(name, is_stub=True)
                logger.debug("Auto-created stub node for unresolved symbol: %s", name)

        self._graph.add_edge(
            edge.caller,
            edge.callee,
            file_path=edge.file_path,
            line_number=edge.line_number,
            call_type=edge.call_type,
        )

    def add_calls_bulk(self, edges: list[CallEdge]) -> None:
        """
        Batch-insert multiple call edges.

        Used after parsing an entire file to flush all detected call
        relationships at once.

        Parameters
        ----------
        edges : list[CallEdge]
            All call edges extracted from a single parsed file.
        """
        for edge in edges:
            self.add_call(edge)
        logger.debug("Bulk-added %d call edges", len(edges))

    # ── Graph Queries ─────────────────────────────────────────────────────────

    def get_callees(self, function_name: str) -> list[str]:
        """
        Return all functions directly called by `function_name`.

        This is a forward traversal (what does this function depend on?).
        Useful for dependency analysis and for understanding what a function
        needs to work correctly.

        Parameters
        ----------
        function_name : str
            Qualified name of the function to inspect.

        Returns
        -------
        list[str]
            Qualified names of all directly called functions.
        """
        if function_name not in self._graph:
            logger.warning("Function '%s' not found in call graph", function_name)
            return []
        return list(self._graph.successors(function_name))

    def get_callers(self, function_name: str) -> list[str]:
        """
        Return all functions that directly call `function_name`.

        This is a reverse/backward traversal (who depends on me?).
        This is the first step of impact analysis — if you change
        `function_name`, every direct caller is immediately affected.

        Parameters
        ----------
        function_name : str
            Qualified name of the function to inspect.

        Returns
        -------
        list[str]
            Qualified names of all direct callers.
        """
        if function_name not in self._graph:
            logger.warning("Function '%s' not found in call graph", function_name)
            return []
        return list(self._graph.predecessors(function_name))

    def get_blast_radius(
        self,
        function_name: str,
        max_depth: int = 5,
    ) -> dict[str, int]:
        """
        Compute the full blast radius of changing `function_name`.

        Performs a BFS on the *reversed* graph starting from `function_name`.
        Each reachable node's value is its BFS depth (distance from the
        changed function).  Depth 1 = direct callers, depth 2 = their callers, etc.

        Depth is used by features/impact_analysis.py to assign risk levels:
          depth 1 → HIGH risk
          depth 2 → MEDIUM risk
          depth 3+ → LOW risk

        Parameters
        ----------
        function_name : str
            The function being changed.
        max_depth : int
            Maximum BFS depth.  Prevents runaway traversal on densely
            connected utility functions (like logging or config helpers)
            that are called everywhere.

        Returns
        -------
        dict[str, int]
            Mapping of affected function names to their BFS depth.
        """
        if function_name not in self._graph:
            logger.warning("Function '%s' not found in call graph", function_name)
            return {}

        reversed_graph = self._graph.reverse(copy=False)
        blast: dict[str, int] = {}

        # Standard BFS on reversed graph.
        queue = [(function_name, 0)]
        visited = {function_name}

        while queue:
            current, depth = queue.pop(0)
            if depth > max_depth:
                break
            if current != function_name:
                blast[current] = depth
            for neighbor in reversed_graph.successors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        logger.info(
            "Blast radius of '%s': %d affected functions (max_depth=%d)",
            function_name, len(blast), max_depth,
        )
        return blast

    def get_all_paths(self, source: str, target: str) -> list[list[str]]:
        """
        Find all simple paths from `source` to `target` in the call graph.

        Used to explain *why* two modules are coupled — "here are the call
        chains that connect them."  Displayed in the frontend as a chain
        of function names the user can click to navigate to source.

        Note: simple_all_paths can be expensive on large graphs.  Only
        call this on demand (user-triggered), not in a hot loop.

        Parameters
        ----------
        source : str
            Starting function (caller side).
        target : str
            Ending function (callee side).

        Returns
        -------
        list[list[str]]
            Each inner list is one complete call chain from source to target.
        """
        try:
            return list(nx.all_simple_paths(self._graph, source, target))
        except nx.NetworkXNoPath:
            return []
        except nx.NodeNotFound as exc:
            logger.warning("Path query failed — node not found: %s", exc)
            return []

    # ── Graph Metrics ─────────────────────────────────────────────────────────

    def compute_centrality(self) -> dict[str, float]:
        """
        Compute PageRank centrality for every node in the call graph.

        PageRank is a better centrality measure than simple in-degree
        for call graphs because it accounts for the *importance* of callers,
        not just their count.  A function called only by the main entry point
        has lower centrality than one called by many critical subsystems.

        Centrality scores are stored as node metadata and later indexed in
        Qdrant so that retrieval can prefer "important" functions when
        answering broad queries like "explain the authentication system."

        Returns
        -------
        dict[str, float]
            Mapping of function name to its PageRank score (sum = 1.0).
        """
        scores = nx.pagerank(self._graph)
        # Write scores back onto the nodes for easy access during traversal.
        nx.set_node_attributes(self._graph, scores, name="pagerank")
        logger.info("Computed PageRank for %d nodes", len(scores))
        return scores

    def get_entry_points(self) -> list[str]:
        """
        Identify entry-point functions — nodes with in-degree 0.

        Entry points are functions that are never called by other code in
        the repository.  They are typically:
          • HTTP route handlers
          • CLI command functions
          • Celery task definitions
          • Test functions

        Knowing the entry points is useful for generating high-level
        summaries ("this repo exposes these 12 API endpoints") and for
        seeding graph traversal during impact analysis.

        Returns
        -------
        list[str]
            Qualified names of all entry-point functions.
        """
        return [n for n, deg in self._graph.in_degree() if deg == 0]

    def get_hub_functions(self, top_n: int = 10) -> list[tuple[str, int]]:
        """
        Return the `top_n` most-called functions, sorted by in-degree.

        Hub functions are high-risk targets for refactoring because they
        are depended on by many callers.  Surfacing them helps developers
        make informed decisions about what to touch carefully.

        Parameters
        ----------
        top_n : int
            How many top hubs to return.

        Returns
        -------
        list[tuple[str, int]]
            Each tuple is (function_name, in_degree), sorted descending.
        """
        in_degrees = sorted(
            self._graph.in_degree(), key=lambda x: x[1], reverse=True
        )
        return in_degrees[:top_n]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, output_path: str | Path) -> None:
        """
        Serialize the call graph to a GraphML file on disk.

        GraphML is chosen over pickle because:
          1. It is human-readable XML — you can inspect it in any text editor.
          2. It is language-agnostic — other tools can consume it.
          3. It preserves all node/edge attributes.

        The saved file is loaded back by graph_embedder.py and
        graph_retriever.py without re-parsing the entire repository.

        Parameters
        ----------
        output_path : str | Path
            Destination file path, e.g. ".cache/my_repo_call_graph.graphml".
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(self._graph, str(path))
        logger.info("Call graph saved to %s (%d nodes, %d edges)",
                    path, self._graph.number_of_nodes(), self._graph.number_of_edges())

    @classmethod
    def load(cls, input_path: str | Path, repo_name: str = "unknown") -> "CallGraphBuilder":
        """
        Deserialize a previously saved call graph from a GraphML file.

        Avoids re-parsing the whole repository on every startup.  Only
        nodes whose source file has changed (detected by change_detector.py)
        need to be re-parsed and re-inserted.

        Parameters
        ----------
        input_path : str | Path
            Path to a GraphML file previously created by save().
        repo_name : str
            Label to attach to the reconstructed graph object.

        Returns
        -------
        CallGraphBuilder
            A fully initialized builder with the loaded graph.
        """
        builder = cls(repo_name)
        builder._graph = nx.read_graphml(str(input_path))
        logger.info("Call graph loaded from %s (%d nodes, %d edges)",
                    input_path,
                    builder._graph.number_of_nodes(),
                    builder._graph.number_of_edges())
        return builder

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def graph(self) -> nx.DiGraph:
        """
        Expose the raw NetworkX DiGraph.

        Used by graph_embedder.py (which needs the raw graph to run node2vec)
        and by retrieval/graph_retriever.py (which may run custom traversals).
        """
        return self._graph

    @property
    def node_count(self) -> int:
        """Total number of function nodes in the graph."""
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        """Total number of call relationships recorded in the graph."""
        return self._graph.number_of_edges()

    def __repr__(self) -> str:
        return (
            f"CallGraphBuilder(repo='{self.repo_name}', "
            f"nodes={self.node_count}, edges={self.edge_count})"
        )
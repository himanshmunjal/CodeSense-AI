"""
retrieval/graph_retriever.py
════════════════════════════

WHY THIS FILE EXISTS
────────────────────
Vector similarity search is powerful, but it has a fundamental blind spot:
it cannot answer questions about relationships between code entities.

Consider the query: "What functions call authenticate()?"

A semantic search for "authenticate" will return the *definition* of
authenticate() and other functions with similar names — but it cannot tell
you which functions in the codebase *invoke* authenticate(). That information
is not in the text of any single function; it lives in the edges of the call
graph.

This file answers that class of questions. It operates on the call graph built
by parsing/call_graph_builder.py — a directed graph where:
    Nodes = functions/methods (identified by file_path::function_name)
    Edges = "A calls B" (directed, A → B)

Given a starting node, this file can:
    (1) Find all callers of a function (reverse traversal: who calls X?)
    (2) Find all callees of a function (forward traversal: what does X call?)
    (3) Find the blast radius of a change (impact analysis: what breaks if X changes?)
    (4) Find the call chain between two functions (shortest path: how does main reach X?)
    (5) Find all functions in a given module/file (subgraph extraction)

WHY GRAPH TRAVERSAL SPECIFICALLY
──────────────────────────────────
The call graph is a property of the *structure* of the code, not its *content*.
You cannot learn it from reading individual functions in isolation — you must
read all functions and track which identifiers are called.

This is the key insight that separates CodeSense from generic RAG systems:
generic RAG indexes documents (functions as text). CodeSense indexes documents
AND their relationships. The graph retriever is what makes the relationships
queryable.

This is also what allows the impact analysis feature (features/impact_analysis.py)
to work: by traversing the call graph in reverse, we can determine what would
break if a given function changed — something no vector search can tell you.

HOW THE GRAPH IS STORED
────────────────────────
The call graph is built once during ingestion by parsing/call_graph_builder.py
and stored as a networkx DiGraph. In production, this graph is serialized to
disk (pickle or GraphML format) and loaded at application startup.

Each node in the graph has the following attributes (stored as node metadata):
    file_path       : Relative path from repo root
    function_name   : Name of the function
    language        : Programming language
    start_line      : First line of the function in the file
    end_line        : Last line of the function in the file
    complexity      : Cyclomatic complexity
    num_callers     : Number of functions that call this one (in-degree)
    num_callees     : Number of functions this one calls (out-degree)

Edges have one attribute:
    call_site_line  : The line in the caller where the call appears
                      (useful for showing the user exactly where a call happens)

NODE IDENTIFIER FORMAT
───────────────────────
Nodes are identified by the string: "file_path::function_name"
Example: "src/auth/jwt.py::validate_token"

This format is used consistently across:
    - The call graph (node keys)
    - RetrievedChunk.citation (for UI linking)
    - The impact analysis output

When a user query names a function (e.g., "what calls authenticate?"), we do
a fuzzy lookup against all node identifiers to find the best match before
traversing.

DEPENDENCIES
────────────
    networkx        — Graph data structure and BFS/DFS/shortest-path algorithms
    loguru          — Structured logging
    config          — settings (for graph file path, traversal depth limits)
"""

import os
import pickle
import threading
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
from loguru import logger

from config import settings


# ─────────────────────────────────────────────────────────────────────────────
# Result Schemas
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GraphNode:
    """
    Represents a single function/method node returned by graph traversal.

    This is the graph-retrieval analog of RetrievedChunk in semantic_retriever.py.
    Both flow into the same reranker and generator, but GraphNode is richer in
    structural metadata (relationship_type, distance, call_site_line) that is
    not present in vector search results.

    Fields
    ──────
    node_id         : The canonical "file_path::function_name" identifier.
                      Used to cross-reference with Qdrant chunks for code snippet lookup.
    file_path       : Relative file path from repo root.
    function_name   : Name of the function.
    language        : Programming language.
    start_line      : First line of the function.
    end_line        : Last line of the function.
    complexity      : Cyclomatic complexity. Higher = harder to change safely.
    relationship_type: How this node relates to the query target.
                      "direct_caller"   — directly calls the target function
                      "indirect_caller" — calls a function that calls the target
                      "direct_callee"   — directly called by the target function
                      "indirect_callee" — called by a function the target calls
                      "same_file"       — in the same file as the target
                      "shortest_path"   — on the shortest call path between two targets
    distance        : Number of hops from the query target in the call graph.
                      0 = the target itself, 1 = direct caller/callee, 2 = two hops, etc.
    call_site_line  : Line in the caller where the call appears (for direct relationships).
                      None for indirect relationships.
    num_callers     : In-degree of this node (how many functions call it).
    num_callees     : Out-degree of this node (how many functions it calls).
    """

    node_id: str
    file_path: str
    function_name: str
    language: str
    start_line: int
    end_line: int
    complexity: int
    relationship_type: str
    distance: int
    call_site_line: Optional[int] = None
    num_callers: int = 0
    num_callees: int = 0

    @property
    def citation(self) -> str:
        """
        Human-readable citation, matching the format used by RetrievedChunk.

        The generator.py and the frontend expect the same citation format from
        both semantic and graph results, so they can be rendered identically.
        """
        return f"{self.file_path}::{self.function_name} [lines {self.start_line}–{self.end_line}]"

    @property
    def is_high_risk(self) -> bool:
        """
        True if this node represents a high-risk function to change.

        A function is high-risk for impact analysis if it:
            - Has many callers (changes ripple widely), OR
            - Has high complexity (more ways the change can go wrong)

        Used by features/impact_analysis.py to assign risk levels to blast
        radius results. Thresholds are intentionally simple — the point is
        to give the user a rough signal, not a precise risk model.
        """
        return self.num_callers >= 5 or self.complexity >= 10


@dataclass
class GraphSearchResult:
    """
    Container for the output of a graph traversal operation.

    Analogous to SemanticSearchResult, but for graph queries. Carries both
    the results and metadata about the traversal (how deep, how many nodes,
    which strategy was used).

    Fields
    ──────
    nodes           : Retrieved graph nodes, ordered by distance ascending
                      (closest to the query target first).
    query_node_id   : The canonical node_id of the function the user asked about.
    traversal_type  : "callers" | "callees" | "impact" | "path" | "file"
    max_depth       : The maximum depth limit that was applied.
    nodes_visited   : Total graph nodes visited during traversal (for performance tracking).
    traversal_ms    : Time taken for the traversal (milliseconds).
    """

    nodes: list[GraphNode] = field(default_factory=list)
    query_node_id: Optional[str] = None
    traversal_type: str = "unknown"
    max_depth: int = 0
    nodes_visited: int = 0
    traversal_ms: float = 0.0

    @property
    def is_empty(self) -> bool:
        return len(self.nodes) == 0

    @property
    def direct_relationships(self) -> list[GraphNode]:
        """Returns only nodes with distance=1 (direct callers or callees)."""
        return [n for n in self.nodes if n.distance == 1]

    @property
    def by_distance(self) -> dict[int, list[GraphNode]]:
        """
        Group nodes by their hop distance from the query target.

        Useful for the impact analysis view, which shows results in rings:
            Distance 1: "These functions directly call X"
            Distance 2: "These functions call functions that call X"
            etc.

        Returns a dict keyed by distance integer.
        """
        result: dict[int, list[GraphNode]] = {}
        for node in self.nodes:
            result.setdefault(node.distance, []).append(node)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# GraphRetriever Class
# ─────────────────────────────────────────────────────────────────────────────

class GraphRetriever:
    """
    Traverses the codebase call graph to answer structural/relational queries.

    This class is the second retrieval strategy in the hybrid system.
    It is invoked by hybrid_retriever.py for QueryType.RELATIONAL queries
    and by features/impact_analysis.py for blast-radius computations.

    The class wraps a networkx DiGraph and provides high-level traversal methods
    (get_callers, get_callees, get_impact, get_path, get_file_nodes) that the
    rest of the codebase calls without needing to know networkx internals.

    THREAD SAFETY:
    networkx DiGraph read operations (BFS, DFS, shortest_path) are thread-safe.
    Write operations (adding nodes/edges during re-ingestion) are NOT. During
    re-ingestion, the graph is rebuilt as a new object and then atomically
    swapped in — never mutated in place.

    USAGE EXAMPLE (from hybrid_retriever.py):
        retriever = GraphRetriever()
        retriever.load_graph("codesense_tiangolo_fastapi")

        result = retriever.get_callers(
            function_query="authenticate",
            collection_name="codesense_tiangolo_fastapi",
            max_depth=3
        )
        for node in result.nodes:
            print(node.citation, f"(distance={node.distance})")
    """

    def __init__(self) -> None:
        """
        Initialize with an empty graph.

        The graph is not loaded here — it is loaded explicitly via load_graph()
        after construction. This allows the same GraphRetriever instance to serve
        multiple repositories by swapping graphs between requests, and makes
        testing easier (tests can inject a mock graph without disk I/O).
        """
        # Loaded graphs keyed by collection name, alongside the pickle's mtime
        # so a re-ingest (which rewrites call_graph.pkl) is picked up without
        # restarting the API process.
        self._cache: dict[str, tuple[float, nx.DiGraph]] = {}
        self._cache_lock = threading.Lock()

        # The graph the current thread is working on. Module-level instances
        # of this class are shared by concurrent requests (each running in its
        # own worker thread via asyncio.to_thread), so a single shared "active
        # graph" slot would let a query on repo A traverse repo B's graph if
        # another request loaded B in between. Thread-local avoids that.
        self._local = threading.local()

        logger.debug("GraphRetriever initialized (no graph loaded yet).")

    @property
    def _graph(self) -> Optional[nx.DiGraph]:
        return getattr(self._local, "graph", None)

    @_graph.setter
    def _graph(self, graph: Optional[nx.DiGraph]) -> None:
        self._local.graph = graph

    @property
    def _loaded_collection(self) -> Optional[str]:
        return getattr(self._local, "collection", None)

    @_loaded_collection.setter
    def _loaded_collection(self, collection_name: Optional[str]) -> None:
        self._local.collection = collection_name

    def load_graph(self, collection_name: str) -> bool:
        """
        Load the pre-built call graph for a given repository from disk.

        WHY LOAD FROM DISK (NOT BUILD ON THE FLY):
        Building the call graph requires parsing the entire repository with
        tree-sitter, which takes seconds to minutes depending on repo size.
        We build it once during ingestion and persist it. At query time, we
        just load the pre-built graph.

        Graph files are stored at:
            {repo_clone_dir}/{collection_name}/call_graph.pkl

        The .pkl format (Python pickle) is used because networkx can serialize
        DiGraph objects directly with pickle.dump/load, and DiGraphs contain
        complex Python objects (node attribute dicts) that other formats
        (GraphML, GEXF) don't serialize cleanly.

        SECURITY NOTE: Never load pickle files from untrusted sources. Here,
        the graph files are always written by our own ingestion pipeline running
        in our own infrastructure, so this is safe.

        Parameters
        ──────────
        collection_name : str
            The Qdrant collection name (also used as the graph file directory).

        Returns
        ───────
        bool
            True if the graph was loaded successfully. False if the file doesn't
            exist (repo not ingested yet) or the file is corrupted.
        """
        graph_path = os.path.join(
            settings.repo_clone_dir,
            collection_name,
            "call_graph.pkl"
        )

        if not os.path.exists(graph_path):
            logger.warning(
                f"Call graph not found at '{graph_path}'. "
                f"Repository may not have been ingested yet."
            )
            self._graph = None
            self._loaded_collection = None
            return False

        # Reuse the cached graph unless the file has been rewritten since.
        mtime = os.path.getmtime(graph_path)
        with self._cache_lock:
            cached = self._cache.get(collection_name)
        if cached is not None and cached[0] == mtime:
            self._graph = cached[1]
            self._loaded_collection = collection_name
            return True

        try:
            with open(graph_path, "rb") as f:
                graph = pickle.load(f)

            with self._cache_lock:
                self._cache[collection_name] = (mtime, graph)
            self._graph = graph
            self._loaded_collection = collection_name

            logger.info(
                f"Call graph loaded | collection={collection_name} | "
                f"nodes={self._graph.number_of_nodes()} | "
                f"edges={self._graph.number_of_edges()}"
            )
            return True

        except (pickle.UnpicklingError, EOFError, AttributeError) as e:
            logger.error(f"Failed to load call graph from '{graph_path}' | error={e}")
            self._graph = None
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Public Traversal Methods
    # ─────────────────────────────────────────────────────────────────────────

    def get_callers(
        self,
        function_query: str,
        collection_name: str,
        max_depth: int = 3
    ) -> GraphSearchResult:
        """
        Find all functions that call the specified function, up to max_depth hops.

        WHY THIS METHOD:
        "What calls X?" is the most common relational query. It requires traversing
        the call graph in *reverse* — following edges backwards from X to its callers,
        then from those callers to their callers, etc.

        networkx's reverse_bfs (or equivalently, BFS on the reversed graph) handles
        this naturally. We use BFS (breadth-first search) rather than DFS because
        BFS gives us distance information for free — each BFS level is one hop further.

        USE CASES:
            - "What calls authenticate()?" → find all callers
            - "What would break if I changed validate_token()?" → same traversal,
              used by features/impact_analysis.py with a larger max_depth

        Parameters
        ──────────
        function_query : str
            A function name or partial name (e.g., "authenticate", "auth").
            We fuzzy-match this against all node IDs to find the target node.

        collection_name : str
            The repository collection to use. Triggers graph load if not loaded.

        max_depth : int
            Maximum number of hops to traverse. Default 3.
            Depth 1 = direct callers only.
            Depth 2 = callers of callers (2 hops).
            Depth 3 = 3 hops (covers most practical impact zones).
            Larger values can cause exponential blowup on highly connected graphs.

        Returns
        ───────
        GraphSearchResult
            Nodes that call the target, ordered by distance ascending.
            Empty if the function is not found or has no callers.
        """
        import time
        t0 = time.perf_counter()

        if not self._ensure_graph_loaded(collection_name):
            return GraphSearchResult(traversal_type="callers")

        # Find the target node by fuzzy matching the query against all node IDs.
        target_node_id = self._resolve_node(function_query)
        if target_node_id is None:
            logger.warning(
                f"get_callers: No node found matching '{function_query}' "
                f"in collection '{collection_name}'"
            )
            return GraphSearchResult(traversal_type="callers")

        logger.info(
            f"get_callers | target='{target_node_id}' | max_depth={max_depth}"
        )

        # BFS on the *reversed* graph gives us callers at each distance level.
        # nx.reverse_view() creates a view of the graph with all edges flipped —
        # if A→B in the original, then B→A in the reversed view. BFS from our
        # target node in this reversed view gives us all callers.
        reversed_graph = nx.reverse_view(self._graph)

        nodes, visited = self._bfs_with_depth(
            graph=reversed_graph,
            start_node=target_node_id,
            max_depth=max_depth,
            relationship_type_template=lambda depth: (
                "direct_caller" if depth == 1 else "indirect_caller"
            )
        )

        traversal_ms = (time.perf_counter() - t0) * 1000

        return GraphSearchResult(
            nodes=nodes,
            query_node_id=target_node_id,
            traversal_type="callers",
            max_depth=max_depth,
            nodes_visited=visited,
            traversal_ms=traversal_ms
        )

    def get_callees(
        self,
        function_query: str,
        collection_name: str,
        max_depth: int = 2
    ) -> GraphSearchResult:
        """
        Find all functions that the specified function calls, up to max_depth hops.

        WHY THIS METHOD:
        The forward direction: "what does X depend on?" This traverses edges
        *forward* from the target (the default BFS direction).

        Useful for queries like:
            - "What does the payment function depend on?"
            - "What does initialize() call?"
            - "Show me the full call chain under main()."

        max_depth defaults to 2 here (vs 3 for get_callers) because forward
        traversal (callees) tends to fan out more aggressively — a single function
        might call 10 others, each of which calls 10 more — reaching exponential
        blowup faster than reverse traversal.

        Parameters
        ──────────
        function_query  : Partial or full function name to find.
        collection_name : Repository collection name.
        max_depth       : Maximum forward hops. Default 2.

        Returns
        ───────
        GraphSearchResult
            Nodes that are called by the target, ordered by distance ascending.
        """
        import time
        t0 = time.perf_counter()

        if not self._ensure_graph_loaded(collection_name):
            return GraphSearchResult(traversal_type="callees")

        target_node_id = self._resolve_node(function_query)
        if target_node_id is None:
            logger.warning(
                f"get_callees: No node found matching '{function_query}'"
            )
            return GraphSearchResult(traversal_type="callees")

        logger.info(
            f"get_callees | target='{target_node_id}' | max_depth={max_depth}"
        )

        # Forward BFS — standard direction, no graph reversal needed.
        nodes, visited = self._bfs_with_depth(
            graph=self._graph,
            start_node=target_node_id,
            max_depth=max_depth,
            relationship_type_template=lambda depth: (
                "direct_callee" if depth == 1 else "indirect_callee"
            )
        )

        traversal_ms = (time.perf_counter() - t0) * 1000

        return GraphSearchResult(
            nodes=nodes,
            query_node_id=target_node_id,
            traversal_type="callees",
            max_depth=max_depth,
            nodes_visited=visited,
            traversal_ms=traversal_ms
        )

    def get_impact(
        self,
        function_query: str,
        collection_name: str,
        max_depth: int = 4
    ) -> GraphSearchResult:
        """
        Compute the "blast radius" of changing a given function.

        WHY THIS METHOD:
        Impact analysis is the reverse-caller traversal taken further, with
        additional risk scoring. It answers: "If I change X, what might break?"

        The blast radius is defined as all functions that transitively depend
        on X — i.e., all callers of X, callers of callers, etc.

        This is architecturally the same as get_callers() but with:
            - A larger default max_depth (4 instead of 3) to capture wider impact
            - Risk scoring on each returned node (is_high_risk property on GraphNode)
            - Used specifically by features/impact_analysis.py, not by the
              general query routing path

        The result is what powers the "blast radius" visualization in the frontend
        (frontend/src/components/ImpactGraph.jsx), which shows the call graph
        as a visual network of risk.

        Parameters
        ──────────
        function_query  : Function to analyze.
        collection_name : Repository collection.
        max_depth       : How many hops to traverse. Default 4 (wider than get_callers).

        Returns
        ───────
        GraphSearchResult with traversal_type="impact".
            Nodes sorted by distance. Each node has is_high_risk computed from
            its num_callers and complexity attributes.
        """
        import time
        t0 = time.perf_counter()

        if not self._ensure_graph_loaded(collection_name):
            return GraphSearchResult(traversal_type="impact")

        target_node_id = self._resolve_node(function_query)
        if target_node_id is None:
            logger.warning(f"get_impact: No node found matching '{function_query}'")
            return GraphSearchResult(traversal_type="impact")

        logger.info(
            f"get_impact | target='{target_node_id}' | max_depth={max_depth}"
        )

        reversed_graph = nx.reverse_view(self._graph)

        nodes, visited = self._bfs_with_depth(
            graph=reversed_graph,
            start_node=target_node_id,
            max_depth=max_depth,
            relationship_type_template=lambda depth: (
                "direct_caller" if depth == 1 else "indirect_caller"
            )
        )

        traversal_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            f"Impact analysis complete | target='{target_node_id}' | "
            f"blast_radius={len(nodes)} nodes | high_risk={sum(1 for n in nodes if n.is_high_risk)}"
        )

        return GraphSearchResult(
            nodes=nodes,
            query_node_id=target_node_id,
            traversal_type="impact",
            max_depth=max_depth,
            nodes_visited=visited,
            traversal_ms=traversal_ms
        )

    def get_path(
        self,
        source_query: str,
        target_query: str,
        collection_name: str
    ) -> GraphSearchResult:
        """
        Find the shortest call path between two functions.

        WHY THIS METHOD:
        Sometimes the user wants to understand how two parts of the code
        are connected: "How does the HTTP handler reach the database?"

        networkx's nx.shortest_path() solves this efficiently using Dijkstra's
        algorithm (or BFS for unweighted graphs). All edges in our call graph
        are unweighted (a call is a call), so BFS gives the shortest path.

        If no path exists between the two functions, we return an empty result.
        This itself is useful information — it means the two functions are in
        disconnected components of the call graph (e.g., different microservices
        that don't share code).

        Parameters
        ──────────
        source_query    : The starting function (e.g., "main", "handle_request")
        target_query    : The ending function (e.g., "execute_query", "db_connect")
        collection_name : Repository collection.

        Returns
        ───────
        GraphSearchResult with traversal_type="path".
            Nodes are the functions on the shortest path from source to target,
            ordered from source to target. Distance = position in the path.
        """
        import time
        t0 = time.perf_counter()

        if not self._ensure_graph_loaded(collection_name):
            return GraphSearchResult(traversal_type="path")

        source_id = self._resolve_node(source_query)
        target_id = self._resolve_node(target_query)

        if source_id is None or target_id is None:
            logger.warning(
                f"get_path: Could not resolve one or both functions | "
                f"source='{source_query}' ({source_id}) | target='{target_query}' ({target_id})"
            )
            return GraphSearchResult(traversal_type="path")

        try:
            # nx.shortest_path raises NetworkXNoPath if the nodes are not connected.
            path_node_ids = nx.shortest_path(self._graph, source=source_id, target=target_id)
        except nx.NetworkXNoPath:
            logger.info(
                f"No path found between '{source_id}' and '{target_id}' "
                f"— disconnected components"
            )
            return GraphSearchResult(
                query_node_id=source_id,
                traversal_type="path"
            )
        except nx.NodeNotFound as e:
            logger.error(f"get_path NodeNotFound | error={e}")
            return GraphSearchResult(traversal_type="path")

        # Map path node IDs to GraphNode objects, with distance = position in path.
        nodes = []
        for distance, node_id in enumerate(path_node_ids):
            node = self._node_id_to_graph_node(
                node_id=node_id,
                relationship_type="shortest_path",
                distance=distance
            )
            if node:
                nodes.append(node)

        traversal_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            f"Path found | '{source_id}' → '{target_id}' | "
            f"length={len(nodes)} hops | time={traversal_ms:.1f}ms"
        )

        return GraphSearchResult(
            nodes=nodes,
            query_node_id=source_id,
            traversal_type="path",
            max_depth=len(nodes),
            nodes_visited=len(nodes),
            traversal_ms=traversal_ms
        )

    def get_file_nodes(
        self, file_path: str, collection_name: str
    ) -> GraphSearchResult:
        """
        Return all function nodes that belong to a given file.

        WHY THIS METHOD:
        For SUMMARIZATION queries about a specific file or module, we need to
        retrieve all functions in that file so the generator can synthesize an
        explanation of the whole file. Vector search works for this, but graph
        retrieval is faster and more precise for an exact file path match.

        Also used by the frontend's file tree (frontend/src/components/FileTree.jsx)
        to show which functions are indexed for each file.

        Parameters
        ──────────
        file_path       : Relative file path (e.g., "src/auth/jwt.py").
                          Partial paths are supported (prefix match).
        collection_name : Repository collection.

        Returns
        ───────
        GraphSearchResult with traversal_type="file".
            All function nodes in the specified file, sorted by start_line.
        """
        import time
        t0 = time.perf_counter()

        if not self._ensure_graph_loaded(collection_name):
            return GraphSearchResult(traversal_type="file")

        # Filter all graph nodes by file_path prefix.
        matching_nodes = []
        for node_id, attrs in self._graph.nodes(data=True):
            node_file = attrs.get("file_path", "")
            if node_file.startswith(file_path) or file_path in node_file:
                graph_node = self._node_id_to_graph_node(
                    node_id=node_id,
                    relationship_type="same_file",
                    distance=0
                )
                if graph_node:
                    matching_nodes.append(graph_node)

        # Sort by start_line so the generator sees functions in source order.
        matching_nodes.sort(key=lambda n: n.start_line)

        traversal_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            f"get_file_nodes | file='{file_path}' | "
            f"found={len(matching_nodes)} nodes | time={traversal_ms:.1f}ms"
        )

        return GraphSearchResult(
            nodes=matching_nodes,
            traversal_type="file",
            nodes_visited=self._graph.number_of_nodes(),
            traversal_ms=traversal_ms
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Private Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _ensure_graph_loaded(self, collection_name: str) -> bool:
        """
        Load the graph if not already loaded. Returns False if load fails.

        Called at the start of every public method so callers don't need to
        remember to call load_graph() manually. If the graph is already loaded
        for this collection, load_graph() returns immediately from cache.
        """
        return self.load_graph(collection_name)

    def _resolve_node(self, function_query: str) -> Optional[str]:
        """
        Find the best-matching node ID for a function name query.

        WHY FUZZY MATCHING:
        Users don't always type exact function names. "authenticate" should
        match "src/auth/jwt.py::authenticate_user". We use a simple ranking:
            1. Exact full match (file_path::function_name)
            2. Exact function name match (ignoring file path)
            3. Function name starts with the query (prefix match)
            4. Query appears anywhere in the node ID (substring match)

        We return the first match found in priority order. If multiple nodes
        match at the same priority level, we return the one with the highest
        in-degree (most callers) — this tends to be the "main" version of a
        function when there are multiple implementations.

        Parameters
        ──────────
        function_query : str
            A full node ID, function name, or partial name to search for.

        Returns
        ───────
        str or None
            The best-matching node ID, or None if no match found.
        """
        if self._graph is None:
            return None

        query_lower = function_query.lower().strip()

        # Priority 1: Exact full match (user typed the full node ID)
        if function_query.strip() in self._graph.nodes:
            return function_query.strip()
        for node_id in self._graph.nodes:
            if node_id.lower() == query_lower:
                return node_id

        candidates = {
            "exact_name": [],
            "prefix":     [],
            "substring":  [],
        }

        for node_id in self._graph.nodes:
            # Extract the name from "file::function_name" / "file::Class.method".
            # Both the qualified name and the bare method name count as exact.
            func_name = node_id.split("::")[-1].lower() if "::" in node_id else node_id.lower()
            bare_name = func_name.rsplit(".", 1)[-1]

            if query_lower in (func_name, bare_name):
                candidates["exact_name"].append(node_id)
            elif func_name.startswith(query_lower):
                candidates["prefix"].append(node_id)
            elif query_lower in node_id.lower():
                candidates["substring"].append(node_id)

        # Return from highest priority bucket that has results
        for bucket in ["exact_name", "prefix", "substring"]:
            if candidates[bucket]:
                # Among ties, prefer the node with the most callers (in-degree)
                return max(
                    candidates[bucket],
                    key=lambda nid: self._graph.in_degree(nid)
                )

        logger.debug(f"_resolve_node: No match found for '{function_query}'")
        return None

    def _bfs_with_depth(
        self,
        graph: nx.DiGraph,
        start_node: str,
        max_depth: int,
        relationship_type_template,
    ) -> tuple[list[GraphNode], int]:
        """
        Perform BFS from start_node up to max_depth hops, returning GraphNode objects.

        WHY CUSTOM BFS INSTEAD OF nx.bfs_tree():
        networkx's bfs_tree() returns a tree structure but loses depth
        information per node. We need the depth (distance) of each node to:
            - Set GraphNode.distance
            - Determine relationship_type (direct vs indirect)
            - Stop at max_depth

        This custom BFS maintains a frontier queue with (node_id, depth) pairs,
        making depth information available for every visited node.

        Parameters
        ──────────
        graph                   : The (possibly reversed) networkx DiGraph to traverse.
        start_node              : Node to start BFS from (not included in results).
        max_depth               : Maximum depth to traverse.
        relationship_type_template : A callable that takes depth (int) and returns
                                  the relationship_type string for that depth level.

        Returns
        ───────
        (list[GraphNode], int)
            Tuple of (result nodes, total nodes visited).
        """
        from collections import deque

        visited = {start_node}    # Prevent revisiting nodes (handles cycles)
        queue = deque([(start_node, 0)])  # (node_id, current_depth)
        result_nodes = []
        nodes_visited = 0

        while queue:
            current_node_id, depth = queue.popleft()
            nodes_visited += 1

            # If we've reached max_depth, don't enqueue neighbors
            # but still process the current node if depth > 0 (it's a result).
            if depth > 0:
                graph_node = self._node_id_to_graph_node(
                    node_id=current_node_id,
                    relationship_type=relationship_type_template(depth),
                    distance=depth
                )
                if graph_node:
                    result_nodes.append(graph_node)

            if depth < max_depth:
                for neighbor in graph.successors(current_node_id):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))

        # Sort by distance ascending so results read from nearest to farthest.
        result_nodes.sort(key=lambda n: n.distance)

        return result_nodes, nodes_visited

    def _node_id_to_graph_node(
        self,
        node_id: str,
        relationship_type: str,
        distance: int,
        call_site_line: Optional[int] = None
    ) -> Optional[GraphNode]:
        """
        Convert a networkx node ID + attributes into a GraphNode dataclass.

        Centralizing this mapping (like _map_results in semantic_retriever.py)
        decouples the rest of the codebase from networkx's attribute dict API.
        If the node attribute schema changes in call_graph_builder.py, only
        this method needs updating.

        Parameters
        ──────────
        node_id             : The "file_path::function_name" node identifier.
        relationship_type   : How this node relates to the query target.
        distance            : Hop distance from the query target.
        call_site_line      : Line where the call appears (for direct relationships).

        Returns
        ───────
        GraphNode or None if the node_id is not in the graph.
        """
        if self._graph is None or node_id not in self._graph.nodes:
            return None

        attrs = self._graph.nodes[node_id]

        return GraphNode(
            node_id=node_id,
            file_path=attrs.get("file_path", "unknown"),
            function_name=attrs.get("function_name", node_id.split("::")[-1]),
            language=attrs.get("language", "unknown"),
            start_line=attrs.get("start_line", 0),
            end_line=attrs.get("end_line", 0),
            complexity=attrs.get("complexity", 0),
            relationship_type=relationship_type,
            distance=distance,
            call_site_line=call_site_line,
            num_callers=self._graph.in_degree(node_id),
            num_callees=self._graph.out_degree(node_id),
        )

    @property
    def graph_stats(self) -> dict:
        """
        Summary statistics about the currently loaded graph.

        Used in the /ingest status endpoint to report graph size to the user,
        and in the README evaluation table.

        Returns an empty dict if no graph is loaded.
        """
        if self._graph is None:
            return {}

        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
            "is_dag": nx.is_directed_acyclic_graph(self._graph),
            # Average in-degree = average number of callers per function.
            # High values indicate a highly interconnected codebase.
            "avg_in_degree": (
                sum(d for _, d in self._graph.in_degree()) / self._graph.number_of_nodes()
                if self._graph.number_of_nodes() > 0 else 0.0
            ),
            "collection": self._loaded_collection,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

# Shared instance used by hybrid_retriever.py and features/impact_analysis.py:
#   from retrieval.graph_retriever import graph_retriever
#
# Note: unlike the semantic_retriever, the graph_retriever does NOT load a graph
# at import time — it loads on the first query for a specific collection.
# This is intentional: we don't know which repo the user will query at startup.

graph_retriever = GraphRetriever()
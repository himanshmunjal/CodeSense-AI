"""
embeddings/graph_embedder.py — Structural embeddings from the call graph.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY THIS FILE EXISTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Semantic embeddings (code_embedder.py) tell us WHAT a function does
based on its text. But they cannot tell us HOW a function is connected
to the rest of the codebase.

Two functions might have very different code but be structurally similar:
  - Both are called by 50+ other functions (high in-degree = central/critical)
  - Both are leaf nodes with no outgoing calls (pure utilities)
  - Both sit at the same "layer" of the system (e.g., both are data-access
    functions that sit between business logic and the database)

Structural embeddings encode this positional, relational information
into a vector. When combined with semantic embeddings at retrieval time,
they let us answer questions like:
  - "What other functions are architecturally similar to authenticate_user?"
  - "What would break if I change this function?" (impact analysis)
  - "Find all functions at the same abstraction layer as this one"

How it works:
  1. The call graph (built by parsing/call_graph_builder.py) is a
     directed graph: node = function, edge = "A calls B"
  2. node2vec runs biased random walks on this graph
  3. The walk sequences are treated like sentences in Word2Vec
  4. Each node (function) gets a low-dimensional embedding vector that
     captures its graph neighborhood

Functions that are called together or share many callers end up with
similar structural vectors — even if their code looks completely different.

This file is consumed by:
  → indexing/qdrant_client.py  (stored as second vector per chunk)
  → retrieval/hybrid_retriever.py  (used in hybrid scoring formula)
  → features/impact_analysis.py  (graph traversal uses this embedder's graph)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os

import numpy as np
import networkx as nx
from typing import Dict, Optional, List
from loguru import logger

from config import settings


# ── Constants ─────────────────────────────────────────────────────────────────

# Dimension of the structural embedding vectors produced by node2vec.
# Must match settings.structural_embedding_dim (128).
# Why 128 and not 768?
#   - Structural relationships are simpler than semantic content.
#     128 dimensions is enough to encode graph topology.
#   - Smaller dimension = faster similarity computation in Qdrant.
STRUCTURAL_DIM = 128

# node2vec hyperparameters.
# These control the nature of the random walks:

# Walk length: how many nodes each random walk visits.
# Longer walks = more global structure captured, but slower.
WALK_LENGTH = 30

# Number of random walks starting from each node.
# More walks = more stable embeddings, but more memory and time.
NUM_WALKS = 200

# p (return parameter): controls probability of returning to the previous node.
# Low p → walker is less likely to backtrack → explores more of the graph.
# p=1 is the default balanced setting.
P_PARAM = 1

# q (in-out parameter): controls BFS vs DFS bias.
# q < 1 → DFS-like → captures structural equivalence (nodes with similar roles)
# q > 1 → BFS-like → captures community membership (nodes in same module)
# q=0.5 biases toward structural equivalence — useful for detecting
# functions that play the same architectural role (e.g., "all router functions")
Q_PARAM = 0.5

# Embedding window size for the Word2Vec step inside node2vec.
# Larger window = more context from the walk is used per training step.
WINDOW_SIZE = 10

# Minimum number of nodes a graph must have to be worth embedding.
# A graph with 1–2 nodes has no structural relationships to learn from.
MIN_NODES_FOR_EMBEDDING = 3


# ── GraphEmbedder Class ───────────────────────────────────────────────────────

class GraphEmbedder:
    """
    Produces structural embeddings for all functions in a call graph
    using the node2vec algorithm.

    node2vec is a graph representation learning algorithm that:
      1. Runs biased random walks on the graph (like a drunk walk
         that prefers to keep going forward rather than turning back)
      2. Feeds the walk sequences into a Word2Vec-style skip-gram model
      3. Produces a vector for each node where nodes with similar
         graph neighborhoods have similar vectors

    This captures structural roles: a function that's called by many
    things and calls many things (hub function) will have a different
    structural vector than a leaf function, regardless of what the
    code actually does.

    Usage:
        graph = build_call_graph(...)       # from parsing/call_graph_builder.py
        embedder = GraphEmbedder()
        vectors = embedder.embed_graph(graph)
        # vectors = {"auth.login": np.array([...]), "db.connect": np.array([...])}
    """

    def __init__(self):
        """
        Initialize the GraphEmbedder.

        We don't load any model here — node2vec trains fresh on each graph
        because the graph structure is unique to each repository.
        There's no pre-trained structural model to load.
        """
        logger.info("GraphEmbedder initialized.")

    def embed_graph(
        self,
        graph: nx.DiGraph,
        node_ids: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Run node2vec on a call graph and return a vector for each node.

        This is the primary public method. It takes the full call graph
        produced by the parser and returns a dictionary mapping each
        function identifier to its structural embedding vector.

        Args:
            graph: A networkx DiGraph where:
                   - Each node is a string identifier like "module.function_name"
                   - Each edge (A → B) means "function A calls function B"
                   Built by parsing/call_graph_builder.py.

            node_ids: Optional subset of node IDs to return vectors for.
                      If None, returns vectors for all nodes.
                      Useful when only new/changed files need re-embedding.

        Returns:
            Dict mapping node_id (string) → np.ndarray of shape (STRUCTURAL_DIM,).
            Nodes not in the graph return a zero vector (see _zero_vector).

        Raises:
            ValueError: If graph is None or not a DiGraph.
        """
        if graph is None or not isinstance(graph, nx.DiGraph):
            raise ValueError("embed_graph expects a networkx DiGraph.")

        node_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()
        logger.info(
            f"Running node2vec on call graph: "
            f"{node_count} nodes, {edge_count} edges."
        )

        # Guard: if the graph is too small, node2vec won't learn anything useful.
        # Return zero vectors for all nodes instead.
        if node_count < MIN_NODES_FOR_EMBEDDING:
            logger.warning(
                f"Graph has only {node_count} nodes (minimum: {MIN_NODES_FOR_EMBEDDING}). "
                f"Returning zero vectors for all nodes."
            )
            target_nodes = node_ids or list(graph.nodes())
            return {node: self._zero_vector() for node in target_nodes}

        # node2vec works best on undirected graphs for structural equivalence.
        # We convert the DiGraph to undirected here because:
        #   - We want to capture "is near this function" not "is called by / calls"
        #   - Directionality is already captured in the call graph structure
        #     used by impact_analysis.py — we don't need to double-encode it here
        undirected = graph.to_undirected()

        # Train the node2vec model on this graph.
        model = self._train_node2vec(undirected)

        # Extract vectors for all requested nodes.
        target_nodes = node_ids or list(graph.nodes())
        embeddings = self._extract_node_vectors(model, target_nodes)

        logger.info(
            f"Graph embedding complete. "
            f"Produced vectors for {len(embeddings)} / {node_count} nodes."
        )
        return embeddings

    def get_node_vector(
        self,
        node_id: str,
        embeddings: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Safely retrieve a structural vector for a node.

        This is a safe accessor used by the hybrid retriever. If a node
        wasn't in the graph when it was embedded (e.g., new function added
        after last ingestion), we return a zero vector rather than crashing.

        A zero vector has cosine similarity of 0 with everything, which
        means the hybrid retriever will rely entirely on semantic similarity
        for that node — a safe, sensible fallback.

        Args:
            node_id: The function identifier to look up.
            embeddings: The dict returned by embed_graph().

        Returns:
            np.ndarray of shape (STRUCTURAL_DIM,). Zero vector if not found.
        """
        if node_id not in embeddings:
            logger.debug(
                f"Node '{node_id}' not found in structural embeddings. "
                f"Returning zero vector (will rely on semantic score only)."
            )
            return self._zero_vector()
        return embeddings[node_id]

    def compute_centrality_features(
        self,
        graph: nx.DiGraph
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute graph centrality metrics for each node.

        These are NOT embeddings — they're scalar features stored as
        metadata alongside the vectors in Qdrant. They're used for:
          1. Answering "what are the most critical functions?" (high centrality)
          2. Ranking impact analysis results (high PageRank = high blast radius risk)
          3. Filtering in Qdrant: "find auth functions with degree > 5"

        Metrics computed:
          - in_degree:    How many functions call this one.
                          High in-degree = widely depended upon = risky to change.
          - out_degree:   How many functions this one calls.
                          High out-degree = broad reach = potential side effects.
          - pagerank:     Random-walk-based importance score (like Google's PageRank).
                          Better than degree alone because it considers the importance
                          of callers, not just their count.
          - betweenness:  How often this node lies on the shortest path between
                          two other nodes. High betweenness = architectural bottleneck.

        Args:
            graph: The call graph DiGraph.

        Returns:
            Dict mapping node_id → {"in_degree": float, "out_degree": float,
                                     "pagerank": float, "betweenness": float}
        """
        logger.info("Computing graph centrality features...")

        # Degree centrality: normalized by (n-1) so values are in [0, 1]
        in_degree  = dict(nx.in_degree_centrality(graph))
        out_degree = dict(nx.out_degree_centrality(graph))

        # PageRank: iterative algorithm, may not converge on very sparse graphs.
        # We catch the exception and fall back to uniform scores if it fails.
        try:
            pagerank = nx.pagerank(graph, alpha=0.85, max_iter=100)
        except nx.PowerIterationFailedConvergence:
            logger.warning(
                "PageRank did not converge. Assigning uniform scores. "
                "This usually means the graph is very sparse or disconnected."
            )
            pagerank = {node: 1.0 / graph.number_of_nodes() for node in graph.nodes()}

        # Betweenness centrality is O(VE) — expensive on large graphs.
        # We use an approximation (k=100 random samples) for graphs over 500 nodes.
        node_count = graph.number_of_nodes()
        if node_count > 500:
            logger.info(
                f"Graph has {node_count} nodes. "
                f"Using approximate betweenness (k=100 samples)."
            )
            betweenness = nx.betweenness_centrality(graph, k=100, normalized=True)
        else:
            betweenness = nx.betweenness_centrality(graph, normalized=True)

        # Merge all metrics per node into one dict
        features = {}
        for node in graph.nodes():
            features[node] = {
                "in_degree":    round(in_degree.get(node, 0.0), 6),
                "out_degree":   round(out_degree.get(node, 0.0), 6),
                "pagerank":     round(pagerank.get(node, 0.0), 6),
                "betweenness":  round(betweenness.get(node, 0.0), 6),
            }

        logger.info(f"Centrality features computed for {len(features)} nodes.")
        return features

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _train_node2vec(self, graph: nx.Graph):
        """
        Train a node2vec model on the given (undirected) graph.

        node2vec is implemented in the `node2vec` Python package, which
        wraps the random walk generation and feeds it into gensim's Word2Vec.

        Why these hyperparameter choices?
          - WALK_LENGTH=30: Functions are typically 3–5 calls deep.
                            30 steps ensures we cover the local neighborhood.
          - NUM_WALKS=200:  More walks = more stable vectors. 200 gives good
                            convergence without taking forever on large graphs.
          - q=0.5:          Biases toward DFS-like structural equivalence.
                            We want "same role in the codebase" not
                            "same module cluster".
          - workers=os.cpu_count(): Use all available CPU cores for random walk
                            generation. The node2vec package (unlike sklearn)
                            takes this literally and passes it straight to
                            numpy.array_split() — it does NOT support sklearn's
                            "-1 means all cores" convention, so passing -1
                            raises "ValueError: number sections must be larger
                            than 0."

        Args:
            graph: An undirected networkx Graph.

        Returns:
            A trained node2vec Word2Vec model with .wv[node_id] → vector.
        """
        try:
            from node2vec import Node2Vec
        except ImportError:
            raise ImportError(
                "node2vec package not installed. Run: pip install node2vec"
            )

        logger.debug(
            f"Training node2vec: walk_length={WALK_LENGTH}, "
            f"num_walks={NUM_WALKS}, p={P_PARAM}, q={Q_PARAM}"
        )

        # Node2Vec first generates all random walks (this is the slow step).
        # quiet=True suppresses the noisy progress bar from the underlying gensim.
        node2vec_model = Node2Vec(
            graph,
            dimensions=STRUCTURAL_DIM,
            walk_length=WALK_LENGTH,
            num_walks=NUM_WALKS,
            p=P_PARAM,
            q=Q_PARAM,
            workers=os.cpu_count() or 1,  # node2vec needs a real positive int, not -1
            quiet=True,
        )

        # fit() trains the Word2Vec model on the generated walks.
        # window=WINDOW_SIZE: how many nodes in the walk are treated as context.
        # min_count=1: don't ignore nodes that appear rarely (small graphs have few walks).
        # sg=1: use skip-gram (vs CBOW) — better for rare nodes.
        model = node2vec_model.fit(
            window=WINDOW_SIZE,
            min_count=1,
            batch_words=4,
            sg=1,
        )

        logger.debug("node2vec training complete.")
        return model

    def _extract_node_vectors(
        self,
        model,
        node_ids: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Extract embedding vectors from a trained node2vec model.

        After training, node2vec stores vectors in model.wv (the KeyedVectors
        object from gensim). We extract the vector for each node by ID.

        Some nodes might not be in model.wv if:
          - They were isolated (no edges) and got dropped during training
          - The node ID contains characters that caused issues during walks

        These nodes get zero vectors as a safe fallback.

        Args:
            model: The trained node2vec Word2Vec model.
            node_ids: List of node identifiers to extract vectors for.

        Returns:
            Dict mapping node_id → np.ndarray of shape (STRUCTURAL_DIM,).
        """
        embeddings = {}
        missing_count = 0

        for node_id in node_ids:
            node_str = str(node_id)  # node2vec requires string node IDs
            if node_str in model.wv:
                # Copy the vector to a plain numpy array.
                # model.wv[node_str] returns a view — we copy to avoid
                # accidental mutation if the model is freed later.
                embeddings[node_id] = model.wv[node_str].copy().astype(np.float32)
            else:
                embeddings[node_id] = self._zero_vector()
                missing_count += 1

        if missing_count > 0:
            logger.warning(
                f"{missing_count} nodes had no structural vector (likely isolated nodes). "
                f"Zero vectors assigned. These will rely entirely on semantic similarity."
            )

        return embeddings

    def _zero_vector(self) -> np.ndarray:
        """
        Return a zero vector of the correct structural embedding dimension.

        Zero vectors are used as a safe neutral fallback for nodes that
        couldn't be embedded. In Qdrant's cosine similarity:
            cosine_sim(zero_vector, any_vector) = 0

        This means a zero structural vector contributes 0 to the hybrid
        score, effectively falling back to 100% semantic scoring for
        that particular chunk.

        Returns:
            np.ndarray of shape (STRUCTURAL_DIM,) filled with zeros.
        """
        return np.zeros(STRUCTURAL_DIM, dtype=np.float32)
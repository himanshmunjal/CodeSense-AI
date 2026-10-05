"""
indexing/metadata_builder.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY THIS FILE EXISTS
─────────────────────
The parsing layer (tree_sitter_parser.py, entity_extractor.py) produces raw
Python dicts describing what it found in the AST:

    {
        "type": "function",
        "name": "authenticate_user",
        "start_byte": 1204,
        "end_byte": 1893,
        ...
    }

These raw dicts are intentionally "dumb" — the parser's job is just to extract
what the AST contains. It doesn't know about the repo, the git history, the
call graph, or what Qdrant needs.

This file is the bridge. metadata_builder.py takes those raw dicts and:
  1. Enriches them with repo-level context (owner, name, URL)
  2. Converts tree-sitter's byte offsets to human-readable line numbers
  3. Computes cyclomatic complexity from the raw AST node counts
  4. Looks up call graph metrics (in_degree, out_degree) from networkx
  5. Resolves the git commit hash for cache keying
  6. Constructs a validated CodeChunk object via chunk_schema.py

By separating this enrichment step from both the parser and the indexer, we
keep each layer focused on one responsibility:
  - Parser:           reads AST, extracts raw entities
  - MetadataBuilder:  enriches raw entities into typed, validated chunks
  - Qdrant client:    stores chunks, no data transformation logic

HOW IT FITS INTO THE PIPELINE
───────────────────────────────

    parsing/entity_extractor.py
        │  yields List[Dict] — one dict per parsed entity
        │
        ▼
    indexing/metadata_builder.py   ◄── YOU ARE HERE
        │  MetadataBuilder.build_chunks_for_file(raw_entities, context)
        │  returns List[CodeChunk]
        │
        ▼
    embeddings/code_embedder.py
           receives List[CodeChunk], attaches semantic_vector to each

DESIGN DECISIONS
─────────────────
- MetadataBuilder is a class rather than a collection of free functions because
  it needs to hold repo-level context (owner, name, URL, git commit) that is
  constant for all chunks in a given ingestion run. Injecting this via __init__
  once is cleaner than passing it as an argument to every function call.

- The call graph (networkx DiGraph) is optional at chunk-build time. The graph
  is only complete after ALL files in a repo are parsed — you can't know the
  in_degree of function X until you've seen all the files that might call X.
  So in_degree and out_degree are populated in a second pass via
  enrich_with_call_graph(), called after full repo parsing is done.

- Cyclomatic complexity is computed here (not in the parser) because complexity
  is a property of the chunk as a unit, not a property of the raw AST node.
  The parser just counts decision points; this layer turns that into a proper
  McCabe complexity score.

- Line number conversion from tree-sitter's 0-based rows to 1-based lines
  happens here, not in the parser. The parser stays close to tree-sitter's
  native format; this layer speaks the language of humans (line 1, not line 0).
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
from loguru import logger

from indexing.chunk_schema import (
    ChunkType,
    CodeChunk,
    Language,
    generate_chunk_id,
)


# ── Types ─────────────────────────────────────────────────────────────────────

# A raw entity dict as produced by parsing/entity_extractor.py.
# We use Dict[str, Any] rather than a TypedDict because the parser
# may include language-specific fields that vary per language.
RawEntity = Dict[str, Any]


# ── Main Class ────────────────────────────────────────────────────────────────

class MetadataBuilder:
    """
    Converts raw parsed entities into validated, enriched CodeChunk objects.

    One MetadataBuilder instance is created per ingestion run (per repo).
    It holds all the repo-level context that every chunk in that run shares,
    so callers don't have to pass those values on every call.

    Usage:
        builder = MetadataBuilder(
            repo_url="https://github.com/tiangolo/fastapi",
            repo_owner="tiangolo",
            repo_name="fastapi",
            commit_hash="a1b2c3d4",
        )

        # For each file parsed:
        raw_entities = entity_extractor.extract(file_path)
        chunks = builder.build_chunks_for_file(
            raw_entities=raw_entities,
            file_path="fastapi/routing.py",
            language=Language.PYTHON,
        )

        # After all files parsed, enrich with call graph:
        call_graph = call_graph_builder.build()
        enriched = builder.enrich_with_call_graph(chunks, call_graph)
    """

    def __init__(
        self,
        repo_url: str,
        repo_owner: str,
        repo_name: str,
        commit_hash: Optional[str] = None,
    ) -> None:
        """
        Initializes the builder with repo-level context shared by all chunks.

        Args:
            repo_url:    Full GitHub URL. Example: "https://github.com/tiangolo/fastapi"
            repo_owner:  GitHub owner. Example: "tiangolo"
            repo_name:   Repository name. Example: "fastapi"
            commit_hash: The git commit SHA of the HEAD being ingested.
                         Used as a cache key in Redis — if the commit hash
                         matches the cached value for a file, we skip re-embedding.
                         None is acceptable on first ingestion.
        """
        self.repo_url    = repo_url
        self.repo_owner  = repo_owner
        self.repo_name   = repo_name
        self.commit_hash = commit_hash

        logger.debug(
            f"MetadataBuilder initialized for {repo_owner}/{repo_name} "
            f"at commit {commit_hash or 'unknown'}"
        )

    # ── Public API ────────────────────────────────────────────────

    def build_chunks_for_file(
        self,
        raw_entities: List[RawEntity],
        file_path: str,
        language: Language,
    ) -> List[CodeChunk]:
        """
        Converts a list of raw parsed entities from one file into CodeChunk objects.

        This is the primary method called by the ingestion pipeline for each file.
        It iterates over every entity the parser found, enriches it with
        repo context and computed metrics, and constructs a validated CodeChunk.

        Entities that fail validation are skipped with a warning rather than
        crashing the whole file — a single malformed entity should not block
        the other 50 valid functions in the same file from being indexed.

        Args:
            raw_entities: List of dicts from entity_extractor.py. Each dict
                          has at minimum: type, name, start_row, end_row,
                          source_code. May also have: signature, docstring,
                          decision_points, class_name, imports.
            file_path:    Path to the file relative to repo root.
            language:     The Language enum value for this file.

        Returns:
            List of validated CodeChunk objects, ready for embedding.
            May be shorter than raw_entities if some entities failed validation.
        """
        chunks: List[CodeChunk] = []

        for raw in raw_entities:
            try:
                chunk = self._build_single_chunk(raw, file_path, language)
                chunks.append(chunk)
            except Exception as e:
                # Log the bad entity but continue — don't let one bad entity
                # block the entire file from being indexed.
                logger.warning(
                    f"Skipping entity '{raw.get('name', 'unknown')}' "
                    f"in '{file_path}': {e}"
                )

        logger.debug(
            f"Built {len(chunks)}/{len(raw_entities)} chunks for '{file_path}'"
        )
        return chunks

    def build_module_chunk(
        self,
        file_path: str,
        language: Language,
        source_code: str,
        imports: List[str],
        docstring: Optional[str] = None,
    ) -> CodeChunk:
        """
        Builds a MODULE-level chunk representing an entire file.

        WHY MODULE CHUNKS EXIST:
        Some queries are about a whole file, not a specific function.
        "What does auth.py do?" or "Summarize the routing module" need a
        chunk that represents the file as a whole.

        A module chunk's source_code is the entire file content. Its
        embeddable_text is constructed from the file-level docstring (if any)
        and the import list, which together summarize the file's purpose and
        dependencies without requiring the model to process thousands of tokens.

        Args:
            file_path:   Path relative to repo root.
            language:    Language enum for this file.
            source_code: Full raw content of the file.
            imports:     List of module paths imported by this file.
            docstring:   Module-level docstring, if the file has one.

        Returns:
            A single CodeChunk with chunk_type=MODULE.
        """
        entity_name = Path(file_path).stem   # "routing" from "fastapi/routing.py"
        line_count  = source_code.count("\n") + 1

        chunk_id = generate_chunk_id(
            repo_url    = self.repo_url,
            file_path   = file_path,
            entity_name = entity_name,
            start_line  = 1,
        )

        return CodeChunk(
            chunk_id             = chunk_id,
            repo_url             = self.repo_url,
            repo_owner           = self.repo_owner,
            repo_name            = self.repo_name,
            file_path            = file_path,
            language             = language,
            start_line           = 1,
            end_line             = line_count,
            chunk_type           = ChunkType.MODULE,
            entity_name          = entity_name,
            class_name           = None,
            signature            = None,
            docstring            = docstring,
            source_code          = source_code,
            cyclomatic_complexity= None,   # Complexity is function-level, not module-level
            imports              = imports,
            last_modified_commit = self.commit_hash,
            ingested_at          = datetime.utcnow(),
        )

    def enrich_with_call_graph(
        self,
        chunks: List[CodeChunk],
        call_graph: nx.DiGraph,
    ) -> List[CodeChunk]:
        """
        Populates in_degree and out_degree on each chunk from the call graph.

        WHY THIS IS A SEPARATE PASS:
        The call graph is only complete after ALL files in a repo are parsed.
        If we tried to compute in_degree while building chunks file-by-file,
        we'd always get incomplete numbers — file A's function might be called
        by file B, but if we haven't parsed file B yet, we'd record in_degree=0.

        So the ingestion pipeline first builds all chunks (first pass), then
        builds the call graph over all parsed entities, then calls this method
        to enrich every chunk with accurate graph metrics (second pass).

        The call graph nodes are keyed "{file_path}::{fully_qualified_name}"
        (e.g. "fastapi/routing.py::Router.add_api_route"), not bare
        fully_qualified_name — this matches the convention
        retrieval/graph_retriever.py and tasks/celery_worker.py already use,
        which disambiguates same-named functions/methods defined in
        different files. Chunks not present in the graph (e.g. private
        helpers never called by other tracked code) simply keep their
        default None values.

        Args:
            chunks:     The list of all CodeChunk objects built in the first pass.
            call_graph: A networkx DiGraph where nodes are
                        "{file_path}::{fully_qualified_name}" and edges are
                        (caller, callee) relationships.

        Returns:
            The same list of chunks, mutated in-place with in_degree and
            out_degree populated. Returns the list for chaining convenience.
        """
        enriched_count = 0

        for chunk in chunks:
            node_key = f"{chunk.file_path}::{chunk.fully_qualified_name}"

            if call_graph.has_node(node_key):
                # in_degree: how many other functions call this one
                # Pydantic v2 allows mutation via model_copy or direct assignment
                # when model_config allows it. We use object.__setattr__ to
                # bypass Pydantic's immutability for this controlled update.
                object.__setattr__(chunk, "in_degree",  call_graph.in_degree(node_key))
                object.__setattr__(chunk, "out_degree", call_graph.out_degree(node_key))
                enriched_count += 1

        logger.info(
            f"Enriched {enriched_count}/{len(chunks)} chunks with call graph metrics "
            f"({call_graph.number_of_nodes()} nodes, {call_graph.number_of_edges()} edges)"
        )
        return chunks

    # ── Private Helpers ───────────────────────────────────────────

    def _build_single_chunk(
        self,
        raw: RawEntity,
        file_path: str,
        language: Language,
    ) -> CodeChunk:
        """
        Builds one CodeChunk from one raw entity dict.

        This is the core transformation. Each step is commented to explain
        why that transformation is necessary.

        Args:
            raw:       Raw entity dict from entity_extractor.py.
            file_path: File path relative to repo root.
            language:  Language enum for this file.

        Returns:
            A validated CodeChunk.

        Raises:
            ValueError: If required fields are missing or the chunk fails
                        Pydantic validation. Caller catches and logs this.
        """
        # Step 1: Resolve the entity type.
        # tree-sitter reports raw types like "function_definition" or
        # "method_declaration". We normalize these to our ChunkType enum.
        chunk_type = self._resolve_chunk_type(
            raw.get("type", ""), language, raw.get("class_name")
        )

        # Step 2: Convert tree-sitter's 0-based row indices to 1-based line numbers.
        # tree-sitter uses 0-indexed rows (row 0 = first line). Humans and every
        # code editor in existence use 1-indexed lines. Adding 1 here means all
        # downstream code — citations, source viewer, eval dataset — works in
        # the familiar 1-based coordinate system.
        start_line = raw["start_row"] + 1
        end_line   = raw["end_row"]   + 1

        # Step 3: Compute cyclomatic complexity from raw decision point count.
        # entity_extractor.py counts decision points (branches) in the AST:
        # if/elif/else/for/while/except/and/or each add 1. McCabe complexity
        # is decision_points + 1. We only compute this for function/method chunks;
        # class and module complexity is not meaningful.
        complexity = None
        if chunk_type in (ChunkType.FUNCTION, ChunkType.METHOD):
            decision_points = raw.get("decision_points", 0)
            complexity = self._compute_cyclomatic_complexity(decision_points)

        # Step 4: Generate a deterministic chunk_id.
        # Using the four fields that uniquely identify a code entity across
        # all repos. See generate_chunk_id() in chunk_schema.py for rationale.
        chunk_id = generate_chunk_id(
            repo_url    = self.repo_url,
            file_path   = file_path,
            entity_name = raw["name"],
            start_line  = start_line,
        )

        # Step 5: Clean the docstring.
        # Docstrings from the AST may have inconsistent indentation
        # (the parser captures the raw string including leading whitespace).
        # We normalize it before storing so the embedding model sees clean text.
        docstring = self._clean_docstring(raw.get("docstring"))

        # Step 6: Construct and return the validated CodeChunk.
        # Pydantic runs __init__ validation here, including the model_validators
        # defined in chunk_schema.py (line range check, method-has-class check).
        return CodeChunk(
            chunk_id             = chunk_id,
            repo_url             = self.repo_url,
            repo_owner           = self.repo_owner,
            repo_name            = self.repo_name,
            file_path            = file_path,
            language             = language,
            start_line           = start_line,
            end_line             = end_line,
            chunk_type           = chunk_type,
            entity_name          = raw["name"],
            class_name           = raw.get("class_name"),
            signature            = raw.get("signature"),
            docstring            = docstring,
            source_code          = raw["source_code"],
            cyclomatic_complexity= complexity,
            imports              = raw.get("imports", []),
            last_modified_commit = self.commit_hash,
            ingested_at          = datetime.utcnow(),
        )

    def _resolve_chunk_type(
        self, raw_type: str, language: Language, class_name: Optional[str] = None
    ) -> ChunkType:
        """
        Maps a tree-sitter node type string to our ChunkType enum.

        WHY THIS IS NEEDED:
        tree-sitter uses language-specific node type names. Python calls a
        function "function_definition"; Java calls it "method_declaration";
        TypeScript has both "function_declaration" and "arrow_function".
        This method abstracts all of that into our four canonical types.

        We use a flat lookup dict rather than if/elif chains because it is
        easier to read and extend when adding new language support.

        Args:
            raw_type: The tree-sitter node type string from entity_extractor.py.
            language: The language of the source file (used for disambiguation).
            class_name: Enclosing class, if any — upgrades FUNCTION to METHOD.

        Returns:
            The matching ChunkType enum value.

        Raises:
            ValueError: If raw_type is not recognized. This propagates up to
                        build_chunks_for_file which logs and skips the entity.
        """
        # Maps tree-sitter node type → ChunkType, covering all four supported languages.
        # Some types (like "method_definition") map to METHOD in all languages,
        # while others (like "function_definition") mean FUNCTION in Python
        # but METHOD in Java (where all functions are methods).
        TYPE_MAP: Dict[str, ChunkType] = {
            # Python
            "function_definition":         ChunkType.FUNCTION,
            "async_function_definition":   ChunkType.FUNCTION,
            "class_definition":            ChunkType.CLASS,
            # JavaScript / TypeScript
            "function_declaration":        ChunkType.FUNCTION,
            "arrow_function":              ChunkType.FUNCTION,
            "function_expression":         ChunkType.FUNCTION,
            "method_definition":           ChunkType.METHOD,
            "class_declaration":           ChunkType.CLASS,
            "abstract_class_declaration":  ChunkType.CLASS,
            # Java
            "method_declaration":          ChunkType.METHOD,
            "constructor_declaration":     ChunkType.METHOD,
            "class_declaration_java":      ChunkType.CLASS,
            "interface_declaration":       ChunkType.CLASS,
            # Module-level (set explicitly by entity_extractor, not from AST node type)
            "module":                      ChunkType.MODULE,
        }

        # Special case: Java top-level "method_declaration" is a METHOD,
        # but Python "function_definition" inside a class body is also a METHOD.
        # entity_extractor.py sets "class_name" when a function is nested inside
        # a class, so we check for that here to upgrade FUNCTION → METHOD.
        #
        # We do this AFTER the lookup so the TYPE_MAP stays clean and this
        # override logic is isolated and easy to find.
        chunk_type = TYPE_MAP.get(raw_type)

        if chunk_type is None:
            raise ValueError(
                f"Unrecognized tree-sitter node type '{raw_type}' "
                f"for language '{language.value}'. "
                f"Add it to MetadataBuilder._resolve_chunk_type() if valid."
            )

        if chunk_type == ChunkType.FUNCTION and class_name:
            return ChunkType.METHOD
        return chunk_type

    def _compute_cyclomatic_complexity(self, decision_points: int) -> int:
        """
        Computes McCabe cyclomatic complexity from a raw decision point count.

        FORMULA: M = decision_points + 1

        WHY +1:
        McCabe's formula is M = E - N + 2P where E=edges, N=nodes, P=connected
        components in the control flow graph. For a single function (P=1), this
        simplifies to: M = (number of branching points) + 1.

        The minimum complexity is 1 (a straight-line function with no branches).
        A function with one `if` statement has complexity 2 (two paths through it).

        COMPLEXITY INTERPRETATION (standard thresholds):
            1–5:   Simple, low risk
            6–10:  Moderately complex, acceptable
            11–20: Complex, consider refactoring
            21+:   Very complex, high defect risk — flag in the UI

        Args:
            decision_points: Count of branching AST nodes in the function body.
                             entity_extractor.py counts: if, elif, else, for, while,
                             except, ExceptHandler, and, or, conditional expressions.

        Returns:
            Integer McCabe complexity, minimum 1.
        """
        return max(1, decision_points + 1)

    def _clean_docstring(self, raw_docstring: Optional[str]) -> Optional[str]:
        """
        Normalizes a raw docstring extracted from the AST.

        WHY THIS IS NEEDED:
        tree-sitter extracts docstrings as raw string literals, including:
        - The surrounding triple quotes (''' or \""")
        - Inconsistent indentation (the parser preserves the original indentation)
        - Leading/trailing blank lines

        The embedding model doesn't benefit from triple quotes or extra whitespace.
        Cleaned docstrings produce better embeddings and more readable output
        in the source viewer and generated answers.

        WHAT WE DO:
        1. Strip surrounding triple-quote delimiters (both ''' and \""" variants)
        2. Use textwrap.dedent to remove common leading whitespace from all lines
        3. Strip leading/trailing blank lines
        4. Return None if the result is empty (no point storing an empty string)

        Args:
            raw_docstring: The raw docstring string from the AST, or None.

        Returns:
            Cleaned docstring string, or None if input was None or empty.
        """
        if not raw_docstring:
            return None

        import textwrap

        cleaned = raw_docstring.strip()

        # Remove triple-quote delimiters ("""...""" and '''...''')
        # We use regex to handle both styles and optional r/f/b prefixes.
        cleaned = re.sub(r'^[rRfFbBuU]*["\']{{3}}|["\']{{3}}$', "", cleaned)

        # Remove Python single-line docstring quotes too (edge case)
        cleaned = re.sub(r'^["\']|["\']$', "", cleaned)

        # Normalize indentation — removes common leading whitespace from all lines
        cleaned = textwrap.dedent(cleaned).strip()

        # Return None for empty results (e.g., a docstring that was just whitespace)
        return cleaned if cleaned else None


# ── Standalone Utility Functions ──────────────────────────────────────────────
# These functions are used outside the class — e.g., in the evaluation layer
# and in tests — so they live at module level rather than as static methods.

def infer_language_from_path(file_path: str) -> Language:
    """
    Infers the Language enum value from a file's extension.

    WHY THIS EXISTS AS A STANDALONE FUNCTION:
    The MetadataBuilder receives a Language that the file_walker already
    resolved via config.language_for_file(). But other parts of the codebase
    (evaluation scripts, test fixtures) need to resolve language from a path
    without instantiating a full MetadataBuilder. Putting this logic here
    avoids duplicating it.

    Args:
        file_path: Any file path string. Can be relative or absolute.

    Returns:
        The matching Language enum value, or Language.UNKNOWN if not recognized.

    Examples:
        infer_language_from_path("src/auth/login.py")     → Language.PYTHON
        infer_language_from_path("components/App.tsx")    → Language.TYPESCRIPT
        infer_language_from_path("build/output.min.js")   → Language.JAVASCRIPT
        infer_language_from_path("README.md")             → Language.UNKNOWN
    """
    EXTENSION_MAP = {
        ".py":   Language.PYTHON,
        ".js":   Language.JAVASCRIPT,
        ".mjs":  Language.JAVASCRIPT,
        ".cjs":  Language.JAVASCRIPT,
        ".ts":   Language.TYPESCRIPT,
        ".tsx":  Language.TYPESCRIPT,
        ".java": Language.JAVA,
        ".go":   Language.GO,
    }
    ext = Path(file_path).suffix.lower()
    return EXTENSION_MAP.get(ext, Language.UNKNOWN)


def summarize_chunks(chunks: List[CodeChunk]) -> Dict[str, Any]:
    """
    Returns a summary dict describing a list of chunks.

    WHY THIS EXISTS:
    After building chunks for a repo, the ingestion pipeline logs a summary
    to confirm the parsing worked as expected before moving to the embedding
    step. Catching "0 functions found" here (before embedding costs money)
    is far better than finding out at query time.

    Also used in the evaluation layer to describe the composition of
    the indexed repo (how many functions, classes, etc.).

    Args:
        chunks: Any list of CodeChunk objects.

    Returns:
        A dict with counts by chunk_type, average complexity, and
        the count of chunks with missing docstrings (a code quality signal).

    Example output:
        {
            "total": 142,
            "by_type": {"function": 98, "method": 31, "class": 8, "module": 5},
            "avg_complexity": 4.2,
            "missing_docstrings": 67,
            "missing_docstring_pct": 47.2,
        }
    """
    if not chunks:
        return {"total": 0, "by_type": {}, "avg_complexity": None,
                "missing_docstrings": 0, "missing_docstring_pct": 0.0}

    by_type: Dict[str, int] = {}
    for chunk in chunks:
        key = chunk.chunk_type.value
        by_type[key] = by_type.get(key, 0) + 1

    # Compute average complexity only over function/method chunks that have it
    complexity_values = [
        c.cyclomatic_complexity
        for c in chunks
        if c.cyclomatic_complexity is not None
    ]
    avg_complexity = (
        round(sum(complexity_values) / len(complexity_values), 2)
        if complexity_values else None
    )

    missing_docstrings = sum(1 for c in chunks if not c.docstring)

    return {
        "total":                len(chunks),
        "by_type":              by_type,
        "avg_complexity":       avg_complexity,
        "missing_docstrings":   missing_docstrings,
        "missing_docstring_pct": round(missing_docstrings / len(chunks) * 100, 1),
    }
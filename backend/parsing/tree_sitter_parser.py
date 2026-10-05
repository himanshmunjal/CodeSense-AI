"""
tree_sitter_parser.py
---------------------
PURPOSE:
    This module is responsible for converting raw source file content into
    structured Abstract Syntax Trees (ASTs) using the tree-sitter library.
    It is the foundation of CodeSense's structural understanding of code —
    every downstream module (entity_extractor, call_graph_builder) depends
    on the ASTs produced here.

WHY THIS FILE EXISTS:
    Generic RAG systems treat source code as plain text and chunk it by token
    count or line breaks. This destroys semantic structure:
        - A 500-token chunk might contain half a class definition.
        - A line-split might cut a multi-line function signature in half.
        - An import statement gets separated from the function that uses it.

    tree-sitter solves this by parsing code according to its actual grammar.
    The resulting AST lets us extract functions, classes, and import statements
    as complete, meaningful units — regardless of how long they are. This is
    the core technical differentiation of CodeSense over naive RAG approaches.

WHY TREE-SITTER SPECIFICALLY:
    - Single library, multiple languages: one API for Python, JavaScript,
      TypeScript, and Java. No need to maintain separate parsers.
    - Incremental parsing: can re-parse only changed sections of a file,
      which is important for the change-detection pipeline.
    - Error recovery: tree-sitter produces a partial AST even for files with
      syntax errors, which is critical for real-world repos that may not be
      in a clean state.
    - Fast: written in C, with Python bindings. Can parse thousands of files
      per second even on modest hardware.

DESIGN DECISIONS:
    - We initialise one parser per language at module load time and cache them.
      Creating a parser is cheap; loading the language grammar is slightly more
      expensive. Caching avoids repeated grammar loading across thousands of files.
    - We return raw tree-sitter Node objects from the low-level functions and
      wrap them in a ParsedFile dataclass at the higher level. This keeps the
      module composable: entity_extractor.py works directly with the Node API.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import tree_sitter_go as tsgo
import tree_sitter_java as tsjava
import tree_sitter_javascript as tsjs
import tree_sitter_python as tspython
import tree_sitter_typescript as tsts
from loguru import logger
from tree_sitter import Language, Node, Parser, Tree

from ingestion.file_walker import SourceFile


# ---------------------------------------------------------------------------
# Language registry — maps language name → tree-sitter Language object
# ---------------------------------------------------------------------------

def _build_language_registry() -> dict[str, Language]:
    """
    Initialise and return the tree-sitter Language objects for all supported
    languages.

    WHY:
        tree-sitter requires a Language object to configure a Parser. Each
        Language object wraps the compiled grammar for one language. We build
        all four at startup so any per-file parsing call is instant — no
        grammar loading at parse time.

    WHY a module-level registry rather than per-parser instances:
        Parser objects are stateful (they hold the current parse tree for
        incremental re-parsing). Language objects are stateless and can be
        shared safely. Separating them means we can create new Parser instances
        per-file when needed without reloading grammars.

    Returns:
        Dict mapping language name (e.g. "python") to its Language instance.
    """
    return {
        "python":     Language(tspython.language()),
        "javascript": Language(tsjs.language()),
        "typescript": Language(tsts.language_typescript()),
        "java":       Language(tsjava.language()),
        "go":         Language(tsgo.language()),
    }


# Module-level language registry — built once at import time.
_LANGUAGE_REGISTRY: dict[str, Language] = _build_language_registry()


def _get_parser(language: str) -> Parser:
    """
    Create and return a configured tree-sitter Parser for the given language.

    WHY we create a new Parser per call rather than caching Parsers:
        Parser.parse() is NOT thread-safe when reusing the same Parser instance
        across threads. Since our Celery workers may process files in parallel,
        creating a fresh Parser per invocation is the safest approach.
        Parser creation is cheap (microseconds); grammar loading is where the
        cost is, and that's cached in _LANGUAGE_REGISTRY.

    Args:
        language: Language identifier — "python", "javascript", "typescript",
                  or "java".

    Returns:
        A tree-sitter Parser configured for the requested language.

    Raises:
        ValueError: If the language is not in the registry.
    """
    lang_obj = _LANGUAGE_REGISTRY.get(language)
    if lang_obj is None:
        raise ValueError(
            f"Unsupported language: '{language}'. "
            f"Available: {list(_LANGUAGE_REGISTRY.keys())}"
        )
    parser = Parser(lang_obj)
    return parser


# ---------------------------------------------------------------------------
# Data contract: ParsedFile
# ---------------------------------------------------------------------------

@dataclass
class ParsedFile:
    """
    Represents a successfully parsed source file, including its AST and
    metadata needed by downstream pipeline stages.

    WHY:
        entity_extractor.py, call_graph_builder.py, and complexity_analyzer.py
        all need both the AST (to traverse nodes) and the raw source bytes
        (to extract text from node byte-ranges). Bundling them in a dataclass
        ensures these modules receive everything they need in one object, with
        explicit field names rather than positional tuple indexing.

    Attributes:
        source_file:    The SourceFile this was parsed from. Carries path,
                        language, and size metadata.
        tree:           tree-sitter Tree object. The root of the AST.
                        Access via tree.root_node.
        source_bytes:   Raw source content as bytes. Required for extracting
                        node text via node.start_byte / node.end_byte slices.
        content_hash:   SHA-256 of the source content. Used by the change
                        detector to determine if a file needs re-embedding.
        has_errors:     True if tree-sitter detected syntax errors in the file.
                        The AST is still usable (tree-sitter recovers), but
                        callers may want to log or skip error nodes.
    """
    source_file: SourceFile
    tree: Tree
    source_bytes: bytes
    content_hash: str
    has_errors: bool


# ---------------------------------------------------------------------------
# Core parsing functions
# ---------------------------------------------------------------------------

def parse_file(source_file: SourceFile) -> Optional[ParsedFile]:
    """
    Parse a single source file and return its ParsedFile representation.

    This is the primary entry point for this module. It reads the file from
    disk, runs tree-sitter parsing, and returns a ParsedFile ready for
    entity extraction.

    WHY we read as bytes (not str):
        tree-sitter operates on raw bytes and uses byte offsets for node
        positions (start_byte, end_byte). If we decoded to a string first,
        multi-byte Unicode characters would shift all byte offsets, making
        node.start_byte / node.end_byte incorrect for text extraction.
        Always pass bytes; decode only when displaying to humans.

    Args:
        source_file: A SourceFile instance from file_walker.walk_repository().

    Returns:
        ParsedFile if parsing succeeded, None if the file could not be read
        (e.g. permission error, file deleted between walk and parse).
    """
    try:
        source_bytes = source_file.absolute_path.read_bytes()
    except OSError as exc:
        logger.warning(f"Could not read {source_file.relative_path}: {exc}")
        return None

    parser = _get_parser(source_file.language)

    try:
        tree: Tree = parser.parse(source_bytes)
    except Exception as exc:
        # tree-sitter is extremely robust — this branch should almost never
        # trigger. If it does, it indicates a grammar or version mismatch.
        logger.error(
            f"tree-sitter parse failed for {source_file.relative_path}: {exc}"
        )
        return None

    content_hash = hashlib.sha256(source_bytes).hexdigest()
    has_errors = _check_for_errors(tree.root_node)

    if has_errors:
        logger.debug(
            f"Syntax errors detected in {source_file.relative_path} "
            f"(tree-sitter will attempt partial AST extraction)."
        )

    return ParsedFile(
        source_file=source_file,
        tree=tree,
        source_bytes=source_bytes,
        content_hash=content_hash,
        has_errors=has_errors,
    )


def parse_source_string(
    source_code: str,
    language: str,
) -> Optional[Tree]:
    """
    Parse a raw source code string and return the tree-sitter Tree.

    WHY:
        Useful for unit tests, for parsing code snippets extracted from
        docstrings, and for the query-time analysis where we parse small
        code fragments the user references in their question.
        Unlike parse_file(), this does not produce a ParsedFile — it returns
        the raw Tree for callers that don't need the full dataclass.

    Args:
        source_code: Raw source code as a string.
        language:    Language identifier.

    Returns:
        tree-sitter Tree, or None if parsing failed.
    """
    source_bytes = source_code.encode("utf-8")
    parser = _get_parser(language)
    try:
        return parser.parse(source_bytes)
    except Exception as exc:
        logger.error(f"Failed to parse source string ({language}): {exc}")
        return None


def parse_files_batch(
    source_files: list[SourceFile],
    stop_on_error: bool = False,
) -> list[ParsedFile]:
    """
    Parse a list of SourceFile instances and return all successful ParsedFiles.

    WHY:
        Most of the pipeline operates on lists of files. This batch function
        handles per-file error logging and optional early termination so callers
        don't need to write the loop themselves. Failed files are logged but
        do not abort the batch by default.

    Args:
        source_files:   List of SourceFile instances to parse.
        stop_on_error:  If True, raise on the first parse failure rather than
                        skipping and continuing. Useful for debugging.

    Returns:
        List of ParsedFile instances for all files that parsed successfully.
        Files that failed are excluded (with logged warnings).
    """
    results: list[ParsedFile] = []
    failed = 0

    logger.info(f"Parsing {len(source_files)} source files...")

    for source_file in source_files:
        parsed = parse_file(source_file)
        if parsed is None:
            failed += 1
            if stop_on_error:
                raise RuntimeError(
                    f"Parsing failed for {source_file.relative_path} "
                    "and stop_on_error=True."
                )
            continue
        results.append(parsed)

    logger.info(
        f"Parsing complete: {len(results)} succeeded, {failed} failed."
    )
    return results


# ---------------------------------------------------------------------------
# AST traversal utilities
# ---------------------------------------------------------------------------

def extract_node_text(node: Node, source_bytes: bytes) -> str:
    """
    Extract the source text corresponding to a tree-sitter Node.

    WHY:
        tree-sitter nodes store byte offsets, not text. To get the actual
        source code for a node (e.g. a function body), we slice the raw
        source bytes using start_byte and end_byte, then decode to UTF-8.
        This helper centralises that operation so callers don't duplicate it.

    WHY 'replace' error handling:
        Real-world source files occasionally contain invalid UTF-8 sequences
        (e.g. a Latin-1 encoded comment in an otherwise ASCII file). 'replace'
        substitutes the replacement character (U+FFFD) rather than raising,
        which keeps the pipeline running at the cost of minor text corruption
        in edge cases. For code analysis, this is an acceptable tradeoff.

    Args:
        node:         A tree-sitter Node.
        source_bytes: The raw bytes of the file the node was parsed from.

    Returns:
        The source text for the node, decoded as UTF-8.
    """
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def iter_children_of_type(node: Node, node_type: str) -> list[Node]:
    """
    Return all immediate children of a node that match a specific node type.

    WHY:
        tree-sitter ASTs are deeply nested. When extracting entities, we
        frequently need to find specific child node types (e.g. all
        "function_definition" children of a "module" node). This avoids
        writing the same list comprehension in every extraction function.

    Args:
        node:      The parent node to search under.
        node_type: The tree-sitter node type string to match (e.g.
                   "function_definition", "class_definition", "import_statement").

    Returns:
        List of matching child nodes, in source order.
    """
    return [child for child in node.children if child.type == node_type]


def walk_tree(node: Node):
    """
    Yield every node in the AST subtree rooted at `node`, depth-first.

    WHY:
        Some extraction tasks (e.g. finding all call expressions anywhere
        in a function body) require a full recursive traversal. This generator
        provides a clean way to do that without writing recursive functions
        in every module that needs it.

    Args:
        node: Root of the subtree to traverse.

    Yields:
        Every Node in the subtree, including the root, depth-first.
    """
    yield node
    for child in node.children:
        yield from walk_tree(child)


def find_nodes_by_type(root: Node, node_type: str) -> list[Node]:
    """
    Find all nodes of a specific type anywhere in the AST subtree.

    WHY:
        Used by entity_extractor.py and call_graph_builder.py to find all
        occurrences of a construct (e.g. all "call_expression" nodes in a
        file to build the call graph). More convenient than manual recursion.

    Args:
        root:      Root node to search from.
        node_type: tree-sitter node type string to match.

    Returns:
        All matching nodes found anywhere in the subtree, in depth-first order.
    """
    return [node for node in walk_tree(root) if node.type == node_type]


def get_node_line_range(node: Node) -> tuple[int, int]:
    """
    Return the (start_line, end_line) of a node, using 1-based line numbers.

    WHY 1-based:
        tree-sitter uses 0-based line numbers internally. However, every code
        editor, GitHub URL, and error message uses 1-based line numbers. We
        convert here so all metadata stored in Qdrant is human-readable and
        directly usable in UI source links without an off-by-one correction.

    Args:
        node: Any tree-sitter Node.

    Returns:
        (start_line, end_line) tuple, both 1-based inclusive.
    """
    return (node.start_point[0] + 1, node.end_point[0] + 1)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_for_errors(node: Node) -> bool:
    """
    Recursively check whether the AST contains any ERROR or MISSING nodes.

    WHY:
        tree-sitter inserts ERROR nodes where it couldn't parse the source
        according to the grammar, and MISSING nodes where it inferred a
        missing token for error recovery. Their presence indicates the file
        has syntax errors. We surface this as ParsedFile.has_errors so that
        downstream modules can decide whether to skip or special-case these files.

    Args:
        node: Root node of the tree (typically tree.root_node).

    Returns:
        True if any ERROR or MISSING node exists in the subtree.
    """
    if node.type in ("ERROR", "MISSING") or node.is_missing:
        return True
    return any(_check_for_errors(child) for child in node.children)
"""
file_walker.py
--------------
PURPOSE:
    This module walks a locally cloned repository directory and returns a
    filtered, structured list of source files that CodeSense should parse
    and index. It is responsible for deciding WHICH files enter the pipeline.

WHY THIS FILE EXISTS:
    A cloned repository is not just source code — it also contains build
    artefacts, test fixtures, lock files, auto-generated code, binary assets,
    and vendor dependencies. Feeding all of that into the parsing and
    embedding pipeline would:
        - Waste embedding budget on files that carry no semantic value.
        - Pollute the vector index with noise (e.g. minified JS, lock files).
        - Slow down ingestion dramatically on large monorepos.

    This module applies a layered filter strategy:
        1. Extension whitelist  — only process languages we explicitly support.
        2. Path blacklist       — skip directories like node_modules, .git, dist.
        3. Size guard           — skip files above a configurable byte threshold
                                  (very large files are usually generated or minified).
        4. Binary check         — skip any file that contains null bytes.

    The output is a list of SourceFile dataclass instances — a clean, typed
    contract between the ingestion layer and the parsing layer downstream.

DESIGN DECISIONS:
    - We use os.walk() rather than pathlib.glob() because os.walk() lets us
      prune entire subtrees (e.g. skip all of node_modules/) without recursing
      into them, which is significantly faster on repos with deep vendor trees.
    - Language detection is extension-based, not content-based. This is a
      deliberate speed/accuracy tradeoff — content-based detection (e.g. via
      `python-magic`) is more accurate but adds latency for thousands of files.
      Extension-based detection is sufficient for well-structured repos.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional

from loguru import logger

from config import settings


# ---------------------------------------------------------------------------
# Supported languages and their file extensions
# ---------------------------------------------------------------------------

# These are the four languages CodeSense supports in this version.
# tree-sitter grammars exist for all four; expanding this list later
# requires adding a grammar + updating the parser module.
SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".py":   "python",
    ".js":   "javascript",
    ".jsx":  "javascript",
    ".ts":   "typescript",
    ".tsx":  "typescript",
    ".java": "java",
    ".go":   "go",
}

# ---------------------------------------------------------------------------
# Directories to skip entirely during traversal
# ---------------------------------------------------------------------------

# These directories either contain non-source files, vendored dependencies,
# or generated artefacts. Recursing into them would produce noise in the index
# and dramatically increase ingestion time on large repos.
BLACKLISTED_DIRS: frozenset[str] = frozenset({
    ".git",
    ".github",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    "out",
    ".next",
    ".nuxt",
    "target",          # Java/Rust build output
    "bin",
    "obj",
    "venv",
    ".venv",
    "env",
    ".env",
    "vendor",
    "third_party",
    "site-packages",
    "coverage",
    ".tox",
    "migrations",      # Django auto-generated migration files
    "generated",
    "auto_generated",
    "__generated__",
    "proto",           # Protocol buffer generated files
    "stubs",
})

# ---------------------------------------------------------------------------
# File name patterns to skip regardless of extension
# ---------------------------------------------------------------------------

BLACKLISTED_FILENAMES: frozenset[str] = frozenset({
    "package-lock.json",
    "yarn.lock",
    "poetry.lock",
    "Pipfile.lock",
    "composer.lock",
    "Gemfile.lock",
    ".DS_Store",
    "Thumbs.db",
})

# Default maximum file size in bytes. Files above this threshold are almost
# always minified bundles, auto-generated code, or fixture data — not worth
# parsing or indexing.
DEFAULT_MAX_FILE_SIZE_BYTES: int = 500_000  # 500 KB


# ---------------------------------------------------------------------------
# Data contract: SourceFile
# ---------------------------------------------------------------------------

@dataclass
class SourceFile:
    """
    Represents a single source file that has passed all filters and is
    ready to be handed to the parsing pipeline.

    WHY A DATACLASS:
        Using a typed dataclass as the output contract (rather than raw dicts
        or tuples) makes downstream code in tree_sitter_parser.py and
        entity_extractor.py self-documenting and easier to test. Type checkers
        can validate field access; IDEs provide autocomplete.

    Attributes:
        absolute_path:  Full filesystem path to the file. Used by tree-sitter
                        to open and read the file content.
        relative_path:  Path relative to the repo root. Stored as metadata in
                        Qdrant so query results can show "src/auth/login.py"
                        rather than a machine-specific absolute path.
        language:       Normalised language identifier — one of: "python",
                        "javascript", "typescript", "java". Used to select the
                        correct tree-sitter grammar.
        size_bytes:     File size in bytes. Stored in metadata for observability
                        and used to skip re-embedding unchanged files.
        extension:      File extension (e.g. ".py"). Kept for quick filtering
                        downstream without re-deriving from the path.
    """
    absolute_path: Path
    relative_path: str
    language: str
    size_bytes: int
    extension: str
    extra_metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core walker
# ---------------------------------------------------------------------------

def walk_repository(
    repo_root: Path,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
    include_tests: bool = True,
) -> list[SourceFile]:
    """
    Traverse a cloned repository and return all source files that should
    be parsed and indexed by CodeSense.

    This is the main entry point for this module. It applies the full
    filter stack (extension whitelist → path blacklist → size guard →
    binary check) and returns a clean list of SourceFile instances.

    WHY include_tests=True by default:
        Test files are valuable for understanding how functions are intended
        to be used. They're often the best documentation of expected behaviour.
        We include them by default but expose the flag so callers can opt out
        when they only want production code.

    Args:
        repo_root:           Path to the cloned repository root directory.
        max_file_size_bytes: Files larger than this are skipped.
        include_tests:       If False, directories named "tests", "test",
                             "__tests__", or "spec" are also skipped.

    Returns:
        Sorted list of SourceFile instances, ordered by relative_path.
        Sorting ensures deterministic ingestion order and makes diffs readable.

    Raises:
        FileNotFoundError: If repo_root does not exist.
        NotADirectoryError: If repo_root is a file, not a directory.
    """
    if not repo_root.exists():
        raise FileNotFoundError(f"Repository root not found: {repo_root}")
    if not repo_root.is_dir():
        raise NotADirectoryError(f"Expected a directory, got a file: {repo_root}")

    logger.info(f"Walking repository: {repo_root}")

    source_files: list[SourceFile] = []
    skipped_counts: dict[str, int] = {
        "blacklisted_dir": 0,
        "unsupported_extension": 0,
        "blacklisted_filename": 0,
        "too_large": 0,
        "binary": 0,
    }

    # Build the effective directory blacklist, optionally including test dirs.
    active_blacklist = set(BLACKLISTED_DIRS)
    if not include_tests:
        active_blacklist.update({"tests", "test", "__tests__", "spec", "fixtures"})

    for dirpath, dirnames, filenames in os.walk(repo_root, topdown=True):
        # Prune blacklisted directories IN PLACE.
        # Modifying dirnames[:] while topdown=True prevents os.walk from
        # recursing into those directories at all — much faster than filtering
        # files after the fact.
        dirnames[:] = [
            d for d in dirnames
            if d not in active_blacklist and not d.startswith(".")
        ]

        current_dir = Path(dirpath)

        for filename in filenames:
            file_path = current_dir / filename
            relative_path = str(file_path.relative_to(repo_root))

            # ── Filter 1: Blacklisted filename ────────────────────────────
            if filename in BLACKLISTED_FILENAMES:
                skipped_counts["blacklisted_filename"] += 1
                continue

            # ── Filter 2: Extension whitelist ─────────────────────────────
            extension = Path(filename).suffix.lower()
            language = SUPPORTED_EXTENSIONS.get(extension)
            if language is None:
                skipped_counts["unsupported_extension"] += 1
                continue

            # ── Filter 3: File size guard ──────────────────────────────────
            try:
                size_bytes = file_path.stat().st_size
            except OSError:
                logger.warning(f"Could not stat file, skipping: {file_path}")
                continue

            if size_bytes > max_file_size_bytes:
                logger.debug(
                    f"Skipping oversized file ({size_bytes:,} bytes): {relative_path}"
                )
                skipped_counts["too_large"] += 1
                continue

            # ── Filter 4: Binary check ─────────────────────────────────────
            if _is_binary_file(file_path):
                logger.debug(f"Skipping binary file: {relative_path}")
                skipped_counts["binary"] += 1
                continue

            source_files.append(
                SourceFile(
                    absolute_path=file_path,
                    relative_path=relative_path,
                    language=language,
                    size_bytes=size_bytes,
                    extension=extension,
                )
            )

    # Sort for deterministic ordering.
    source_files.sort(key=lambda f: f.relative_path)

    _log_walk_summary(repo_root, source_files, skipped_counts)
    return source_files


# ---------------------------------------------------------------------------
# Language-filtered walk
# ---------------------------------------------------------------------------

def walk_by_language(
    repo_root: Path,
    language: str,
    **kwargs,
) -> list[SourceFile]:
    """
    Return only source files matching a specific language.

    WHY:
        Some pipeline stages (e.g. running Java-specific AST rules) only
        apply to one language. This convenience wrapper avoids callers having
        to filter the full list themselves.

    Args:
        repo_root: Path to the cloned repository root.
        language:  One of "python", "javascript", "typescript", "java".
        **kwargs:  Forwarded to walk_repository().

    Returns:
        Filtered list of SourceFile instances for the requested language.

    Raises:
        ValueError: If language is not in SUPPORTED_EXTENSIONS.values().
    """
    supported_languages = set(SUPPORTED_EXTENSIONS.values())
    if language not in supported_languages:
        raise ValueError(
            f"Unsupported language '{language}'. "
            f"Choose from: {sorted(supported_languages)}"
        )

    all_files = walk_repository(repo_root, **kwargs)
    filtered = [f for f in all_files if f.language == language]
    logger.info(f"Language filter '{language}': {len(filtered)} files retained.")
    return filtered


# ---------------------------------------------------------------------------
# Generator variant for large repos
# ---------------------------------------------------------------------------

def stream_repository_files(
    repo_root: Path,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
) -> Generator[SourceFile, None, None]:
    """
    Yield SourceFile instances one at a time without building the full list.

    WHY:
        For very large repositories (10,000+ files), building the entire list
        in memory before starting parsing wastes time and memory. This generator
        variant allows the pipeline to start parsing and embedding immediately
        as files are discovered, enabling better pipelining and lower peak
        memory usage.

        Use walk_repository() when you need random access or sorted order.
        Use this when you're streaming into a processing queue.

    Args:
        repo_root:           Path to the cloned repository root.
        max_file_size_bytes: Files larger than this are skipped.

    Yields:
        SourceFile instances, in filesystem traversal order (not sorted).
    """
    if not repo_root.exists():
        raise FileNotFoundError(f"Repository root not found: {repo_root}")

    for dirpath, dirnames, filenames in os.walk(repo_root, topdown=True):
        dirnames[:] = [
            d for d in dirnames
            if d not in BLACKLISTED_DIRS and not d.startswith(".")
        ]

        current_dir = Path(dirpath)

        for filename in filenames:
            file_path = current_dir / filename
            extension = Path(filename).suffix.lower()
            language = SUPPORTED_EXTENSIONS.get(extension)

            if language is None:
                continue
            if filename in BLACKLISTED_FILENAMES:
                continue

            try:
                size_bytes = file_path.stat().st_size
            except OSError:
                continue

            if size_bytes > max_file_size_bytes:
                continue
            if _is_binary_file(file_path):
                continue

            yield SourceFile(
                absolute_path=file_path,
                relative_path=str(file_path.relative_to(repo_root)),
                language=language,
                size_bytes=size_bytes,
                extension=extension,
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_binary_file(file_path: Path, sample_size: int = 1024) -> bool:
    """
    Return True if the file appears to be binary (not valid UTF-8 text).

    WHY:
        Some repositories contain files with supported extensions that are
        actually binary (e.g. compiled .class files renamed, or .py files
        that are really pickle dumps). We check the first 1 KB for null bytes
        as a fast heuristic. Reading only a small sample keeps this O(1)
        regardless of file size.

    WHY null bytes specifically:
        Null bytes (0x00) are extremely rare in text files but common in
        binary formats. This is the same heuristic used by git itself to
        distinguish text from binary files.

    Args:
        file_path:   Path to the file to inspect.
        sample_size: Number of bytes to read from the start of the file.

    Returns:
        True if the file is likely binary, False if it appears to be text.
    """
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(sample_size)
        return b"\x00" in chunk
    except OSError:
        # If we can't read the file, treat it as binary to be safe.
        return True


def _log_walk_summary(
    repo_root: Path,
    source_files: list[SourceFile],
    skipped_counts: dict[str, int],
) -> None:
    """
    Emit a structured summary log after a repository walk completes.

    WHY:
        Without this summary it's hard to know why a repo produced fewer
        indexed files than expected. The breakdown by skip reason makes
        debugging ingestion issues fast — e.g. if 10,000 files were skipped
        as "unsupported_extension", the user may have a large asset directory
        that should be blacklisted.

    Args:
        repo_root:      The repository that was walked.
        source_files:   The files that passed all filters.
        skipped_counts: Per-reason skip counts from the walk loop.
    """
    language_counts: dict[str, int] = {}
    for f in source_files:
        language_counts[f.language] = language_counts.get(f.language, 0) + 1

    total_skipped = sum(skipped_counts.values())

    logger.info(
        f"Walk complete for {repo_root.name} | "
        f"accepted: {len(source_files)} files | "
        f"skipped: {total_skipped} files"
    )
    logger.info(f"Language breakdown: {language_counts}")
    logger.debug(f"Skip reasons: {skipped_counts}")
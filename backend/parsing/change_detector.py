"""
change_detector.py
───────────────────
PURPOSE
-------
Determines which files in a repository have changed since the last
indexing run, so that only modified files are re-parsed and re-embedded.

WHY THIS FILE EXISTS
--------------------
Re-indexing an entire large repository on every query or on every push
is prohibitively expensive — both in time (re-parsing + re-embedding)
and in cost (OpenAI embedding API charges per token).

This file solves the incremental update problem by using git's own change
tracking to identify exactly which files need work:

    Full re-index: 500 files × 2s each = ~17 minutes
    Incremental:    12 changed files × 2s each = ~24 seconds

The strategy is:
  1. Store a "last indexed commit SHA" after every successful index run.
  2. On the next run, compute `git diff <last_sha>..HEAD` to get the list
     of changed files.
  3. Only re-parse, re-embed, and re-upsert those files in Qdrant.
  4. Delete Qdrant points for deleted files so the index stays consistent.

USED BY
-------
- api/routes/ingest.py          triggers incremental re-index on webhook
- tasks/celery_worker.py        schedules periodic incremental syncs
- embeddings/embedding_cache.py reads file hashes to validate cache hits
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import git
from git import Repo, InvalidGitRepositoryError, GitCommandError

logger = logging.getLogger(__name__)

# Filename of the state file stored inside the repo's .codesense cache directory.
# This file persists the last indexed commit SHA across process restarts.
STATE_FILENAME = ".codesense_index_state.json"

# File extensions CodeSense supports indexing.
# Any changed file whose extension is not in this set is ignored.
INDEXED_EXTENSIONS = {
    ".py",   # Python
    ".js",   # JavaScript
    ".jsx",  # React/JSX
    ".ts",   # TypeScript
    ".tsx",  # TypeScript React
    ".java", # Java
}


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

class ChangeType(str, Enum):
    """
    Classification of how a file changed between two git commits.

    Mirrors git's own diff status codes (A/M/D/R/C) so the mapping
    is explicit and auditable.
    """
    ADDED = "added"        # New file — must be parsed and indexed for the first time.
    MODIFIED = "modified"  # Existing file changed — delete old vectors, re-index.
    DELETED = "deleted"    # File removed — delete all its vectors from Qdrant.
    RENAMED = "renamed"    # File moved — update file_path metadata on its vectors.
    COPIED = "copied"      # File duplicated — index the new copy.


@dataclass
class FileChange:
    """
    Describes a single file change detected between two git commits.

    Attributes
    ----------
    file_path : str
        Absolute path to the file on disk (or its old path for RENAMED files).
    change_type : ChangeType
        How the file changed.
    new_path : str | None
        New path after rename/copy.  None for all other change types.
    content_hash : str | None
        SHA-256 hash of the file's current content.  Stored so
        embedding_cache.py can verify cache validity without re-reading
        the file.  None for DELETED files (no content to hash).
    """
    file_path: str
    change_type: ChangeType
    new_path: str | None = None
    content_hash: str | None = None


@dataclass
class IndexState:
    """
    Persistent record of the last successful index run.

    Serialized as JSON to disk so it survives process restarts.

    Attributes
    ----------
    repo_path : str
        Absolute path of the indexed repository root.
    last_indexed_commit : str
        Full SHA of the git commit that was HEAD during the last index.
        Empty string means the repo has never been indexed (full index needed).
    indexed_file_count : int
        How many files were in the index after the last run.
        Used for sanity-check logging.
    last_indexed_at : str
        ISO-8601 timestamp of the last index run.
    file_hashes : dict[str, str]
        Mapping of file_path → SHA-256 content hash for every indexed file.
        Used by embedding_cache.py to detect cache invalidation without
        running a full git diff.
    """
    repo_path: str
    last_indexed_commit: str = ""
    indexed_file_count: int = 0
    last_indexed_at: str = ""
    file_hashes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "repo_path": self.repo_path,
            "last_indexed_commit": self.last_indexed_commit,
            "indexed_file_count": self.indexed_file_count,
            "last_indexed_at": self.last_indexed_at,
            "file_hashes": self.file_hashes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IndexState":
        return cls(
            repo_path=data["repo_path"],
            last_indexed_commit=data.get("last_indexed_commit", ""),
            indexed_file_count=data.get("indexed_file_count", 0),
            last_indexed_at=data.get("last_indexed_at", ""),
            file_hashes=data.get("file_hashes", {}),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Core Detector
# ─────────────────────────────────────────────────────────────────────────────

class ChangeDetector:
    """
    Detects file changes in a git repository since the last indexed commit.

    Uses GitPython to interact with the local git repository.  The detector
    is stateless between instantiations — all persistent state lives in the
    JSON state file managed by save_state() / load_state().

    Parameters
    ----------
    repo_path : str | Path
        Absolute path to the root of the git repository to monitor.
    cache_dir : str | Path | None
        Directory where the index state file is stored.  Defaults to
        `<repo_path>/.codesense/` if not provided.
    """

    def __init__(
        self,
        repo_path: str | Path,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.repo_path = Path(repo_path).resolve()

        # Validate that the path is actually a git repository before doing anything.
        try:
            self._repo = Repo(str(self.repo_path), search_parent_directories=True)
        except InvalidGitRepositoryError:
            raise ValueError(
                f"'{self.repo_path}' is not a git repository. "
                "Ensure the repository has been cloned before running ChangeDetector."
            )

        self._cache_dir = Path(cache_dir) if cache_dir else self.repo_path / ".codesense"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self._cache_dir / STATE_FILENAME

        logger.info("ChangeDetector initialized for repo at %s", self.repo_path)

    # ── Main Public Interface ─────────────────────────────────────────────────

    def get_changed_files(self) -> list[FileChange]:
        """
        Return all files that have changed since the last indexed commit.

        This is the primary method called by the ingestion pipeline.  It:
          1. Loads the saved index state to find the last indexed commit.
          2. Computes git diff between that commit and HEAD.
          3. Filters the diff to only include supported source file types.
          4. Returns structured FileChange objects for each changed file.

        If no state exists (first run), returns an empty list — the caller
        should use get_all_indexed_files() for a full initial index instead.

        Returns
        -------
        list[FileChange]
            Changed files since last indexed commit.  Empty if this is the
            first run or if no files have changed.
        """
        state = self.load_state()

        if not state.last_indexed_commit:
            logger.info(
                "No previous index state found for repo '%s'. "
                "A full index is required — use get_all_indexed_files().",
                self.repo_path,
            )
            return []

        current_commit = self._get_head_sha()
        if current_commit == state.last_indexed_commit:
            logger.info("Repo is up to date (HEAD = last indexed commit %s). No changes.",
                        current_commit[:8])
            return []

        logger.info(
            "Detecting changes from %s → %s",
            state.last_indexed_commit[:8],
            current_commit[:8],
        )
        changes = self._diff_commits(state.last_indexed_commit, current_commit)
        logger.info("Found %d changed source files", len(changes))
        return changes

    def get_all_indexed_files(self) -> list[FileChange]:
        """
        Return all source files in the repository as ADDED FileChange objects.

        Called on the first-ever index run when no previous state exists.
        Every file is treated as "added" so the indexing pipeline processes
        each one through the full parse → embed → upsert pipeline.

        Returns
        -------
        list[FileChange]
            Every indexable source file in the repo as ChangeType.ADDED entries.
        """
        all_files = []
        for path in self.repo_path.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in INDEXED_EXTENSIONS:
                continue
            if self._is_excluded_path(path):
                continue

            content_hash = self._hash_file(path)
            all_files.append(FileChange(
                file_path=str(path),
                change_type=ChangeType.ADDED,
                content_hash=content_hash,
            ))

        logger.info("Full index: found %d indexable files in %s",
                    len(all_files), self.repo_path)
        return all_files

    def needs_reindex(self, file_path: str | Path) -> bool:
        """
        Check whether a specific file needs re-indexing.

        Compares the file's current content hash against the hash stored
        in the last index state.  Used by embedding_cache.py to decide
        whether a cached embedding is still valid.

        This is faster than running a full git diff when you only need to
        check one file (e.g. in response to a file-watcher event).

        Parameters
        ----------
        file_path : str | Path
            Absolute path to the file to check.

        Returns
        -------
        bool
            True if the file's content has changed since last indexing.
        """
        path = Path(file_path).resolve()
        state = self.load_state()

        if str(path) not in state.file_hashes:
            # File was never indexed — must index it.
            return True

        current_hash = self._hash_file(path)
        stored_hash = state.file_hashes[str(path)]
        changed = current_hash != stored_hash

        if changed:
            logger.debug("File has changed since last index: %s", path.name)
        return changed

    def is_first_run(self) -> bool:
        """
        Return True if this repository has never been indexed by CodeSense.

        Checked by api/routes/ingest.py to decide whether to trigger a full
        index or an incremental update.

        Returns
        -------
        bool
            True if no index state file exists for this repo.
        """
        state = self.load_state()
        return not bool(state.last_indexed_commit)

    # ── State Persistence ─────────────────────────────────────────────────────

    def save_state(
        self,
        indexed_files: list[FileChange],
        indexed_file_count: int | None = None,
    ) -> None:
        """
        Persist the current index state to disk after a successful index run.

        Must be called by the ingestion pipeline AFTER all files have been
        successfully parsed, embedded, and upserted into Qdrant.  If the
        pipeline crashes mid-run, the state is not saved and the next run
        will re-process from the last successful checkpoint.

        Parameters
        ----------
        indexed_files : list[FileChange]
            The files that were processed in this index run.  Their content
            hashes are recorded for future cache validation.
        indexed_file_count : int | None
            Total number of files now in the index (not just this run's batch).
            If None, defaults to len(indexed_files).
        """
        from datetime import datetime, timezone

        # Build updated file hashes by merging existing state with new hashes.
        existing_state = self.load_state()
        updated_hashes = dict(existing_state.file_hashes)

        for change in indexed_files:
            if change.change_type == ChangeType.DELETED:
                # Remove deleted files from the hash map.
                updated_hashes.pop(change.file_path, None)
            elif change.content_hash:
                updated_hashes[change.file_path] = change.content_hash

        state = IndexState(
            repo_path=str(self.repo_path),
            last_indexed_commit=self._get_head_sha(),
            indexed_file_count=indexed_file_count or len(indexed_files),
            last_indexed_at=datetime.now(timezone.utc).isoformat(),
            file_hashes=updated_hashes,
        )

        with open(self._state_file, "w") as f:
            json.dump(state.to_dict(), f, indent=2)

        logger.info(
            "Index state saved: commit=%s, files=%d",
            state.last_indexed_commit[:8],
            state.indexed_file_count,
        )

    def load_state(self) -> IndexState:
        """
        Load the index state from disk.

        Returns a fresh empty IndexState if no state file exists yet
        (first run).  This design avoids requiring callers to handle
        FileNotFoundError.

        Returns
        -------
        IndexState
            The last saved index state, or an empty state for first runs.
        """
        if not self._state_file.exists():
            return IndexState(repo_path=str(self.repo_path))

        try:
            with open(self._state_file) as f:
                data = json.load(f)
            return IndexState.from_dict(data)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning(
                "Corrupted state file at %s (%s). Treating as first run.",
                self._state_file, exc,
            )
            return IndexState(repo_path=str(self.repo_path))

    # ── Git Internals ─────────────────────────────────────────────────────────

    def _diff_commits(self, old_sha: str, new_sha: str) -> list[FileChange]:
        """
        Compute the set of source files that changed between two commit SHAs.

        Uses GitPython's diff API which internally runs `git diff --name-status`.
        The raw git diff items are mapped to FileChange objects with proper
        ChangeType classification.

        Parameters
        ----------
        old_sha : str
            The older commit SHA (last indexed state).
        new_sha : str
            The newer commit SHA (current HEAD).

        Returns
        -------
        list[FileChange]
            Source file changes between the two commits.
        """
        try:
            old_commit = self._repo.commit(old_sha)
            new_commit = self._repo.commit(new_sha)
        except GitCommandError as exc:
            logger.error("Failed to resolve commits %s or %s: %s", old_sha, new_sha, exc)
            raise

        diff_items = old_commit.diff(new_commit)
        changes = []

        for diff_item in diff_items:
            change = self._parse_diff_item(diff_item)
            if change is not None:
                changes.append(change)

        return changes

    def _parse_diff_item(self, diff_item: git.Diff) -> FileChange | None:
        """
        Convert a single GitPython Diff object into a FileChange.

        Filters out non-source files (binaries, config files, etc.) and
        maps git change types (A/M/D/R/C) to ChangeType enum values.

        Parameters
        ----------
        diff_item : git.Diff
            A single item from a GitPython diff result.

        Returns
        -------
        FileChange | None
            A FileChange if the file should be re-indexed, None to skip it.
        """
        # GitPython uses single-character change type codes matching git.
        git_change_type_map = {
            "A": ChangeType.ADDED,
            "M": ChangeType.MODIFIED,
            "D": ChangeType.DELETED,
            "R": ChangeType.RENAMED,
            "C": ChangeType.COPIED,
        }

        raw_type = diff_item.change_type  # 'A', 'M', 'D', 'R100', 'C100', etc.
        # Rename/copy codes include a similarity percentage (R100, C75).
        # Strip the numeric suffix to get the base code.
        base_type = raw_type[0] if raw_type else "M"
        change_type = git_change_type_map.get(base_type, ChangeType.MODIFIED)

        # Determine the file path to check.
        file_path = diff_item.b_path or diff_item.a_path
        if not file_path:
            return None

        # Filter to only supported source file extensions.
        if Path(file_path).suffix not in INDEXED_EXTENSIONS:
            return None

        # Build the absolute path.
        abs_path = str(self.repo_path / file_path)

        # Hash the current content (skip for deleted files).
        content_hash = None
        if change_type != ChangeType.DELETED:
            target = Path(abs_path)
            if target.exists():
                content_hash = self._hash_file(target)

        new_path = None
        if change_type == ChangeType.RENAMED:
            new_path = str(self.repo_path / diff_item.b_path) if diff_item.b_path else None

        return FileChange(
            file_path=abs_path,
            change_type=change_type,
            new_path=new_path,
            content_hash=content_hash,
        )

    def _get_head_sha(self) -> str:
        """
        Return the full SHA of the current HEAD commit.

        Used as the checkpoint identifier saved after each index run.
        Full SHA (not short) is used to avoid ambiguity in large repos.

        Returns
        -------
        str
            40-character hex SHA of HEAD.
        """
        return self._repo.head.commit.hexsha

    # ── File Hashing ─────────────────────────────────────────────────────────

    @staticmethod
    def _hash_file(file_path: Path) -> str:
        """
        Compute the SHA-256 hash of a file's content.

        SHA-256 is used (not MD5 or CRC32) because:
          1. Collision resistance — we don't want false "no change" results.
          2. It is fast enough for source files (typically < 1MB each).
          3. The hash is stored in the state file and shared with
             embedding_cache.py, which also needs a collision-resistant hash.

        The file is read in binary mode so line-ending differences between
        platforms (CRLF vs LF) don't cause spurious cache misses.

        Parameters
        ----------
        file_path : Path
            Path to the file to hash.

        Returns
        -------
        str
            Hex-encoded SHA-256 digest of the file content.
        """
        hasher = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                # Read in 64KB chunks to avoid loading large files into memory.
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)
        except OSError as exc:
            logger.warning("Could not hash file %s: %s", file_path, exc)
            return ""
        return hasher.hexdigest()

    # ── Path Filtering ────────────────────────────────────────────────────────

    @staticmethod
    def _is_excluded_path(path: Path) -> bool:
        """
        Return True if a file path should be excluded from indexing.

        Excludes:
          • Virtual environments (venv/, .venv/, env/)
          • Build and dist output directories
          • Test fixture and snapshot directories
          • Minified JS/TS bundles (*.min.js)
          • Node modules

        WHY THIS MATTERS: Without this filter, indexing a Python project
        would include thousands of stdlib and third-party library files
        from the virtual environment, which are not part of the project's
        own codebase and would pollute retrieval results.

        Parameters
        ----------
        path : Path
            File path to evaluate.

        Returns
        -------
        bool
            True if the file should be skipped.
        """
        excluded_dirs = {
            "venv", ".venv", "env", "__pycache__",
            "node_modules", ".git", "dist", "build",
            ".mypy_cache", ".pytest_cache", "htmlcov",
            "migrations",  # Django/Flask DB migration files — auto-generated.
        }
        excluded_suffixes = {".min.js", ".min.ts", ".d.ts"}

        # Check if any path component is an excluded directory.
        if any(part in excluded_dirs for part in path.parts):
            return True

        # Check for excluded file suffixes.
        if any(str(path).endswith(suffix) for suffix in excluded_suffixes):
            return True

        return False

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def get_repo_summary(self) -> dict:
        """
        Return a summary of the repository's current state.

        Used by api/routes/ingest.py to populate the ingestion status
        response and by the frontend to show repository metadata in the
        file tree panel.

        Returns
        -------
        dict
            Repository metadata: branch, commit, file counts, index state.
        """
        state = self.load_state()
        current_commit = self._get_head_sha()

        try:
            branch = self._repo.active_branch.name
        except TypeError:
            # Detached HEAD state (common in CI environments).
            branch = "detached HEAD"

        return {
            "repo_path": str(self.repo_path),
            "current_branch": branch,
            "current_commit": current_commit,
            "last_indexed_commit": state.last_indexed_commit,
            "is_up_to_date": current_commit == state.last_indexed_commit,
            "indexed_file_count": state.indexed_file_count,
            "last_indexed_at": state.last_indexed_at,
        }

    def __repr__(self) -> str:
        state = self.load_state()
        return (
            f"ChangeDetector(repo='{self.repo_path.name}', "
            f"last_commit='{state.last_indexed_commit[:8] if state.last_indexed_commit else 'none'}')"
        )
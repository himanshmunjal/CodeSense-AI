"""
github_loader.py
----------------
PURPOSE:
    This module is the entry point for all repository ingestion in CodeSense.
    It handles two distinct access patterns:
        1. GitHub API access  — for fetching metadata, file trees, and small files
                                without cloning the full repo.
        2. Local git clone    — for large repos where we need full filesystem
                                access to run tree-sitter parsing and AST extraction.

WHY THIS FILE EXISTS:
    CodeSense needs to ingest source code from arbitrary GitHub repositories.
    Rather than asking users to manually download and point to local paths,
    this module abstracts away all GitHub interaction behind a clean interface.
    It also handles authentication (public vs private repos), rate limiting,
    and incremental re-ingestion using git history — so we only re-process
    files that have actually changed since the last index run.

DESIGN DECISIONS:
    - We use PyGithub for metadata/API calls and gitpython for local clone ops.
      These are complementary: PyGithub is clean for traversal, gitpython gives
      us low-level access to commits and diffs we need for change detection.
    - Clone target is a configurable temp directory, not the project root,
      so multiple repos can be ingested simultaneously without collision.
    - We do NOT store credentials anywhere in this module — all auth flows
      through environment variables loaded at startup via config.py.
"""

import base64
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import git  # gitpython
from github import Auth, Github, GithubException, Repository
from loguru import logger

from config import settings


# ---------------------------------------------------------------------------
# Client initialisation
# ---------------------------------------------------------------------------

def _build_github_client() -> Github:
    """
    Initialise and return an authenticated PyGithub client.

    WHY:
        Unauthenticated GitHub API calls are rate-limited to 60 req/hour.
        With a personal access token (PAT) this rises to 5,000 req/hour —
        essential when ingesting large repos with hundreds of files.
        We use the token from settings so credentials never appear in code.
    """
    token = settings.github_token
    if not token:
        logger.warning(
            "GITHUB_TOKEN not set. Falling back to unauthenticated access "
            "(60 req/hr rate limit). Set the token in your .env file."
        )
        return Github()
    return Github(auth=Auth.Token(token))


# Module-level client — created once, reused across all calls in this process.
_github_client: Github = _build_github_client()


# ---------------------------------------------------------------------------
# Repository resolution
# ---------------------------------------------------------------------------

def parse_repo_identifier(repo_input: str) -> tuple[str, str]:
    """
    Parse a GitHub repository identifier into (owner, repo_name).

    Accepts two formats:
        - Full URL:   "https://github.com/owner/repo"
        - Short form: "owner/repo"

    WHY:
        Users and API callers should not have to worry about format. This
        normalisation layer ensures the rest of the codebase always works
        with a consistent (owner, repo_name) tuple, regardless of input.

    Args:
        repo_input: URL or shorthand string identifying the repository.

    Returns:
        A (owner, repo_name) tuple — e.g. ("tiangolo", "fastapi").

    Raises:
        ValueError: If the input cannot be parsed into a valid owner/repo pair.
    """
    repo_input = repo_input.strip().rstrip("/")

    if repo_input.startswith("https://") or repo_input.startswith("http://"):
        parsed = urlparse(repo_input)
        parts = parsed.path.strip("/").split("/")
        if len(parts) < 2:
            raise ValueError(f"Cannot extract owner/repo from URL: {repo_input}")
        return parts[0], parts[1]

    parts = repo_input.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"Invalid repo identifier '{repo_input}'. "
            "Expected 'owner/repo' or a full GitHub URL."
        )
    return parts[0], parts[1]


def get_github_repo(repo_input: str) -> Repository.Repository:
    """
    Fetch and return a PyGithub Repository object for the given identifier.

    WHY:
        Centralising this call means all callers benefit from the same
        error handling and logging. It also makes mocking in tests trivial —
        patch this one function and every downstream caller gets a fake repo.

    Args:
        repo_input: Full GitHub URL or "owner/repo" shorthand.

    Returns:
        A PyGithub Repository object with full metadata access.

    Raises:
        GithubException: On API errors (repo not found, insufficient permissions).
        ValueError: If repo_input cannot be parsed.
    """
    owner, repo_name = parse_repo_identifier(repo_input)
    full_name = f"{owner}/{repo_name}"

    logger.info(f"Fetching GitHub repository metadata: {full_name}")
    try:
        repo = _github_client.get_repo(full_name)
        logger.info(
            f"Resolved repo: {repo.full_name} | "
            f"default branch: {repo.default_branch} | "
            f"size: {repo.size} KB"
        )
        return repo
    except GithubException as exc:
        logger.error(f"Failed to fetch repo '{full_name}': {exc.status} {exc.data}")
        raise


# ---------------------------------------------------------------------------
# Local cloning
# ---------------------------------------------------------------------------

def clone_repository(
    repo_input: str,
    target_dir: Optional[str] = None,
    branch: Optional[str] = None,
) -> Path:
    """
    Clone a GitHub repository to a local directory and return its path.

    WHY:
        The GitHub API is not suitable for bulk file reading — it's rate-limited
        and requires a separate HTTP call per file. tree-sitter parsing (the next
        stage in our pipeline) needs direct filesystem access to every source file.
        Cloning once upfront is far more efficient than fetching file-by-file.

    WHY shallow clone (depth=1):
        For CodeSense we only need the current state of the code, not the full
        commit history. A shallow clone is dramatically faster and uses a fraction
        of disk space — a repo with years of history might be 500 MB as a full
        clone but only 20 MB shallow.

    Args:
        repo_input:  GitHub URL or "owner/repo" shorthand.
        target_dir:  Local path to clone into. If None, a temp directory is
                     created automatically and the caller is responsible for
                     cleanup via delete_cloned_repo().
        branch:      Branch to clone. Defaults to the repo's default branch.

    Returns:
        Path object pointing to the root of the cloned repository.

    Raises:
        git.GitCommandError: On clone failure (network error, auth failure, etc.)
        ValueError: If repo_input cannot be parsed.
    """
    owner, repo_name = parse_repo_identifier(repo_input)
    clone_url = _build_clone_url(owner, repo_name)

    if target_dir is None:
        target_dir = tempfile.mkdtemp(prefix=f"codesense_{repo_name}_")
        logger.debug(f"Created temp clone directory: {target_dir}")

    clone_path = Path(target_dir)

    # If the directory already contains a git repo, update it in place instead
    # of re-cloning. Returning the existing checkout as-is would mean a
    # re-ingest never sees any commit pushed after the first clone.
    if (clone_path / ".git").exists():
        _update_existing_clone(clone_path, clone_url, branch)
        return clone_path

    logger.info(f"Cloning {owner}/{repo_name} → {clone_path}")
    try:
        clone_kwargs: dict = {
            "depth": 1,          # Shallow clone — current snapshot only.
            "single_branch": True,
        }
        if branch:
            clone_kwargs["branch"] = branch

        _with_auth_fallback(
            lambda env: git.Repo.clone_from(clone_url, str(clone_path), env=env, **clone_kwargs),
            cleanup=lambda: shutil.rmtree(clone_path, ignore_errors=True),
        )
        logger.success(f"Clone complete: {clone_path}")
        return clone_path

    except git.GitCommandError as exc:
        logger.error(f"Git clone failed for {owner}/{repo_name}: {exc}")
        # Clean up the partial clone directory to avoid leaving behind broken state.
        if clone_path.exists():
            shutil.rmtree(clone_path, ignore_errors=True)
        raise


def _update_existing_clone(clone_path: Path, clone_url: str, branch: Optional[str]) -> None:
    """
    Fast-forward an existing shallow clone to the latest commit of `branch`
    (or the remote's default branch when None).

    A shallow, single-branch clone can't simply `git pull` another branch, so
    we fetch exactly one commit of the requested ref and hard-reset onto it.
    The previous HEAD commit stays in the object store, so ChangeDetector can
    still compare the old and new commits afterwards.
    """
    repo = git.Repo(str(clone_path))
    origin = repo.remote("origin")
    # Clones made before auth moved to _git_auth_env() have the token embedded
    # in their origin URL (and therefore in .git/config) — reset it.
    origin.set_url(clone_url)

    ref = branch or "HEAD"
    logger.info(f"Repository already cloned at {clone_path}. Fetching latest '{ref}'.")
    def fetch(env: dict[str, str]) -> None:
        with repo.git.custom_environment(**env):
            origin.fetch(ref, depth=1)

    _with_auth_fallback(fetch)
    repo.git.reset("--hard", "FETCH_HEAD")
    logger.info(f"Updated {clone_path} to {repo.head.commit.hexsha[:8]}")


def _with_auth_fallback(git_call, cleanup=None):
    """
    Run `git_call(env)` with the GitHub token, retrying once without it if
    GitHub rejects the token.

    A bad token (expired, revoked, or rotated in .env while the worker kept
    the old value) makes GitHub refuse even PUBLIC repos — so for the common
    case it's better to warn and retry anonymously. Private repos still fail,
    with GitHub's own "not found" error.
    """
    auth_env = _git_auth_env()
    try:
        return git_call(auth_env)
    except git.GitCommandError as exc:
        if not auth_env or "Authentication failed" not in str(exc):
            raise
        logger.warning(
            "GitHub rejected GITHUB_TOKEN (invalid, expired, or changed in .env "
            "after this process started — restart it). Retrying without auth."
        )
        if cleanup:
            cleanup()
        return git_call({"GIT_TERMINAL_PROMPT": "0"})


def _build_clone_url(owner: str, repo_name: str) -> str:
    """
    Construct the plain HTTPS clone URL for a repository.

    The token is deliberately NOT embedded in the URL: git persists the clone
    URL as the `origin` remote in .git/config, and GitCommandError messages
    (which end up in logs and in the ingest status API response) include the
    full command line. Authentication is supplied via _git_auth_env() instead.
    """
    return f"https://github.com/{owner}/{repo_name}.git"


def _git_auth_env() -> dict[str, str]:
    """
    Environment variables that make git send the GitHub token as an HTTP
    Authorization header for a single command, without writing it to any
    config file or putting it on the command line (GIT_CONFIG_COUNT/KEY/VALUE
    are git's documented per-process config overrides, git >= 2.31).
    """
    token = settings.github_token
    if not token:
        return {}
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return {
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
        "GIT_CONFIG_VALUE_0": f"Authorization: Basic {basic}",
        # Never block on an interactive credential prompt inside a worker.
        "GIT_TERMINAL_PROMPT": "0",
    }


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def delete_cloned_repo(clone_path: Path) -> None:
    """
    Remove a previously cloned repository from the local filesystem.

    WHY:
        Cloned repos can be large (tens to hundreds of MB). After parsing and
        embedding are complete, the raw source files are no longer needed —
        all structured data lives in Qdrant. This function should be called
        at the end of a successful ingestion pipeline run to reclaim disk space.

    Args:
        clone_path: Path returned by clone_repository().
    """
    if clone_path.exists():
        logger.info(f"Removing cloned repo at {clone_path}")
        shutil.rmtree(clone_path, ignore_errors=True)
        logger.debug(f"Deleted: {clone_path}")
    else:
        logger.warning(f"delete_cloned_repo called on non-existent path: {clone_path}")


# ---------------------------------------------------------------------------
# Repository metadata helpers
# ---------------------------------------------------------------------------

def get_repo_metadata(repo_input: str) -> dict:
    """
    Return a lightweight metadata dict for a repository.

    WHY:
        We store this metadata alongside indexed chunks in Qdrant so that
        query results can include source context (which repo, which branch,
        when it was last updated). This function consolidates all the fields
        we care about into one place so the schema doesn't scatter across modules.

    Args:
        repo_input: GitHub URL or "owner/repo" shorthand.

    Returns:
        Dictionary with keys: full_name, owner, repo_name, default_branch,
        description, stars, language, size_kb, clone_url, last_pushed_at.
    """
    repo = get_github_repo(repo_input)
    return {
        "full_name": repo.full_name,
        "owner": repo.owner.login,
        "repo_name": repo.name,
        "default_branch": repo.default_branch,
        "description": repo.description or "",
        "stars": repo.stargazers_count,
        "language": repo.language or "unknown",
        "size_kb": repo.size,
        "clone_url": repo.clone_url,
        "last_pushed_at": repo.pushed_at.isoformat() if repo.pushed_at else None,
    }
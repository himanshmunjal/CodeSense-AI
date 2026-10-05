// Thin fetch wrapper for the CodeSense backend.
// All routes live under /api/v1/ (see backend/api/main.py and
// backend/api/routes/*.py).

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || body.error || JSON.stringify(body);
    } catch {
      // response wasn't JSON — fall back to statusText
    }
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }

  if (res.status === 204) return null;
  return res.json();
}

// ── Ingestion ────────────────────────────────────────────────────────────────

export function ingestRepository({ repoUrl, branch, forceReindex }) {
  return request("/api/v1/ingest/", {
    method: "POST",
    body: JSON.stringify({
      repo_url: repoUrl,
      branch: branch || undefined,
      force_reindex: !!forceReindex,
    }),
  });
}

export function getIngestStatus(taskId) {
  return request(`/api/v1/ingest/status/${taskId}`);
}

export function listRepositories() {
  return request("/api/v1/ingest/repositories");
}

export function deleteRepository(owner, repo) {
  return request(`/api/v1/ingest/${owner}/${repo}`, { method: "DELETE" });
}

// ── Query ────────────────────────────────────────────────────────────────────

export function queryRepository({
  repoUrl,
  question,
  maxResults,
  filterLanguage,
  filterFilePath,
}) {
  return request("/api/v1/query/", {
    method: "POST",
    body: JSON.stringify({
      repo_url: repoUrl,
      question,
      max_results: maxResults || 5,
      filter_language: filterLanguage || undefined,
      filter_file_path: filterFilePath || undefined,
    }),
  });
}

export function getQueryHistory(owner, repo) {
  return request(`/api/v1/query/history/${owner}/${repo}`);
}

// ── Impact analysis ──────────────────────────────────────────────────────────

export function analyzeImpact({ repoUrl, functionName, filePath, maxDepth }) {
  return request("/api/v1/impact/", {
    method: "POST",
    body: JSON.stringify({
      repo_url: repoUrl,
      function_name: functionName,
      file_path: filePath || undefined,
      max_depth: maxDepth || 4,
    }),
  });
}

// ── Summarization ────────────────────────────────────────────────────────────

export function summarize({ repoUrl, filePath, directoryPath, focusHint }) {
  return request("/api/v1/summarize/", {
    method: "POST",
    body: JSON.stringify({
      repo_url: repoUrl,
      file_path: filePath || undefined,
      directory_path: directoryPath || undefined,
      focus_hint: focusHint || undefined,
    }),
  });
}

export function summarizeModule({ repoUrl, modulePath }) {
  return request("/api/v1/summarize/module", {
    method: "POST",
    body: JSON.stringify({ repo_url: repoUrl, module_path: modulePath }),
  });
}

// ── Helpers ──────────────────────────────────────────────────────────────────

export function parseGithubUrl(repoUrl) {
  const parts = repoUrl.replace(/\/+$/, "").split("/");
  return { owner: parts[parts.length - 2], repo: parts[parts.length - 1] };
}

import { useState } from "react";
import "./FileTree.css";

// NOTE ON DEVIATION FROM THE README:
// The README describes this panel as a "repo file tree with per-file indexing
// status." The real backend (backend/api/routes/*.py) does not expose any
// endpoint that lists individual files or their per-file status — ingestion
// status is repo-level only (GET /api/v1/ingest/status/{task_id} reports
// files_processed/files_total counts, not a per-file tree). There is also no
// "list files in repo X" endpoint once ingestion completes.
//
// So this panel does two real things instead:
//   1. Lists indexed repositories (GET /api/v1/ingest/repositories) with
//      their real metadata (languages, chunk count, branch, indexed_at) and
//      lets you pick which one is "active" for the chat/impact panels.
//   2. Shows the set of files actually seen so far via query citations
//      (sources[].file_path from POST /api/v1/query/) as a lightweight
//      "touched files" list per repo — this is real, per-file, backed by
//      actual API responses, just not a full repo tree (which the backend
//      doesn't support today).

const STATUS_LABEL = {
  pending: "queued",
  running: "indexing…",
  completed: "indexed",
  failed: "failed",
};

export default function FileTree({
  repositories,
  activeRepoUrl,
  onSelectRepo,
  onIngest,
  onDeleteRepo,
  ingestJobs,
  seenFiles,
  onSelectFile,
}) {
  const [repoUrlInput, setRepoUrlInput] = useState("");
  const [branchInput, setBranchInput] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [expandedRepo, setExpandedRepo] = useState(null);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!repoUrlInput.trim()) return;
    setSubmitting(true);
    try {
      await onIngest(repoUrlInput.trim(), branchInput.trim() || undefined);
      setRepoUrlInput("");
      setBranchInput("");
    } finally {
      setSubmitting(false);
    }
  }

  const activeJobs = Object.values(ingestJobs || {});

  return (
    <div className="filetree">
      <div className="filetree__header">
        <h2>Repositories</h2>
      </div>

      <form className="filetree__ingest-form" onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="https://github.com/owner/repo"
          value={repoUrlInput}
          onChange={(e) => setRepoUrlInput(e.target.value)}
        />
        <input
          type="text"
          placeholder="branch (optional)"
          value={branchInput}
          onChange={(e) => setBranchInput(e.target.value)}
          className="filetree__branch-input"
        />
        <button type="submit" disabled={submitting || !repoUrlInput.trim()}>
          {submitting ? "Queuing…" : "Index Repository"}
        </button>
      </form>

      {activeJobs.length > 0 && (
        <div className="filetree__jobs">
          {activeJobs.map((job) => (
            <div key={job.taskId} className="filetree__job">
              <span className={`filetree__job-dot filetree__job-dot--${job.status}`} />
              <span className="filetree__job-repo">{job.repoUrl.replace("https://github.com/", "")}</span>
              <span className="filetree__job-status">
                {STATUS_LABEL[job.status] || job.status}
                {job.progress?.files_total
                  ? ` (${job.progress.files_parsed ?? 0}/${job.progress.files_total})`
                  : ""}
              </span>
              {job.error && <span className="filetree__job-error" title={job.error}>error</span>}
            </div>
          ))}
        </div>
      )}

      <div className="filetree__list">
        {repositories.length === 0 && (
          <p className="filetree__empty">No repositories indexed yet. Paste a GitHub URL above.</p>
        )}
        {repositories.map((repo) => {
          const isActive = repo.repo_url === activeRepoUrl;
          const isExpanded = expandedRepo === repo.repo_url;
          const files = seenFiles?.[repo.repo_url] || [];
          return (
            <div
              key={repo.repo_url}
              className={`filetree__repo ${isActive ? "filetree__repo--active" : ""}`}
            >
              <div
                className="filetree__repo-row"
                onClick={() => onSelectRepo(repo.repo_url)}
              >
                <span className="filetree__repo-name">
                  {repo.owner}/{repo.repo}
                </span>
                <span className="filetree__repo-chunks">{repo.total_chunks} chunks</span>
              </div>
              <div className="filetree__repo-meta">
                <span>{repo.branch}</span>
                <span>{(repo.languages || []).join(", ")}</span>
              </div>
              <div className="filetree__repo-actions">
                <button
                  className="filetree__link-btn"
                  onClick={() => setExpandedRepo(isExpanded ? null : repo.repo_url)}
                >
                  {isExpanded ? "hide files" : `files (${files.length})`}
                </button>
                <button
                  className="filetree__link-btn filetree__link-btn--danger"
                  onClick={() => onDeleteRepo(repo.owner, repo.repo)}
                >
                  delete
                </button>
              </div>
              {isExpanded && (
                <div className="filetree__files">
                  {files.length === 0 && (
                    <p className="filetree__empty filetree__empty--small">
                      No files seen yet — ask a question in the chat to populate this list from citations.
                    </p>
                  )}
                  {files.map((f) => (
                    <button
                      key={f}
                      className="filetree__file"
                      onClick={() => onSelectFile(repo.repo_url, f)}
                    >
                      {f}
                    </button>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

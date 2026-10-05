import { useState, useCallback, useEffect, useRef } from "react";
import FileTree from "./components/FileTree.jsx";
import ChatPanel from "./components/ChatPanel.jsx";
import SourceViewer from "./components/SourceViewer.jsx";
import ImpactGraph from "./components/ImpactGraph.jsx";
import * as api from "./api/client.js";
import "./App.css";

let idCounter = 0;
const nextId = () => `msg-${++idCounter}`;

export default function App() {
  const [repositories, setRepositories] = useState([]);
  const [activeRepoUrl, setActiveRepoUrl] = useState(null);
  const [ingestJobs, setIngestJobs] = useState({}); // taskId -> { taskId, repoUrl, status, progress, error }
  const [chatsByRepo, setChatsByRepo] = useState({}); // repoUrl -> messages[]
  const [loading, setLoading] = useState(false);
  const [rightTab, setRightTab] = useState("source"); // "source" | "impact"
  const [selectedSource, setSelectedSource] = useState(null);
  const [seenFiles, setSeenFiles] = useState({}); // repoUrl -> string[]
  const [impactState, setImpactState] = useState({ result: null, loading: false, error: null });

  const pollTimers = useRef({});

  const refreshRepositories = useCallback(async () => {
    try {
      const res = await api.listRepositories();
      setRepositories(res.repositories || []);
      if (!activeRepoUrl && res.repositories?.length) {
        setActiveRepoUrl(res.repositories[0].repo_url);
      }
    } catch (e) {
      console.error("Failed to list repositories", e);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeRepoUrl]);

  useEffect(() => {
    refreshRepositories();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Ingestion ────────────────────────────────────────────────────────────
  const pollIngestStatus = useCallback((taskId, repoUrl) => {
    if (pollTimers.current[taskId]) return;
    const tick = async () => {
      try {
        const status = await api.getIngestStatus(taskId);
        setIngestJobs((prev) => ({
          ...prev,
          [taskId]: {
            taskId,
            repoUrl,
            status: status.status,
            progress: status.progress,
            error: status.error_message,
          },
        }));

        if (status.status === "completed" || status.status === "failed") {
          clearInterval(pollTimers.current[taskId]);
          delete pollTimers.current[taskId];
          if (status.status === "completed") {
            refreshRepositories();
            // Clear the job chip after a short delay so success is visible briefly.
            setTimeout(() => {
              setIngestJobs((prev) => {
                const copy = { ...prev };
                delete copy[taskId];
                return copy;
              });
            }, 4000);
          }
        }
      } catch (e) {
        console.error("Failed to poll ingest status", e);
      }
    };
    tick();
    pollTimers.current[taskId] = setInterval(tick, 5000); // recommended interval per ingest.py docs
  }, [refreshRepositories]);

  useEffect(() => {
    return () => {
      Object.values(pollTimers.current).forEach(clearInterval);
    };
  }, []);

  const handleIngest = useCallback(
    async (repoUrl, branch) => {
      const res = await api.ingestRepository({ repoUrl, branch });
      setIngestJobs((prev) => ({
        ...prev,
        [res.task_id]: { taskId: res.task_id, repoUrl: res.repo_url, status: res.status },
      }));
      pollIngestStatus(res.task_id, res.repo_url);
    },
    [pollIngestStatus]
  );

  const handleDeleteRepo = useCallback(
    async (owner, repo) => {
      await api.deleteRepository(owner, repo);
      await refreshRepositories();
    },
    [refreshRepositories]
  );

  // ── Chat / Query ─────────────────────────────────────────────────────────
  const messages = activeRepoUrl ? chatsByRepo[activeRepoUrl] || [] : [];

  const appendMessage = useCallback((repoUrl, msg) => {
    setChatsByRepo((prev) => ({
      ...prev,
      [repoUrl]: [...(prev[repoUrl] || []), msg],
    }));
  }, []);

  const recordSeenFiles = useCallback((repoUrl, sources) => {
    if (!sources?.length) return;
    setSeenFiles((prev) => {
      const existing = new Set(prev[repoUrl] || []);
      sources.forEach((s) => existing.add(s.file_path));
      return { ...prev, [repoUrl]: [...existing].sort() };
    });
  }, []);

  const handleAsk = useCallback(
    async (question) => {
      if (!activeRepoUrl) return;
      const repoUrl = activeRepoUrl;
      appendMessage(repoUrl, { id: nextId(), role: "user", text: question });
      setLoading(true);
      try {
        const res = await api.queryRepository({ repoUrl, question });
        appendMessage(repoUrl, {
          id: nextId(),
          role: "assistant",
          text: res.answer,
          confidence: res.confidence,
          sourcesUsed: res.sources_used,
          queryType: res.query_type,
          latencyMs: res.latency_ms,
          sources: res.sources,
          followupQueries: res.followup_queries,
        });
        recordSeenFiles(repoUrl, res.sources);
      } catch (e) {
        appendMessage(repoUrl, {
          id: nextId(),
          role: "error",
          text: e.message || "Query failed.",
        });
      } finally {
        setLoading(false);
      }
    },
    [activeRepoUrl, appendMessage, recordSeenFiles]
  );

  // ── Impact analysis ──────────────────────────────────────────────────────
  const runImpact = useCallback(
    async (functionName, filePath) => {
      if (!activeRepoUrl) return;
      setRightTab("impact");
      setImpactState({ result: null, loading: true, error: null });
      try {
        const res = await api.analyzeImpact({
          repoUrl: activeRepoUrl,
          functionName,
          filePath,
        });
        setImpactState({ result: res, loading: false, error: null });
      } catch (e) {
        setImpactState({ result: null, loading: false, error: e.message || "Impact analysis failed." });
      }
    },
    [activeRepoUrl]
  );

  const handleSelectSource = useCallback((source) => {
    setSelectedSource(source);
    setRightTab("source");
  }, []);

  const handleSelectFile = useCallback((repoUrl, filePath) => {
    // We only have snippet-level data from prior query citations for this file,
    // not a full-file endpoint (see SourceViewer.jsx note). Look up the most
    // recent citation for this file in chat history.
    const msgsForRepo = chatsByRepo[repoUrl] || [];
    for (let i = msgsForRepo.length - 1; i >= 0; i--) {
      const found = msgsForRepo[i].sources?.find((s) => s.file_path === filePath);
      if (found) {
        setSelectedSource(found);
        setRightTab("source");
        return;
      }
    }
  }, [chatsByRepo]);

  const activeRepoMeta = repositories.find((r) => r.repo_url === activeRepoUrl);

  return (
    <div className="app">
      <aside className="app__left">
        <FileTree
          repositories={repositories}
          activeRepoUrl={activeRepoUrl}
          onSelectRepo={setActiveRepoUrl}
          onIngest={handleIngest}
          onDeleteRepo={handleDeleteRepo}
          ingestJobs={ingestJobs}
          seenFiles={seenFiles}
          onSelectFile={handleSelectFile}
        />
      </aside>

      <main className="app__center">
        <ChatPanel
          activeRepo={activeRepoUrl}
          messages={messages}
          onAsk={handleAsk}
          onRunImpact={(fnName) => runImpact(fnName)}
          onSelectSource={handleSelectSource}
          loading={loading}
        />
      </main>

      <aside className="app__right">
        <div className="app__right-tabs">
          <button
            className={rightTab === "source" ? "app__tab app__tab--active" : "app__tab"}
            onClick={() => setRightTab("source")}
          >
            Source
          </button>
          <button
            className={rightTab === "impact" ? "app__tab app__tab--active" : "app__tab"}
            onClick={() => setRightTab("impact")}
          >
            Impact Graph
          </button>
        </div>
        <div className="app__right-body">
          {rightTab === "source" ? (
            <SourceViewer
              selectedSource={selectedSource}
              onRunImpact={(fnName, filePath) => runImpact(fnName, filePath)}
            />
          ) : (
            <ImpactGraph
              result={impactState.result}
              loading={impactState.loading}
              error={impactState.error}
            />
          )}
        </div>
      </aside>

      {activeRepoMeta && (
        <div className="app__statusbar">
          Active: {activeRepoMeta.owner}/{activeRepoMeta.repo} · {activeRepoMeta.total_chunks} chunks ·
          indexed {new Date(activeRepoMeta.indexed_at).toLocaleString()}
        </div>
      )}
    </div>
  );
}

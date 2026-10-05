import { useState, useRef, useEffect } from "react";
import SourcesUsedBadge from "./SourcesUsedBadge.jsx";
import "./ChatPanel.css";

// Talks to POST /api/v1/query/ (backend/api/routes/query.py).
// QueryResponse: { question, answer, sources: CodeSource[], confidence,
//                   sources_used, query_type, repo_url, followup_queries,
//                   latency_ms }
// sources_used replaced confidence as the primary per-answer badge — see
// SourcesUsedBadge.jsx for why (confidence is retrieval similarity, not an
// answer-correctness signal, and its old badge thresholds had drifted out
// of sync with the backend's actual grounding threshold).
// CodeSource: { file_path, function_name, start_line, end_line, language,
//               relevance_score, snippet }
//
// "impact analysis: fn_name" is handled as a special-cased message (per the
// README's Usage section) that runs POST /api/v1/impact/ instead of /query/.

const IMPACT_PATTERN = /^impact analysis:\s*(.+)$/i;

export default function ChatPanel({
  activeRepo,
  messages,
  onAsk,
  onRunImpact,
  onSelectSource,
  loading,
}) {
  const [input, setInput] = useState("");
  const scrollRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  function handleSubmit(e) {
    e.preventDefault();
    const text = input.trim();
    if (!text || loading || !activeRepo) return;

    const impactMatch = text.match(IMPACT_PATTERN);
    if (impactMatch) {
      onRunImpact(impactMatch[1].trim());
    } else {
      onAsk(text);
    }
    setInput("");
  }

  return (
    <div className="chatpanel">
      <div className="chatpanel__header">
        <h2>Chat</h2>
        <span className="chatpanel__repo">
          {activeRepo ? activeRepo.replace("https://github.com/", "") : "No repository selected"}
        </span>
      </div>

      <div className="chatpanel__messages" ref={scrollRef}>
        {messages.length === 0 && (
          <div className="chatpanel__hint">
            <p>Ask anything about the codebase, e.g.:</p>
            <ul>
              <li>"Where is authentication handled?"</li>
              <li>"What calls the payment processing function?"</li>
              <li>impact analysis: verify_token</li>
            </ul>
          </div>
        )}

        {messages.map((msg) => (
          <div key={msg.id} className={`chatpanel__msg chatpanel__msg--${msg.role}`}>
            {msg.role === "user" && <div className="chatpanel__bubble">{msg.text}</div>}

            {msg.role === "assistant" && (
              <div className="chatpanel__bubble chatpanel__bubble--assistant">
                <div className="chatpanel__answer-head">
                  <SourcesUsedBadge
                    sourcesUsed={msg.sourcesUsed}
                    sourcesTotal={msg.sources?.length ?? 0}
                  />
                  <span className="chatpanel__query-type">{msg.queryType}</span>
                  <span className="chatpanel__latency">{msg.latencyMs?.toFixed?.(0)}ms</span>
                </div>
                <p className="chatpanel__answer-text">{msg.text}</p>

                {msg.sources?.length > 0 && (
                  <div className="chatpanel__sources">
                    <div className="chatpanel__sources-label">Sources</div>
                    {msg.sources.map((s, i) => (
                      <button
                        key={i}
                        className="chatpanel__citation"
                        onClick={() => onSelectSource(s)}
                      >
                        {s.file_path}:{s.function_name}:{s.start_line}-{s.end_line}
                      </button>
                    ))}
                  </div>
                )}

                {msg.followupQueries?.length > 0 && (
                  <div className="chatpanel__followups">
                    {msg.followupQueries.map((q, i) => (
                      <button
                        key={i}
                        className="chatpanel__followup"
                        onClick={() => onAsk(q)}
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}

            {msg.role === "error" && (
              <div className="chatpanel__bubble chatpanel__bubble--error">{msg.text}</div>
            )}
          </div>
        ))}

        {loading && (
          <div className="chatpanel__msg chatpanel__msg--assistant">
            <div className="chatpanel__bubble chatpanel__bubble--assistant chatpanel__bubble--loading">
              Thinking…
            </div>
          </div>
        )}
      </div>

      <form className="chatpanel__input-row" onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder={activeRepo ? "Ask a question about the codebase…" : "Select or index a repository first"}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={!activeRepo || loading}
        />
        <button type="submit" disabled={!activeRepo || loading || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}

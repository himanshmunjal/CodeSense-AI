import ConfidenceBadge from "./ConfidenceBadge.jsx";
import "./SourceViewer.css";

// NOTE ON DEVIATION FROM THE README:
// The README implies clicking a citation "jumps to that file and line" as if
// browsing a full file. The backend has no "fetch raw file content" endpoint
// (backend/api/routes/*.py never exposes file bytes) — it only ever returns
// per-chunk `snippet` text embedded in CodeSource (query.py) or nothing at all
// (impact.py's AffectedNode has no snippet field). So this viewer renders the
// exact snippet the backend already gave us for the selected citation, with
// its file_path:function_name:start_line-end_line header, rather than
// simulating a full-file browser we have no data to back.

export default function SourceViewer({ selectedSource, onRunImpact }) {
  if (!selectedSource) {
    return (
      <div className="sourceviewer sourceviewer--empty">
        <p>Click a citation in the chat to view its source here.</p>
      </div>
    );
  }

  const { file_path, function_name, start_line, end_line, language, relevance_score, snippet } =
    selectedSource;

  const lines = (snippet || "").split("\n");

  return (
    <div className="sourceviewer">
      <div className="sourceviewer__header">
        <div className="sourceviewer__path" title={file_path}>
          {file_path}
        </div>
        <div className="sourceviewer__subheader">
          <span className="sourceviewer__fn">{function_name}</span>
          <span className="sourceviewer__lines">
            L{start_line}-{end_line}
          </span>
          <span className="sourceviewer__lang">{language}</span>
          {typeof relevance_score === "number" && (
            <ConfidenceBadge confidence={relevance_score} size="small" />
          )}
        </div>
        <button
          className="sourceviewer__impact-btn"
          onClick={() => onRunImpact(function_name, file_path)}
        >
          Analyze impact of {function_name}()
        </button>
      </div>

      <div className="sourceviewer__code">
        <pre>
          <code>
            {lines.map((line, i) => (
              <div key={i} className="sourceviewer__line">
                <span className="sourceviewer__lineno">{start_line + i}</span>
                <span className="sourceviewer__linetext">{line || " "}</span>
              </div>
            ))}
          </code>
        </pre>
      </div>
    </div>
  );
}

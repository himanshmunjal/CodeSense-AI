import "./ConfidenceBadge.css";

// This is a per-SOURCE relevance score (raw embedding similarity for one
// retrieved chunk) — see SourcesUsedBadge.jsx for the primary per-ANSWER
// indicator in ChatPanel, which this was replaced by there for exactly the
// reason explained below.
//
// Grounding rule (backend config.py: settings.retrieval_confidence_threshold,
// currently 0.5 — NOT 0.65). That value was recalibrated for the current
// embedding model (bge-small-en-v1.5): measured directly on this project's
// real indexed repos, a clearly irrelevant query topped out at ~0.43 cosine
// similarity while genuinely correct answers scored as low as 0.55-0.61 —
// so 0.65 was quietly refusing/mislabeling correct results as "low
// confidence" for this model. These breakpoints previously hardcoded that
// stale 0.65/0.85 split, independent of the backend's actual threshold —
// keep them loosely in sync with config.py if that threshold changes again,
// rather than assuming this file's numbers are authoritative.
function levelFor(confidence) {
  if (confidence < 0.5) return "low";
  if (confidence < 0.7) return "medium";
  return "high";
}

const LABELS = {
  high: "High relevance",
  medium: "Moderate relevance",
  low: "Below grounding threshold",
};

export default function ConfidenceBadge({ confidence, size = "normal" }) {
  const level = levelFor(confidence);
  const pct = Math.round(confidence * 100);

  return (
    <span
      className={`confidence-badge confidence-badge--${level} confidence-badge--${size}`}
      title={`${LABELS[level]} (${pct}%)`}
    >
      <span className="confidence-badge__dot" />
      {pct}%
    </span>
  );
}

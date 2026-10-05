import "./ConfidenceBadge.css";

// Replaces ConfidenceBadge as the primary per-answer indicator.
//
// WHY: `confidence` (backend QueryResponse.confidence) is the highest raw
// embedding SIMILARITY among retrieved chunks — a retrieval-quality signal,
// not a correctness/completeness signal for the generated answer. A fully
// correct, complete answer can legitimately score 55-60% with the current
// embedding model (bge-small — see embeddings/code_embedder.py), which
// read as "low confidence" / a red badge even when the answer was right.
// Observed directly: a correct 5-route summary of a small FastAPI app
// scored 59-60% and showed a red "Below grounding threshold" badge because
// that badge's own breakpoints (0.65/0.85) were hardcoded against an old
// threshold value that no longer matches the backend's actual
// settings.retrieval_confidence_threshold (recalibrated to 0.5 this
// session — seeing this badge is partly what surfaced that mismatch).
//
// `sources_used` (backend QueryResponse.sources_used — see
// api/routes/query.py._count_sources_used) measures something more directly
// useful: of the sources retrieved and shown, how many did the answer
// actually cite. This can't be gamed by embedding-model score calibration
// the way a raw similarity number can, and answers the question a user
// actually has: "is this answer grounded in what was found, or did it
// mostly ignore the retrieved context?"
function levelFor(used, total) {
  if (total === 0) return "low";
  if (used === total) return "high";
  if (used === 0) return "low";
  return "medium";
}

export default function SourcesUsedBadge({ sourcesUsed, sourcesTotal, size = "normal" }) {
  const used = sourcesUsed ?? 0;
  const total = sourcesTotal ?? 0;
  const level = levelFor(used, total);
  const label =
    total === 0
      ? "No sources retrieved"
      : used === total
        ? `All ${total} sources used`
        : used === 0
          ? `${total} sources retrieved, none cited`
          : `${used} of ${total} sources used`;

  return (
    <span
      className={`confidence-badge confidence-badge--${level} confidence-badge--${size}`}
      title={label}
    >
      <span className="confidence-badge__dot" />
      {total === 0 ? "0 sources" : `${used}/${total} sources`}
    </span>
  );
}

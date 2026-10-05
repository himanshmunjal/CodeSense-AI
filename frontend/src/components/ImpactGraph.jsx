import { useMemo } from "react";
import ReactFlow, { Background, Controls, MarkerType } from "reactflow";
import "reactflow/dist/style.css";
import "./ImpactGraph.css";

// Renders the response of POST /api/v1/impact/ (backend/api/routes/impact.py).
// Real response shape:
//   { function, file_path, direct_callers: string[],
//     blast_radius: [{ function, file_path, distance, risk, start_line, end_line, num_callers }],
//     total_affected, nodes_visited, traversal_ms }
// risk is computed server-side from hop distance: distance 1 -> HIGH,
// distance 2 -> MEDIUM, distance >=3 -> LOW (see impact.py).

const RISK_COLOR = {
  HIGH: "#ef5350",
  MEDIUM: "#e0b93c",
  LOW: "#6b6d73",
};

function layout(target, blastRadius) {
  // Group nodes by distance ring, spread each ring horizontally.
  const byDistance = new Map();
  for (const node of blastRadius) {
    if (!byDistance.has(node.distance)) byDistance.set(node.distance, []);
    byDistance.get(node.distance).push(node);
  }

  const nodes = [
    {
      id: "__target__",
      data: { label: `${target.function}()` },
      position: { x: 0, y: 0 },
      style: {
        background: "#5b8cff",
        color: "white",
        border: "2px solid #3b5aa8",
        borderRadius: 8,
        padding: 8,
        fontWeight: 700,
        fontFamily: "monospace",
        fontSize: 12,
      },
    },
  ];
  const edges = [];

  const distances = [...byDistance.keys()].sort((a, b) => a - b);
  const ringGapY = 130;
  const nodeGapX = 190;

  for (const distance of distances) {
    const ring = byDistance.get(distance);
    const y = distance * ringGapY;
    const totalWidth = (ring.length - 1) * nodeGapX;
    ring.forEach((node, i) => {
      const x = i * nodeGapX - totalWidth / 2;
      const id = `${node.file_path}::${node.function}::${distance}`;
      nodes.push({
        id,
        data: {
          label: (
            <div>
              <div className="impactgraph__node-fn">{node.function}()</div>
              <div className="impactgraph__node-meta">
                {node.file_path.split("/").pop()}:{node.start_line}
              </div>
              <div className="impactgraph__node-meta">{node.num_callers} caller(s)</div>
            </div>
          ),
        },
        position: { x, y },
        style: {
          background: "#26272b",
          color: "#e4e4e7",
          border: `2px solid ${RISK_COLOR[node.risk] || RISK_COLOR.LOW}`,
          borderRadius: 8,
          padding: 6,
          fontSize: 11,
          width: 170,
        },
      });

      // Edge from previous ring toward the target (distance 1 connects
      // directly to the target function itself).
      const source = distance === 1 ? "__target__" : null;
      edges.push({
        id: `edge-${id}`,
        source: source || "__target__", // fallback; refined below when we have parent info
        target: id,
        animated: distance === 1,
        style: { stroke: RISK_COLOR[node.risk] || RISK_COLOR.LOW },
        markerEnd: { type: MarkerType.ArrowClosed, color: RISK_COLOR[node.risk] || RISK_COLOR.LOW },
      });
    });
  }

  // For distance >= 2 we don't have explicit parent edges from the API, so we
  // connect each node to the nearest previous-ring node as a reasonable
  // approximation of the traversal path (the backend gives us ranked
  // blast-radius entries, not an edge list).
  if (distances.length > 1) {
    for (let i = 1; i < distances.length; i++) {
      const prevRing = byDistance.get(distances[i - 1]);
      const ring = byDistance.get(distances[i]);
      ring.forEach((node, idx) => {
        const parent = prevRing[idx % prevRing.length];
        const parentId = `${parent.file_path}::${parent.function}::${distances[i - 1]}`;
        const id = `${node.file_path}::${node.function}::${distances[i]}`;
        const edgeId = `edge-${id}`;
        const existing = edges.find((e) => e.id === edgeId);
        if (existing) existing.source = parentId;
      });
    }
  }

  return { nodes, edges };
}

export default function ImpactGraph({ result, loading, error }) {
  const { nodes, edges } = useMemo(() => {
    if (!result) return { nodes: [], edges: [] };
    return layout(result, result.blast_radius);
  }, [result]);

  if (loading) {
    return <div className="impactgraph impactgraph--empty">Running impact analysis…</div>;
  }

  if (error) {
    return <div className="impactgraph impactgraph--empty impactgraph--error">{error}</div>;
  }

  if (!result) {
    return (
      <div className="impactgraph impactgraph--empty">
        <p>
          No impact analysis yet. Click "Analyze impact" on a source, or ask
          "impact analysis: function_name" in chat.
        </p>
      </div>
    );
  }

  return (
    <div className="impactgraph">
      <div className="impactgraph__summary">
        <div>
          <strong>{result.function}()</strong>
          {result.file_path && <span className="impactgraph__filepath"> — {result.file_path}</span>}
        </div>
        <div className="impactgraph__stats">
          <span>{result.total_affected} affected</span>
          <span>{result.nodes_visited} nodes visited</span>
          <span>{result.traversal_ms?.toFixed?.(1)}ms</span>
        </div>
        <div className="impactgraph__legend">
          <span><i style={{ background: RISK_COLOR.HIGH }} /> HIGH (direct caller)</span>
          <span><i style={{ background: RISK_COLOR.MEDIUM }} /> MEDIUM</span>
          <span><i style={{ background: RISK_COLOR.LOW }} /> LOW</span>
        </div>
      </div>
      <div className="impactgraph__canvas">
        <ReactFlow nodes={nodes} edges={edges} fitView proOptions={{ hideAttribution: true }}>
          <Background color="#38393e" gap={20} />
          <Controls />
        </ReactFlow>
      </div>
    </div>
  );
}

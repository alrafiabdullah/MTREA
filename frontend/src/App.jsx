import { useEffect, useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

function AttentionHeatmap({ sourceTokens, targetTokens, matrix }) {
  const maxVal = useMemo(() => {
    if (!matrix?.length) return 1;
    return Math.max(...matrix.flat(), 1e-6);
  }, [matrix]);

  if (!matrix?.length) {
    return <p className="muted">Attention map hidden.</p>;
  }

  return (
    <div className="heatmap-wrap">
      <table className="heatmap-table">
        <thead>
          <tr>
            <th></th>
            {sourceTokens.map((tok, i) => (
              <th key={`${tok}-${i}`} title={tok}>
                {tok}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {targetTokens.map((tgt, rowIdx) => (
            <tr key={`${tgt}-${rowIdx}`}>
              <th>{tgt}</th>
              {sourceTokens.map((src, colIdx) => {
                const value = matrix[rowIdx]?.[colIdx] ?? 0;
                const alpha = Math.min(0.95, Math.max(0.08, value / maxVal));
                return (
                  <td
                    key={`${src}-${colIdx}`}
                    title={`${tgt} -> ${src}: ${value.toFixed(4)}`}
                    style={{ backgroundColor: `rgba(93, 95, 239, ${alpha})` }}
                  />
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RelationGraph({ entities, relations }) {
  if (!entities?.length) return <p className="muted">No entities detected.</p>;
  if (!relations?.length)
    return <p className="muted">Not enough entities to form relation triples.</p>;

  const uniqNodes = [...new Set(entities.map((e) => e.text))];
  const width = 860;
  const height = 230;
  const nodeX = (i) =>
    Math.round(((i + 1) / (uniqNodes.length + 1)) * (width - 80) + 40);
  const nodeY = Math.round(height / 2);

  const nodeMap = Object.fromEntries(uniqNodes.map((n, i) => [n, { x: nodeX(i), y: nodeY }]));

  return (
    <div className="graph-wrap">
      <svg viewBox={`0 0 ${width} ${height}`} className="relation-svg">
        {relations.map((r, i) => {
          const from = nodeMap[r.head];
          const to = nodeMap[r.tail];
          if (!from || !to) return null;
          const mx = (from.x + to.x) / 2;
          return (
            <g key={`${r.head}-${r.tail}-${i}`}>
              <line x1={from.x} y1={from.y} x2={to.x} y2={to.y} className="edge" />
              <text x={mx} y={from.y - 20} textAnchor="middle" className="edge-label">
                {r.relation}
              </text>
            </g>
          );
        })}
        {uniqNodes.map((n, i) => (
          <g key={`${n}-${i}`}>
            <circle cx={nodeMap[n].x} cy={nodeMap[n].y} r="20" className="node" />
            <text x={nodeMap[n].x} y={nodeMap[n].y + 4} textAnchor="middle" className="node-label">
              {n}
            </text>
          </g>
        ))}
      </svg>
      <div className="triples">
        {relations.map((r, i) => (
          <div key={`${r.head}-${r.tail}-${i}`} className="triple-row">
            ({r.head}, <strong>{r.relation}</strong>, {r.tail})
          </div>
        ))}
      </div>
    </div>
  );
}

function MetricInfo({ label, children }) {
  return (
    <span className="metric-info">
      <button type="button" className="info-btn" aria-label={`Explain ${label}`}>
        i
      </button>
      <span className="info-pop">{children}</span>
    </span>
  );
}

function computeWordDiff(leftText, rightText) {
  const left = leftText.trim() ? leftText.trim().split(/\s+/) : [];
  const right = rightText.trim() ? rightText.trim().split(/\s+/) : [];
  const m = left.length;
  const n = right.length;
  const dp = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0));

  for (let i = m - 1; i >= 0; i -= 1) {
    for (let j = n - 1; j >= 0; j -= 1) {
      if (left[i].toLowerCase() === right[j].toLowerCase()) {
        dp[i][j] = dp[i + 1][j + 1] + 1;
      } else {
        dp[i][j] = Math.max(dp[i + 1][j], dp[i][j + 1]);
      }
    }
  }

  const leftMatch = Array(m).fill(false);
  const rightMatch = Array(n).fill(false);
  let i = 0;
  let j = 0;
  while (i < m && j < n) {
    if (left[i].toLowerCase() === right[j].toLowerCase()) {
      leftMatch[i] = true;
      rightMatch[j] = true;
      i += 1;
      j += 1;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      i += 1;
    } else {
      j += 1;
    }
  }

  return {
    left: left.map((word, idx) => ({ word, match: leftMatch[idx] })),
    right: right.map((word, idx) => ({ word, match: rightMatch[idx] }))
  };
}

function DiffText({ tokens, mismatchClass }) {
  return (
    <p className="translation diff-text">
      {tokens.map((token, idx) => {
        return (
        <span
          key={`${token.word}-${idx}`}
          className={token.match ? "diff-token" : `diff-token ${mismatchClass}`}
        >
          {token.word}
          {idx < tokens.length - 1 ? " " : ""}
        </span>
        );
        })}
    </p>
  );
}

export default function App() {
  const [examples, setExamples] = useState({});
  const [text, setText] = useState("");
  const [showAttention, setShowAttention] = useState(true);
  const [showRelations, setShowRelations] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const diff = useMemo(() => {
    if (!result) return null;
    return computeWordDiff(result.baseline_translation, result.attention_translation);
  }, [result]);
  const showDiff = useMemo(() => {
    if (!diff) return false;
    return [...diff.left, ...diff.right].some((token) => !token.match);
  }, [diff]);

  useEffect(() => {
    fetch(`${API_BASE}/api/examples`)
      .then((r) => r.json())
      .then((data) => {
        setExamples(data);
        setText(data.paper ?? "");
      })
      .catch(() => setError("Failed to load examples. Is backend running?"));
  }, []);

  const runDemo = async () => {
    if (!text.trim()) return;
    setError("");
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/demo`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          show_attention: showAttention,
          show_relations: showRelations
        })
      });
      if (!res.ok) {
        throw new Error("Backend request failed");
      }
      const data = await res.json();
      setResult(data);
    } catch {
      setError("Unable to run demo. Start FastAPI backend and retry.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="page">
      <header className="hero">
        <h1>Machine Translation and Relation Extraction with Attention</h1>
        <p>
          English → German demo to understand encoder-decoder bottlenecks, see
          attention maps, and connect translation to downstream relation extraction.
        </p>
      </header>

      <section className="card controls">
        <label className="input-label" htmlFor="src-text">
          Input sentence
        </label>
        <textarea
          id="src-text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={5}
          placeholder="Enter an English sentence..."
        />
        <div className="examples-row">
          <button onClick={() => setText(examples.short ?? "")}>Short Example</button>
          <button onClick={() => setText(examples.long ?? "")}>Long Example</button>
          <button onClick={() => setText(examples.paper ?? "")}>Paper Example</button>
          <button onClick={() => setText(examples.relation ?? "")}>Relation Example</button>
        </div>

        <div className="toggle-row">
          <label>
            <input
              type="checkbox"
              checked={showAttention}
              onChange={(e) => setShowAttention(e.target.checked)}
            />
            Show Attention
          </label>
          <label>
            <input
              type="checkbox"
              checked={showRelations}
              onChange={(e) => setShowRelations(e.target.checked)}
            />
            Show Relation Extraction
          </label>
        </div>

        <button className="primary" disabled={loading} onClick={runDemo}>
          {loading ? "Running..." : "Translate & Analyze"}
        </button>
      </section>

      {error && <p className="error">{error}</p>}

      {result && (
        <>
          {showDiff && (
            <p className="muted" style={{textAlign: "center"}}>
              <span className="legend-chip mismatch-baseline">Mismatch in baseline</span>{" "}
              <span className="legend-chip mismatch-attention">Mismatch in attention</span>
            </p>
          )}
          <section className="grid-2">
            <article className="card">
              <h2>Without Attention (Fixed-Vector Baseline)</h2>
              <DiffText tokens={diff?.left ?? []} mismatchClass="mismatch-baseline" />
              <p className="muted">
                Uses a single fixed source-context vector without token-level attention. (Simulated)
              </p>
            </article>
            <article className="card">
              <h2>With Attention</h2>
              <DiffText tokens={diff?.right ?? []} mismatchClass="mismatch-attention" />
              <p className="metrics">
                Peak alignment: <strong>{result.metrics.peak_alignment.toFixed(4)}</strong>
                <MetricInfo label="Peak alignment">
                  Average strongest source-token focus per generated token. Higher means
                  more confident, sharper alignments.
                </MetricInfo>{" "}
                | Entropy: <strong>{result.metrics.entropy.toFixed(4)}</strong>
                <MetricInfo label="Entropy">
                  Measures how spread attention is across source tokens. Lower means
                  tighter focus; higher means more diffuse attention.
                </MetricInfo>
              </p>
            </article>
          </section>

          <section className="card">
            <h2>Attention Heatmap</h2>
            <AttentionHeatmap
              sourceTokens={result.source_tokens}
              targetTokens={result.target_tokens}
              matrix={result.attention_matrix}
            />
          </section>

          <section className="card">
            <h2>Relation Extraction</h2>
            {!showRelations ? (
              <p className="muted">Relation extraction hidden.</p>
            ) : (
              <>
                <div className="entities">
                  {result.entities.map((e, i) => (
                    <span key={`${e.text}-${i}`} className="entity-pill">
                      {e.text} <small>{e.label}</small>
                    </span>
                  ))}
                </div>
                <RelationGraph entities={result.entities} relations={result.relations} />
              </>
            )}
          </section>
        </>
      )}
    </main>
  );
}

import { startTransition, useEffect, useRef, useState } from "react";

type PlaybackData = {
  metadata: {
    generatedAt: string;
    source: string;
    epochCount: number;
    nodeCount: number;
    edgeCount: number;
  };
  epochs: number[];
  positions: [number, number][];
  edgeIndex: [number[], number[]];
  target: number[];
  predictions: number[][];
  loss: number[];
  learningRate: number[];
  meanRisk: number[];
  p95Risk: number[];
  maxRisk: number[];
  tailThreshold: number;
};

type Speed = "slow" | "medium" | "fast";

const SPEEDS: Record<Speed, number> = {
  slow: 180,
  medium: 90,
  fast: 40,
};

function colorForRisk(risk: number): string {
  const clamped = Math.max(0, Math.min(1, risk));
  const hue = (1 - clamped) * 110;
  const lightness = 40 + clamped * 10;
  return `hsl(${hue}, 72%, ${lightness}%)`;
}

function formatNumber(value: number, digits = 3): string {
  return value.toFixed(digits);
}

function MetricCard(props: { label: string; value: string; note: string }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{props.label}</div>
      <div className="metric-value">{props.value}</div>
      <div className="metric-note">{props.note}</div>
    </div>
  );
}

function MiniLineChart(props: {
  title: string;
  valuesA: number[];
  colorA: string;
  valuesB?: number[];
  colorB?: string;
  currentIndex: number;
  labels?: [string, string];
}) {
  const width = 360;
  const height = 190;
  const padding = 20;
  const allValues = props.valuesB ? props.valuesA.concat(props.valuesB) : props.valuesA;
  const min = Math.min(...allValues);
  const max = Math.max(...allValues);
  const span = Math.max(max - min, 1e-6);

  function toPath(values: number[]) {
    return values
      .map((value, index) => {
        const x = padding + (index / Math.max(values.length - 1, 1)) * (width - padding * 2);
        const y = height - padding - ((value - min) / span) * (height - padding * 2);
        return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(" ");
  }

  const currentX = padding + (props.currentIndex / Math.max(props.valuesA.length - 1, 1)) * (width - padding * 2);
  const currentAY = height - padding - ((props.valuesA[props.currentIndex] - min) / span) * (height - padding * 2);
  const currentBY =
    props.valuesB && props.colorB
      ? height - padding - ((props.valuesB[props.currentIndex] - min) / span) * (height - padding * 2)
      : null;

  return (
    <div className="chart-card">
      <div className="chart-header">
        <h3>{props.title}</h3>
        <div className="chart-legend">
          <span style={{ ["--swatch" as string]: props.colorA }}>{props.labels?.[0] ?? "Series A"}</span>
          {props.valuesB && props.colorB ? (
            <span style={{ ["--swatch" as string]: props.colorB }}>{props.labels?.[1] ?? "Series B"}</span>
          ) : null}
        </div>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="chart-svg" role="img" aria-label={props.title}>
        <rect x="0" y="0" width={width} height={height} rx="18" fill="#fffdf8" />
        <path d={toPath(props.valuesA)} fill="none" stroke={props.colorA} strokeWidth="4" strokeLinecap="round" />
        {props.valuesB && props.colorB ? (
          <path d={toPath(props.valuesB)} fill="none" stroke={props.colorB} strokeWidth="4" strokeLinecap="round" />
        ) : null}
        <line x1={currentX} x2={currentX} y1={padding} y2={height - padding} stroke="#ea580c" strokeDasharray="4 5" />
        <circle cx={currentX} cy={currentAY} r="6" fill={props.colorA} />
        {currentBY !== null && props.colorB ? <circle cx={currentX} cy={currentBY} r="6" fill={props.colorB} /> : null}
      </svg>
    </div>
  );
}

function GraphCanvas(props: {
  data: PlaybackData;
  currentIndex: number;
  showEdges: boolean;
  highlightTargets: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }

    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.fillStyle = "#fffdf8";
    ctx.fillRect(0, 0, rect.width, rect.height);

    const padding = 28;
    const xs = props.data.positions.map((point) => point[1]);
    const ys = props.data.positions.map((point) => point[0]);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    const projectX = (lon: number) => padding + ((lon - minX) / Math.max(maxX - minX, 1e-6)) * (rect.width - padding * 2);
    const projectY = (lat: number) => rect.height - padding - ((lat - minY) / Math.max(maxY - minY, 1e-6)) * (rect.height - padding * 2);

    const prediction = props.data.predictions[props.currentIndex];
    const tailThreshold = props.data.tailThreshold;

    if (props.showEdges) {
      ctx.strokeStyle = "rgba(23, 52, 47, 0.08)";
      ctx.lineWidth = 1;
      const [srcList, dstList] = props.data.edgeIndex;
      ctx.beginPath();
      for (let i = 0; i < srcList.length; i += 1) {
        const src = srcList[i];
        const dst = dstList[i];
        const srcPoint = props.data.positions[src];
        const dstPoint = props.data.positions[dst];
        ctx.moveTo(projectX(srcPoint[1]), projectY(srcPoint[0]));
        ctx.lineTo(projectX(dstPoint[1]), projectY(dstPoint[0]));
      }
      ctx.stroke();
    }

    prediction.forEach((risk, index) => {
      const point = props.data.positions[index];
      const x = projectX(point[1]);
      const y = projectY(point[0]);
      const radius = 2 + risk * 4.2;

      ctx.fillStyle = colorForRisk(risk);
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();

      if (props.highlightTargets && props.data.target[index] >= tailThreshold) {
        ctx.strokeStyle = risk < tailThreshold ? "rgba(15, 118, 110, 0.8)" : "rgba(23, 52, 47, 0.7)";
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.arc(x, y, radius + 2.4, 0, Math.PI * 2);
        ctx.stroke();
      }
    });

    ctx.fillStyle = "#17342f";
    ctx.font = "700 18px 'Avenir Next', 'Segoe UI', sans-serif";
    ctx.fillText(`Node Risk Field · Epoch ${props.data.epochs[props.currentIndex]}`, 24, 28);
  }, [props.currentIndex, props.data, props.highlightTargets, props.showEdges]);

  return <canvas ref={canvasRef} className="graph-canvas" />;
}

export default function App() {
  const [data, setData] = useState<PlaybackData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<Speed>("medium");
  const [showEdges, setShowEdges] = useState(true);
  const [highlightTargets, setHighlightTargets] = useState(true);

  useEffect(() => {
    let cancelled = false;
    const playbackDataUrl = `${import.meta.env.BASE_URL}data/gnn_training_history.json`;

    fetch(playbackDataUrl)
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const contentType = response.headers.get("content-type") ?? "";
        if (!contentType.includes("application/json")) {
          const body = await response.text();
          if (body.trimStart().startsWith("<")) {
            throw new Error(
              "Playback data export is missing. Run `.venv/bin/python scripts/export_gnn_playback_data.py` from the repo root, then reload the app."
            );
          }
          throw new Error(`Unexpected response format from ${playbackDataUrl}`);
        }
        return response.json() as Promise<PlaybackData>;
      })
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setData(payload);
        setCurrentIndex(Math.max(payload.epochs.length - 1, 0));
      })
      .catch((loadError) => {
        if (!cancelled) {
          setError(`Could not load playback data: ${String(loadError)}`);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!playing || !data) {
      return;
    }
    const handle = window.setTimeout(() => {
      startTransition(() => {
        setCurrentIndex((previous) => {
          if (previous >= data.epochs.length - 1) {
            setPlaying(false);
            return previous;
          }
          return previous + 1;
        });
      });
    }, SPEEDS[speed]);
    return () => window.clearTimeout(handle);
  }, [data, playing, speed, currentIndex]);

  const deferredIndex = currentIndex;

  const currentPrediction = data ? data.predictions[deferredIndex] : null;
  const finalPrediction = data ? data.predictions[data.predictions.length - 1] : null;
  const tailMask = data ? data.target.map((value) => value >= data.tailThreshold) : [];
  const finalNeutralized =
    data && finalPrediction
      ? tailMask.reduce((count, flagged, index) => {
          return count + (flagged && finalPrediction[index] < data.tailThreshold ? 1 : 0);
        }, 0)
      : 0;

  if (error) {
    return <div className="app-shell"><div className="error-panel">{error}</div></div>;
  }

  if (!data || !currentPrediction) {
    return <div className="app-shell"><div className="loading-panel">Loading playback data…</div></div>;
  }

  const neutralizedNow = tailMask.reduce((count, flagged, index) => {
    return count + (flagged && currentPrediction[index] < data.tailThreshold ? 1 : 0);
  }, 0);

  return (
    <div className="app-shell">
      <header className="hero">
        <div className="hero-kicker">Dedicated React animation</div>
        <h1>SenTree GNN Playback</h1>
        <p>
          A browser-native animation surface for the training process. Scrub epochs, inspect risk propagation,
          and keep the Streamlit dashboard focused on the rest of the product.
        </p>
        <div className="hero-badges">
          <span>Epochs: {data.metadata.epochCount}</span>
          <span>Nodes: {data.metadata.nodeCount}</span>
          <span>Edges: {data.metadata.edgeCount}</span>
          <span>Exported: {new Date(data.metadata.generatedAt).toLocaleString()}</span>
        </div>
      </header>

      <section className="metrics-grid">
        <MetricCard label="Current epoch" value={`${data.epochs[deferredIndex]}`} note="Training frame in focus" />
        <MetricCard label="Current loss" value={formatNumber(data.loss[deferredIndex], 4)} note="Huber objective" />
        <MetricCard label="Mean node risk" value={formatNumber(data.meanRisk[deferredIndex])} note="Average predicted exposure" />
        <MetricCard label="Neutralized now" value={`${neutralizedNow}`} note={`of ${tailMask.filter(Boolean).length} tail-risk targets`} />
        <MetricCard label="Final neutralized" value={`${finalNeutralized}`} note="At last training epoch" />
      </section>

      <section className="control-bar">
        <div className="epoch-control">
          <label htmlFor="epoch">Epoch</label>
          <input
            id="epoch"
            type="range"
            min={0}
            max={data.epochs.length - 1}
            value={currentIndex}
            onChange={(event) => {
              const nextIndex = Number(event.target.value);
              startTransition(() => setCurrentIndex(nextIndex));
            }}
          />
          <div className="epoch-labels">
            <span>1</span>
            <span>{data.epochs[data.epochs.length - 1]}</span>
          </div>
        </div>

        <div className="button-row">
          <button type="button" onClick={() => setPlaying((value) => !value)}>{playing ? "Pause" : "Play"}</button>
          <button
            type="button"
            onClick={() => {
              setPlaying(false);
              setCurrentIndex(0);
            }}
            className="ghost"
          >
            Reset
          </button>
        </div>

        <label className="inline-control">
          Speed
          <select value={speed} onChange={(event) => setSpeed(event.target.value as Speed)}>
            <option value="slow">Slow</option>
            <option value="medium">Medium</option>
            <option value="fast">Fast</option>
          </select>
        </label>

        <label className="toggle">
          <input type="checkbox" checked={showEdges} onChange={(event) => setShowEdges(event.target.checked)} />
          <span>Show graph</span>
        </label>

        <label className="toggle">
          <input type="checkbox" checked={highlightTargets} onChange={(event) => setHighlightTargets(event.target.checked)} />
          <span>Highlight tail-risk targets</span>
        </label>
      </section>

      <section className="playback-layout">
        <div className="graph-panel">
          <GraphCanvas
            data={data}
            currentIndex={deferredIndex}
            showEdges={showEdges}
            highlightTargets={highlightTargets}
          />
          <div className="legend">
            <div className="legend-bar" />
            <div className="legend-labels">
              <span>Lower risk</span>
              <span>Higher risk</span>
            </div>
          </div>
        </div>

        <div className="sidebar-panel">
          <MiniLineChart title="Optimization Progress" valuesA={data.loss} colorA="#0f766e" currentIndex={deferredIndex} labels={["Loss", ""]} />
          <MiniLineChart
            title="Prediction Profile"
            valuesA={data.meanRisk}
            colorA="#2563eb"
            valuesB={data.p95Risk}
            colorB="#b91c1c"
            currentIndex={deferredIndex}
            labels={["Mean risk", "95th pct risk"]}
          />
        </div>
      </section>
    </div>
  );
}

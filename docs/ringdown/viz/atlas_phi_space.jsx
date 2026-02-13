import React, { useState, useMemo } from "react";

const ATLAS_SUMMARY = {
  kerr_220: [
    { a: 0.00, f: 194.7, Q: 2.100 }, { a: 0.10, f: 201.6, Q: 2.181 },
    { a: 0.20, f: 209.5, Q: 2.277 }, { a: 0.30, f: 218.6, Q: 2.391 },
    { a: 0.40, f: 229.2, Q: 2.531 }, { a: 0.50, f: 241.8, Q: 2.710 },
    { a: 0.55, f: 249.2, Q: 2.820 }, { a: 0.60, f: 257.4, Q: 2.949 },
    { a: 0.625, f: 261.9, Q: 3.023 }, { a: 0.65, f: 266.8, Q: 3.104 },
    { a: 0.675, f: 271.9, Q: 3.195 }, { a: 0.70, f: 277.5, Q: 3.296 },
    { a: 0.725, f: 283.5, Q: 3.411 }, { a: 0.75, f: 290.1, Q: 3.542 },
    { a: 0.80, f: 304.7, Q: 3.859 }, { a: 0.85, f: 322.2, Q: 4.273 },
    { a: 0.90, f: 344.2, Q: 4.872 }, { a: 0.95, f: 376.5, Q: 5.891 },
    { a: 0.99, f: 429.9, Q: 8.750 },
  ],
  observed: { f: 283.43, Q: 1.945, label: "Hilbert estimate" },
  true_kerr: { f: 271.93, Q: 3.195, a: 0.675, label: "Kerr a=0.675 (GR truth)" },
  injected: { f: 251.0, Q: 3.15, label: "Injected signal (f₀=251, τ=4ms)" },
};

function toLog(f, Q) {
  return { x: Math.log(f), y: Math.log(Q) };
}

export default function AtlasVisualization() {
  const [epsilon, setEpsilon] = useState(0.3);
  const [showBeyondKerr, setShowBeyondKerr] = useState(true);

  const bounds = useMemo(() => {
    const allF = ATLAS_SUMMARY.kerr_220.map(e => Math.log(e.f));
    const allQ = ATLAS_SUMMARY.kerr_220.map(e => Math.log(e.Q));
    return {
      xMin: Math.min(...allF) - 0.15,
      xMax: Math.max(...allF) + 0.15,
      yMin: Math.min(...allQ) - 0.8,
      yMax: Math.max(...allQ) + 0.4,
    };
  }, []);

  const W = 700, H = 500;
  const pad = { top: 40, right: 30, bottom: 50, left: 60 };
  const pw = W - pad.left - pad.right;
  const ph = H - pad.top - pad.bottom;

  const sx = (v) => pad.left + ((v - bounds.xMin) / (bounds.xMax - bounds.xMin)) * pw;
  const sy = (v) => pad.top + ph - ((v - bounds.yMin) / (bounds.yMax - bounds.yMin)) * ph;

  const obs = toLog(ATLAS_SUMMARY.observed.f, ATLAS_SUMMARY.observed.Q);
  const truth = toLog(ATLAS_SUMMARY.true_kerr.f, ATLAS_SUMMARY.true_kerr.Q);
  const injected = toLog(ATLAS_SUMMARY.injected.f, ATLAS_SUMMARY.injected.Q);

  const kerrPoints = ATLAS_SUMMARY.kerr_220.map(e => ({
    ...toLog(e.f, e.Q), a: e.a, f: e.f, Q: e.Q,
    d: Math.sqrt((obs.x - Math.log(e.f)) ** 2 + (obs.y - Math.log(e.Q)) ** 2),
  }));

  const nCompatKerr = kerrPoints.filter(p => p.d <= epsilon).length;
  const nTotal = 230;
  
  const beyondKerrGrid = [];
  if (showBeyondKerr) {
    const repSpins = [0.00, 0.30, 0.50, 0.675, 0.80, 0.95];
    const dfs = [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20];
    const dQs = [-0.20, -0.10, 0.10, 0.20];
    for (const a of repSpins) {
      const base = ATLAS_SUMMARY.kerr_220.find(e => Math.abs(e.a - a) < 0.01);
      if (!base) continue;
      for (const df of dfs) {
        for (const dq of dQs) {
          const fNew = base.f * (1 + df);
          const QNew = base.Q * (1 + dq);
          const pt = toLog(fNew, QNew);
          const d = Math.sqrt((obs.x - pt.x) ** 2 + (obs.y - pt.y) ** 2);
          beyondKerrGrid.push({ ...pt, d, df, dq, a });
        }
      }
    }
  }

  const nCompatBK = beyondKerrGrid.filter(p => p.d <= epsilon).length;
  const nCompatTotal = nCompatKerr + nCompatBK;
  const bitsExcl = nCompatTotal > 0 ? Math.log2(nTotal / nCompatTotal) : Math.log2(nTotal);

  const kerrPath = kerrPoints.map((p, i) =>
    `${i === 0 ? "M" : "L"} ${sx(p.x).toFixed(1)} ${sy(p.y).toFixed(1)}`
  ).join(" ");

  const epsPixelsX = (epsilon / (bounds.xMax - bounds.xMin)) * pw;
  const epsPixelsY = (epsilon / (bounds.yMax - bounds.yMin)) * ph;

  const xTicks = [5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1];
  const yTicks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2];

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-6 flex flex-col items-center">
      <h1 className="text-2xl font-bold mb-1 text-amber-400">Atlas Real QNM — Φ-Space</h1>
      <p className="text-sm text-gray-400 mb-4">
        230 entries (Kerr + beyond-Kerr) · qnm package (Leaver method) · M_f=62 M☉
      </p>

      <div className="flex gap-6 mb-4 items-center text-sm">
        <label className="flex items-center gap-2">
          <span className="text-gray-400">ε =</span>
          <input type="range" min="0.05" max="1.5" step="0.05"
            value={epsilon} onChange={e => setEpsilon(+e.target.value)}
            className="w-40 accent-amber-500" />
          <span className="font-mono text-amber-400 w-12">{epsilon.toFixed(2)}</span>
        </label>
        <label className="flex items-center gap-2 cursor-pointer">
          <input type="checkbox" checked={showBeyondKerr}
            onChange={e => setShowBeyondKerr(e.target.checked)}
            className="accent-rose-500" />
          <span className="text-gray-400">Beyond-Kerr</span>
        </label>
      </div>

      <div className="bg-gray-900 rounded-lg border border-gray-800 p-2 mb-4">
        <svg width={W} height={H} className="block">
          {/* Grid */}
          {xTicks.filter(t => t >= bounds.xMin && t <= bounds.xMax).map(t => (
            <g key={`x${t}`}>
              <line x1={sx(t)} y1={pad.top} x2={sx(t)} y2={H - pad.bottom}
                stroke="#1f2937" strokeWidth={1} />
              <text x={sx(t)} y={H - pad.bottom + 16} textAnchor="middle"
                fill="#6b7280" fontSize={10}>
                {Math.round(Math.exp(t))}
              </text>
            </g>
          ))}
          {yTicks.filter(t => t >= bounds.yMin && t <= bounds.yMax).map(t => (
            <g key={`y${t}`}>
              <line x1={pad.left} y1={sy(t)} x2={W - pad.right} y2={sy(t)}
                stroke="#1f2937" strokeWidth={1} />
              <text x={pad.left - 8} y={sy(t) + 3} textAnchor="end"
                fill="#6b7280" fontSize={10}>
                {Math.exp(t).toFixed(1)}
              </text>
            </g>
          ))}

          {/* Axis labels */}
          <text x={W / 2} y={H - 5} textAnchor="middle" fill="#9ca3af" fontSize={12}>
            f (Hz)
          </text>
          <text x={14} y={H / 2} textAnchor="middle" fill="#9ca3af" fontSize={12}
            transform={`rotate(-90, 14, ${H / 2})`}>
            Q
          </text>

          {/* Epsilon circle around observed */}
          <ellipse cx={sx(obs.x)} cy={sy(obs.y)}
            rx={epsPixelsX} ry={epsPixelsY}
            fill="rgba(251, 191, 36, 0.06)" stroke="#fbbf24" strokeWidth={1}
            strokeDasharray="4 3" />

          {/* Beyond-Kerr points */}
          {beyondKerrGrid.map((p, i) => (
            <circle key={`bk${i}`} cx={sx(p.x)} cy={sy(p.y)} r={3}
              fill={p.d <= epsilon ? "#f43f5e" : "#374151"}
              opacity={p.d <= epsilon ? 0.7 : 0.3} />
          ))}

          {/* Kerr curve */}
          <path d={kerrPath} fill="none" stroke="#3b82f6" strokeWidth={2} opacity={0.7} />

          {/* Kerr points */}
          {kerrPoints.map((p, i) => (
            <circle key={`k${i}`} cx={sx(p.x)} cy={sy(p.y)} r={4}
              fill={p.d <= epsilon ? "#60a5fa" : "#1e40af"}
              stroke={Math.abs(p.a - 0.675) < 0.01 ? "#22d3ee" : "none"}
              strokeWidth={Math.abs(p.a - 0.675) < 0.01 ? 2 : 0} />
          ))}

          {/* Spin labels on some Kerr points */}
          {kerrPoints.filter(p => [0.0, 0.30, 0.50, 0.675, 0.80, 0.95].some(a => Math.abs(p.a - a) < 0.01))
            .map((p, i) => (
              <text key={`al${i}`} x={sx(p.x) + 6} y={sy(p.y) - 6}
                fill="#60a5fa" fontSize={9} opacity={0.8}>
                a={p.a.toFixed(2)}
              </text>
            ))}

          {/* Bias arrow: observed → true */}
          <line x1={sx(obs.x)} y1={sy(obs.y)} x2={sx(truth.x)} y2={sy(truth.y)}
            stroke="#ef4444" strokeWidth={1.5} strokeDasharray="5 3"
            markerEnd="url(#arrowRed)" />
          <defs>
            <marker id="arrowRed" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
              <path d="M0,0 L8,3 L0,6" fill="none" stroke="#ef4444" strokeWidth="1.5" />
            </marker>
          </defs>

          {/* Injected signal */}
          <circle cx={sx(injected.x)} cy={sy(injected.y)} r={6}
            fill="none" stroke="#a78bfa" strokeWidth={2} strokeDasharray="3 2" />
          <text x={sx(injected.x) - 8} y={sy(injected.y) - 10}
            fill="#a78bfa" fontSize={10} textAnchor="end">
            Injected
          </text>

          {/* True Kerr */}
          <circle cx={sx(truth.x)} cy={sy(truth.y)} r={7}
            fill="none" stroke="#22d3ee" strokeWidth={2.5} />
          <text x={sx(truth.x) + 10} y={sy(truth.y) + 4}
            fill="#22d3ee" fontSize={10}>
            GR truth
          </text>

          {/* Observed */}
          <circle cx={sx(obs.x)} cy={sy(obs.y)} r={6}
            fill="#fbbf24" stroke="#fbbf24" strokeWidth={2} />
          <text x={sx(obs.x) + 10} y={sy(obs.y) + 4}
            fill="#fbbf24" fontSize={10} fontWeight="bold">
            Observed
          </text>

          {/* Bias label */}
          <text x={(sx(obs.x) + sx(truth.x)) / 2 + 10}
            y={(sy(obs.y) + sy(truth.y)) / 2}
            fill="#ef4444" fontSize={10} fontStyle="italic">
            estimator bias
          </text>
        </svg>
      </div>

      {/* Stats panel */}
      <div className="grid grid-cols-4 gap-3 text-center mb-4">
        <div className="bg-gray-900 rounded-lg p-3 border border-gray-800">
          <div className="text-2xl font-bold text-amber-400">{nCompatTotal}</div>
          <div className="text-xs text-gray-500">compatible / {nTotal}</div>
        </div>
        <div className="bg-gray-900 rounded-lg p-3 border border-gray-800">
          <div className="text-2xl font-bold text-sky-400">{bitsExcl.toFixed(2)}</div>
          <div className="text-xs text-gray-500">bits excluded</div>
        </div>
        <div className="bg-gray-900 rounded-lg p-3 border border-gray-800">
          <div className="text-2xl font-bold text-cyan-400">{nCompatKerr}</div>
          <div className="text-xs text-gray-500">Kerr GR in set</div>
        </div>
        <div className="bg-gray-900 rounded-lg p-3 border border-gray-800">
          <div className="text-2xl font-bold text-rose-400">{nCompatBK}</div>
          <div className="text-xs text-gray-500">beyond-Kerr in set</div>
        </div>
      </div>

      <div className="max-w-2xl text-sm space-y-2 text-gray-400">
        <p>
          <span className="text-amber-400 font-semibold">Diagnóstico clave:</span>{" "}
          El punto observado (amarillo) está lejos de la curva Kerr GR (azul) porque el
          estimador Hilbert subestima Q en un ~39%. La distancia al verdadero Kerr (a=0.675)
          es d=0.498 — casi toda en la dirección vertical (log Q).
        </p>
        <p>
          <span className="text-rose-400 font-semibold">Consecuencia:</span>{" "}
          {nCompatKerr === 0
            ? "Ninguna geometría Kerr GR está en el conjunto compatible. Solo entran deviaciones beyond-Kerr. Un estimador sesgado hace que GR parezca gravedad modificada."
            : `${nCompatKerr} geometrías Kerr en el set compatible.`}
        </p>
        <p>
          <span className="text-cyan-400 font-semibold">Implicación:</span>{" "}
          Confirma que el gate de invariancia temporal es prerrequisito irreducible.
          Sin estimador limpio, el atlas real amplifica errores sistemáticos en vez de resolverlos.
        </p>
      </div>
    </div>
  );
}

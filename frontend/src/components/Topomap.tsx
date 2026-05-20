import type { TopomapPayload } from "../lib/api";

// Diverging RdBu_r palette (low=blue, mid=white, high=red) sampled at 9 stops.
const STOPS = [
  "#053061","#2166ac","#4393c3","#92c5de","#f7f7f7","#f4a582","#d6604d","#b2182b","#67001f",
];

function lerpColor(t: number): string {
  const x = Math.max(0, Math.min(1, t)) * (STOPS.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  if (i >= STOPS.length - 1) return STOPS[STOPS.length - 1];
  const a = hex(STOPS[i]);
  const b = hex(STOPS[i + 1]);
  const r = Math.round(a[0] + (b[0] - a[0]) * f);
  const g = Math.round(a[1] + (b[1] - a[1]) * f);
  const bl = Math.round(a[2] + (b[2] - a[2]) * f);
  return `rgb(${r},${g},${bl})`;
}
function hex(h: string): [number, number, number] {
  const v = h.replace("#", "");
  return [parseInt(v.slice(0, 2), 16), parseInt(v.slice(2, 4), 16), parseInt(v.slice(4, 6), 16)];
}

export function Topomap({ payload, labelOnly = false }: { payload: TopomapPayload; labelOnly?: boolean }) {
  const { points, vmin, vmax } = payload;
  const range = Math.max(1e-6, vmax - vmin);
  return (
    <div className="relative w-full max-w-[360px] aspect-square mx-auto">
      <svg viewBox="-1.2 -1.2 2.4 2.4" className="w-full h-full">
        {/* Head outline */}
        <circle cx="0" cy="0" r="1" fill="none" stroke="rgb(148,163,184)" strokeWidth="0.02" />
        {/* Nose */}
        <path d="M -0.1 1 L 0 1.1 L 0.1 1" fill="none" stroke="rgb(148,163,184)" strokeWidth="0.02" />
        {/* Ears */}
        <ellipse cx="-1" cy="0" rx="0.06" ry="0.18" fill="none" stroke="rgb(148,163,184)" strokeWidth="0.02" />
        <ellipse cx="1"  cy="0" rx="0.06" ry="0.18" fill="none" stroke="rgb(148,163,184)" strokeWidth="0.02" />

        {points.map((p) => {
          const t = labelOnly ? 0.5 : (p.value - vmin) / range;
          return (
            <g key={p.channel}>
              <circle cx={p.x} cy={p.y} r="0.11"
                fill={labelOnly ? "rgb(241,245,249)" : lerpColor(t)}
                stroke="rgb(15,23,42)" strokeWidth="0.012" />
              <text x={p.x} y={p.y + 0.025} fontSize="0.085" textAnchor="middle"
                fill={labelOnly || (t > 0.2 && t < 0.8) ? "rgb(15,23,42)" : "white"}
                fontFamily="ui-sans-serif" fontWeight="500">
                {p.channel}
              </text>
            </g>
          );
        })}
      </svg>
      {!labelOnly && (
        <div className="absolute bottom-1 left-0 right-0 flex items-center gap-2 text-xs text-ink-500 px-2">
          <span>{vmin.toFixed(1)}</span>
          <div className="flex-1 h-2 rounded"
            style={{
              background: `linear-gradient(to right, ${STOPS.map((c, i) => `${c} ${(i / (STOPS.length - 1)) * 100}%`).join(", ")})`,
            }}
          />
          <span>{vmax.toFixed(1)}</span>
        </div>
      )}
    </div>
  );
}

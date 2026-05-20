import {
  LineChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid, ErrorBar,
} from "recharts";
import { fmtDate } from "../lib/format";
import type { LongitudinalPoint } from "../lib/api";

export function LongitudinalChart({ points }: { points: LongitudinalPoint[] }) {
  const data = points
    .filter((p) => p.mes_mean != null)
    .map((p, i) => ({
      idx: i + 1,
      mes: p.mes_mean!,
      err: p.mes_std ?? 0,
      label: fmtDate(p.created_at),
    }));

  if (data.length === 0) {
    return (
      <div className="text-center text-ink-500 text-sm py-10">
        No scored sessions yet. Upload a recording to start the longitudinal series.
      </div>
    );
  }

  return (
    <div style={{ width: "100%", height: 260 }}>
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 12, right: 16, bottom: 0, left: -8 }}>
          <CartesianGrid stroke="rgba(148,163,184,0.2)" />
          <XAxis dataKey="idx" stroke="rgb(100,116,139)" fontSize={11} />
          <YAxis domain={[0, 100]} stroke="rgb(100,116,139)" fontSize={11} />
          <Tooltip
            contentStyle={{ background: "rgba(15,23,42,0.96)", color: "#fff", border: "none", borderRadius: 8, fontSize: 12 }}
            formatter={(v: number) => v.toFixed(1)}
            labelFormatter={(_, payload) => (payload as any)?.[0]?.payload?.label ?? ""}
          />
          <Line type="monotone" dataKey="mes" stroke="#0d9488" strokeWidth={2.5}
            dot={{ r: 4, stroke: "#0d9488", fill: "white", strokeWidth: 2 }}
            animationDuration={900}>
            <ErrorBar dataKey="err" stroke="#0d9488" width={4} />
          </Line>
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

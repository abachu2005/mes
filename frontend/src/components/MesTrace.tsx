import {
  LineChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid,
} from "recharts";

export function MesTrace({ values, compact = false }: { values: number[]; compact?: boolean }) {
  const data = values.map((v, i) => ({ trial: i + 1, mes: v }));
  return (
    <div style={{ width: "100%", height: compact ? 140 : 220 }}>
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 10, right: 16, bottom: 0, left: -10 }}>
          <CartesianGrid stroke="rgba(148,163,184,0.2)" />
          <XAxis dataKey="trial" stroke="rgb(100,116,139)" fontSize={11} />
          <YAxis domain={[0, 100]} stroke="rgb(100,116,139)" fontSize={11} />
          <ReferenceLine y={50} stroke="rgba(148,163,184,0.5)" strokeDasharray="3 3" />
          <Tooltip
            contentStyle={{ background: "rgba(15,23,42,0.96)", color: "#fff", border: "none", borderRadius: 8, fontSize: 12 }}
            formatter={(v: number) => v.toFixed(1)}
            labelFormatter={(t) => `Trial ${t}`}
          />
          <Line type="monotone" dataKey="mes" stroke="#0d9488" strokeWidth={2.5}
            dot={{ r: 3, stroke: "#0d9488", fill: "white" }} activeDot={{ r: 5 }}
            animationDuration={900} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

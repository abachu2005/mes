import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { MesGauge } from "../components/MesGauge";
import { Topomap } from "../components/Topomap";
import { MesTrace } from "../components/MesTrace";
import { GitBranch } from "lucide-react";

export function Comparison() {
  const sessions = useQuery({ queryKey: ["all-sessions"], queryFn: () => api.listSessions() });
  const done = useMemo(() => (sessions.data ?? []).filter((s) => s.status === "done"), [sessions.data]);

  const [aId, setA] = useState<string>("");
  const [bId, setB] = useState<string>("");

  const a = useQuery({ queryKey: ["score", aId], queryFn: () => api.getScore(aId), enabled: !!aId });
  const b = useQuery({ queryKey: ["score", bId], queryFn: () => api.getScore(bId), enabled: !!bId });

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 space-y-6">
      <div>
        <h1 className="text-3xl font-semibold">Side-by-side comparison</h1>
        <p className="text-ink-500 text-sm mt-1">
          Use this to demo healthy-vs-stroke or pre-vs-post recovery sessions.
        </p>
      </div>

      <div className="grid sm:grid-cols-2 gap-4">
        <SessionPicker label="Session A" sessions={done} value={aId} onChange={setA} />
        <SessionPicker label="Session B" sessions={done} value={bId} onChange={setB} />
      </div>

      {!aId || !bId ? (
        <div className="card p-10 text-center text-ink-500">
          <GitBranch className="w-10 h-10 mx-auto" />
          <div className="mt-2">Pick two completed sessions to compare them side-by-side.</div>
        </div>
      ) : (
        <div className="grid lg:grid-cols-2 gap-6">
          {[a, b].map((q, idx) => (
            <div key={idx} className="card p-6">
              <div className="text-sm text-ink-500 uppercase tracking-wider">
                Session {idx === 0 ? "A" : "B"}
              </div>
              {!q.data ? <div className="text-ink-500">Loading…</div> : (
                <>
                  <MesGauge value={q.data.mes_mean} />
                  <div className="text-center text-ink-500 text-xs mt-1">
                    median {q.data.mes_median.toFixed(1)} · std {q.data.mes_std.toFixed(1)} · {q.data.n_trials} trials
                  </div>
                  <div className="mt-4">
                    <Topomap payload={q.data.erd_topomap} />
                  </div>
                  <div className="mt-4">
                    <MesTrace values={q.data.mes_per_trial} compact />
                  </div>
                </>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function SessionPicker({
  label,
  sessions,
  value,
  onChange,
}: {
  label: string;
  sessions: any[];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <label className="block text-sm">
      <div className="font-medium mb-1">{label}</div>
      <select className="input" value={value} onChange={(e) => onChange(e.target.value)}>
        <option value="">— pick a session —</option>
        {sessions.map((s) => (
          <option key={s.id} value={s.id}>
            {s.participant_id.slice(0, 6)} · {s.task} · {new Date(s.created_at).toLocaleDateString()}
          </option>
        ))}
      </select>
    </label>
  );
}

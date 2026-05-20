import { Link, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Upload, ArrowLeft, FileText } from "lucide-react";
import { api } from "../lib/api";
import { fmtDate, mesBand } from "../lib/format";
import { LongitudinalChart } from "../components/LongitudinalChart";

export function ParticipantDetail() {
  const { id } = useParams<{ id: string }>();
  const long = useQuery({
    queryKey: ["longitudinal", id],
    queryFn: () => api.longitudinal(id!),
    enabled: !!id,
  });
  const sessions = useQuery({
    queryKey: ["sessions-for", id],
    queryFn: () => api.listSessions(id),
    enabled: !!id,
    refetchInterval: (q) => {
      const data = (q.state.data as any[]) || [];
      return data.some((s) => s.status === "queued" || s.status === "processing") ? 2000 : false;
    },
  });

  if (long.isLoading) return <Loading />;
  if (long.error || !long.data) return <Empty msg="Participant not found" />;
  const { participant, points } = long.data;

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 space-y-6">
      <Link to="/dashboard" className="btn-ghost text-sm w-fit -ml-3">
        <ArrowLeft className="w-4 h-4" /> All participants
      </Link>
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <div className="text-xs text-ink-500 uppercase tracking-wider">Participant</div>
          <h1 className="text-3xl font-mono font-semibold">{participant.code}</h1>
          {participant.notes && <p className="text-ink-500 text-sm mt-1 max-w-2xl">{participant.notes}</p>}
        </div>
        <Link to={`/participants/${participant.id}/upload`} className="btn-primary">
          <Upload className="w-4 h-4" /> Upload session
        </Link>
      </div>

      <div className="card p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="font-semibold">Longitudinal MES</h2>
          <div className="text-xs text-ink-500">{points.length} session{points.length === 1 ? "" : "s"}</div>
        </div>
        <LongitudinalChart points={points} />
      </div>

      <div className="card overflow-hidden">
        <div className="px-4 py-3 border-b border-ink-200 dark:border-ink-700 font-semibold">
          Session history
        </div>
        {(sessions.data ?? []).length === 0 ? (
          <div className="p-8 text-center text-ink-500 text-sm">
            No sessions yet. Upload a recording to get started.
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead className="text-xs uppercase text-ink-500 bg-ink-50 dark:bg-ink-800/50">
              <tr>
                <th className="text-left px-4 py-2">Created</th>
                <th className="text-left px-4 py-2">Task</th>
                <th className="text-left px-4 py-2">Status</th>
                <th className="text-right px-4 py-2">MES</th>
                <th className="text-right px-4 py-2"></th>
              </tr>
            </thead>
            <tbody>
              {(sessions.data ?? []).map((s) => {
                const p = points.find((pp) => pp.session_id === s.id);
                const band = p?.mes_mean != null ? mesBand(p.mes_mean) : null;
                return (
                  <tr key={s.id} className="border-t border-ink-100 dark:border-ink-800 hover:bg-ink-50 dark:hover:bg-ink-800/30">
                    <td className="px-4 py-3 text-ink-700 dark:text-ink-200">{fmtDate(s.created_at)}</td>
                    <td className="px-4 py-3">{s.task}</td>
                    <td className="px-4 py-3">
                      <StatusPill status={s.status} progress={s.progress} />
                    </td>
                    <td className="px-4 py-3 text-right">
                      {p?.mes_mean != null ? (
                        <span className={band!.cls}>{p.mes_mean.toFixed(1)}</span>
                      ) : (
                        <span className="text-ink-400">—</span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <Link to={`/sessions/${s.id}`} className="btn-ghost text-xs">
                        <FileText className="w-3.5 h-3.5" /> Open
                      </Link>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

function StatusPill({ status, progress }: { status: string; progress: number }) {
  const map: Record<string, string> = {
    queued: "pill-muted",
    processing: "pill-info",
    done: "pill-good",
    failed: "pill-bad",
  };
  return (
    <span className={map[status] ?? "pill-muted"}>
      {status}{status === "processing" ? ` ${progress}%` : ""}
    </span>
  );
}

function Loading() {
  return <div className="max-w-7xl mx-auto px-6 py-8 text-ink-500">Loading…</div>;
}
function Empty({ msg }: { msg: string }) {
  return <div className="max-w-7xl mx-auto px-6 py-8 text-ink-500">{msg}</div>;
}

import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Search, Plus, ChevronRight, Sparkles } from "lucide-react";
import { api, type Participant } from "../lib/api";
import { fmtDate, mesBand } from "../lib/format";

void mesBand;  // retained for future per-row badge use

export function Dashboard() {
  const qc = useQueryClient();
  const nav = useNavigate();
  const [q, setQ] = useState("");
  const [showAdd, setShowAdd] = useState(false);

  const participants = useQuery({
    queryKey: ["participants"],
    queryFn: api.listParticipants,
  });

  const allSessions = useQuery({
    queryKey: ["all-sessions"],
    queryFn: () => api.listSessions(),
  });

  const seed = useMutation({
    mutationFn: api.seedDemo,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["participants"] });
      qc.invalidateQueries({ queryKey: ["all-sessions"] });
    },
  });

  const rows = (participants.data ?? []).filter((p) =>
    p.code.toLowerCase().includes(q.toLowerCase()),
  );

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <div className="flex items-center justify-between mb-6 flex-wrap gap-3">
        <div>
          <h1 className="text-3xl font-semibold">Participants</h1>
          <p className="text-ink-500 mt-1 text-sm">Pseudonymous research codes. No PHI stored.</p>
        </div>
        <div className="flex gap-2">
          <button className="btn-secondary" onClick={() => seed.mutate()} disabled={seed.isPending}>
            <Sparkles className="w-4 h-4" /> Seed demo data
          </button>
          <button className="btn-primary" onClick={() => setShowAdd(true)}>
            <Plus className="w-4 h-4" /> New participant
          </button>
        </div>
      </div>

      <div className="card overflow-hidden">
        <div className="p-4 border-b border-ink-200 dark:border-ink-700 flex items-center gap-2">
          <Search className="w-4 h-4 text-ink-400" />
          <input
            className="input border-0 focus:ring-0 px-0"
            placeholder="Search by code…"
            value={q}
            onChange={(e) => setQ(e.target.value)}
          />
          <span className="text-xs text-ink-400">{rows.length} of {participants.data?.length ?? 0}</span>
        </div>

        {participants.isLoading ? (
          <div className="p-8 text-center text-ink-500">Loading…</div>
        ) : rows.length === 0 ? (
          <EmptyState onAdd={() => setShowAdd(true)} onSeed={() => seed.mutate()} />
        ) : (
          <ul>
            {rows.map((p) => (
              <ParticipantRow key={p.id} p={p} sessions={allSessions.data ?? []} onClick={() => nav(`/participants/${p.id}`)} />
            ))}
          </ul>
        )}
      </div>

      {showAdd && <AddParticipantModal onClose={() => setShowAdd(false)} />}
    </div>
  );
}

function ParticipantRow({
  p,
  sessions,
  onClick,
}: {
  p: Participant;
  sessions: ReturnType<typeof api.listSessions> extends Promise<infer T> ? T : never;
  onClick: () => void;
}) {
  const subj = (sessions as any[]).filter((s) => s.participant_id === p.id);
  const lastSession = subj.find((s) => s.status === "done");
  return (
    <li
      className="px-4 py-3 border-t border-ink-100 dark:border-ink-800 hover:bg-ink-50 dark:hover:bg-ink-800/50 cursor-pointer transition-colors flex items-center justify-between gap-4"
      onClick={onClick}
    >
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <div className="font-mono font-medium text-ink-900 dark:text-ink-50">{p.code}</div>
          {p.notes && <span className="text-xs text-ink-500 truncate">{p.notes}</span>}
        </div>
        <div className="text-xs text-ink-500 mt-0.5">
          {subj.length} session{subj.length === 1 ? "" : "s"} · created {fmtDate(p.created_at)}
        </div>
      </div>
      {lastSession && lastSession.status === "done" && (
        <div className="text-sm font-medium text-teal-700 dark:text-teal-300 hidden sm:block">
          Last: {lastSession.task}
        </div>
      )}
      <ChevronRight className="w-4 h-4 text-ink-400" />
    </li>
  );
}

function EmptyState({ onAdd, onSeed }: { onAdd: () => void; onSeed: () => void }) {
  return (
    <div className="p-12 text-center">
      <h3 className="text-lg font-semibold">No participants yet</h3>
      <p className="text-ink-500 text-sm mt-1">Add a participant or seed two demo cases to get started.</p>
      <div className="mt-4 flex justify-center gap-2">
        <button className="btn-secondary" onClick={onSeed}>
          <Sparkles className="w-4 h-4" /> Seed demo
        </button>
        <button className="btn-primary" onClick={onAdd}>
          <Plus className="w-4 h-4" /> New participant
        </button>
      </div>
    </div>
  );
}

function AddParticipantModal({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient();
  const [code, setCode] = useState("");
  const [notes, setNotes] = useState("");
  const create = useMutation({
    mutationFn: () => api.createParticipant(code.trim(), notes.trim()),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["participants"] });
      onClose();
    },
  });
  return (
    <div className="fixed inset-0 bg-ink-900/40 grid place-items-center z-40 animate-fade-in p-4" onClick={onClose}>
      <div className="card p-6 w-full max-w-md" onClick={(e) => e.stopPropagation()}>
        <h3 className="text-lg font-semibold">New participant</h3>
        <p className="text-xs text-ink-500 mt-1">
          Use a pseudonymous research code. Do not enter names, MRNs, or other identifiers.
        </p>
        <div className="mt-4 space-y-3">
          <label className="block text-sm">
            <div className="text-ink-700 dark:text-ink-300 font-medium mb-1">Code</div>
            <input className="input font-mono" placeholder="e.g. P-0042"
              value={code} onChange={(e) => setCode(e.target.value)} autoFocus />
          </label>
          <label className="block text-sm">
            <div className="text-ink-700 dark:text-ink-300 font-medium mb-1">Notes (optional)</div>
            <textarea className="input min-h-[80px]" placeholder="Research-only notes…"
              value={notes} onChange={(e) => setNotes(e.target.value)} />
          </label>
          {create.error && <div className="text-rose-600 text-xs">{(create.error as Error).message}</div>}
        </div>
        <div className="flex gap-2 justify-end mt-5">
          <button className="btn-ghost" onClick={onClose}>Cancel</button>
          <button className="btn-primary" disabled={!code.trim() || create.isPending}
            onClick={() => create.mutate()}>
            {create.isPending ? "Creating…" : "Create"}
          </button>
        </div>
      </div>
    </div>
  );
}

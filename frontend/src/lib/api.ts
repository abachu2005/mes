/**
 * Thin fetch wrapper for the MES backend.
 *
 * Uses relative URLs in production (frontend is served from the same FastAPI
 * container) and a Vite proxy to localhost:7860 in dev.
 */

export type Participant = {
  id: string;
  code: string;
  notes: string | null;
  created_at: string;
};

export type SessionDTO = {
  id: string;
  participant_id: string;
  task: string;
  target_limb: string;
  headset: string;
  montage: string;
  original_filename: string | null;
  status: "queued" | "processing" | "done" | "failed";
  progress: number;
  error: string | null;
  is_demo: number;
  created_at: string;
  completed_at: string | null;
};

export type TopomapPoint = {
  channel: string;
  x: number;
  y: number;
  value: number;
};
export type TopomapPayload = {
  points: TopomapPoint[];
  vmin: number;
  vmax: number;
};

export type ScoreDTO = {
  id: string;
  session_id: string;
  mes_mean: number;
  mes_median: number;
  mes_std: number;
  n_trials: number;
  lateralization: number;
  mes_per_trial: number[];
  erd_topomap: TopomapPayload;
  raw_features: Record<string, number[]>;
  model_sha: string | null;
  created_at: string;
};

export type LongitudinalPoint = {
  session_id: string;
  created_at: string;
  task: string;
  mes_mean: number | null;
  mes_std: number | null;
  status: string;
};

export type ParticipantLongitudinal = {
  participant: Participant;
  points: LongitudinalPoint[];
};

const headers = { "Content-Type": "application/json" };

async function asJson<T>(r: Response): Promise<T> {
  if (!r.ok) {
    const text = await r.text();
    throw new Error(`${r.status} ${r.statusText}: ${text}`);
  }
  return r.json() as Promise<T>;
}

export const api = {
  health: () => fetch("/api/healthz").then(asJson),
  models: () => fetch("/api/models").then(asJson),

  listParticipants: () => fetch("/api/participants").then((r) => asJson<Participant[]>(r)),
  getParticipant: (id: string) => fetch(`/api/participants/${id}`).then(asJson<Participant>),
  createParticipant: (code: string, notes = "") =>
    fetch("/api/participants", {
      method: "POST",
      headers,
      body: JSON.stringify({ code, notes }),
    }).then(asJson<Participant>),
  longitudinal: (id: string) =>
    fetch(`/api/participants/${id}/longitudinal`).then(asJson<ParticipantLongitudinal>),

  listSessions: (participantId?: string) =>
    fetch("/api/sessions" + (participantId ? `?participant_id=${participantId}` : "")).then(
      asJson<SessionDTO[]>,
    ),
  getSession: (id: string) => fetch(`/api/sessions/${id}`).then(asJson<SessionDTO>),
  getScore: (id: string) => fetch(`/api/sessions/${id}/score`).then(asJson<ScoreDTO>),
  uploadSession: (form: FormData) =>
    fetch("/api/sessions", { method: "POST", body: form }).then(asJson<SessionDTO>),
  deleteSession: (id: string) =>
    fetch(`/api/sessions/${id}`, { method: "DELETE" }).then((r) => {
      if (!r.ok && r.status !== 204) throw new Error(`delete failed ${r.status}`);
      return true;
    }),
  reportUrl: (id: string) => `/api/sessions/${id}/report.pdf`,

  seedDemo: () => fetch("/api/demo/seed", { method: "POST" }).then(asJson<SessionDTO[]>),
};

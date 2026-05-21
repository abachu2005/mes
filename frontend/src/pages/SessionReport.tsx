import { useEffect, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { Download, ArrowLeft, Activity, BarChart3, Compass, Info } from "lucide-react";
import Joyride, { type Step } from "react-joyride";
import { api } from "../lib/api";
import { fmtDate, mesBand } from "../lib/format";
import { formatModelSha } from "../lib/modelInfo";
import { MesGauge } from "../components/MesGauge";
import { MesTrace } from "../components/MesTrace";
import { Topomap } from "../components/Topomap";
import { Tooltip } from "../components/Tooltip";

export function SessionReport() {
  const { id } = useParams<{ id: string }>();
  const [sp] = useSearchParams();
  const [runTour, setRunTour] = useState(sp.get("tour") === "1");

  const session = useQuery({
    queryKey: ["session", id],
    queryFn: () => api.getSession(id!),
    enabled: !!id,
    refetchInterval: (q) => {
      const st = (q.state.data as any)?.status;
      return st === "queued" || st === "processing" ? 1500 : false;
    },
  });

  const score = useQuery({
    queryKey: ["score", id],
    queryFn: () => api.getScore(id!),
    enabled: session.data?.status === "done",
  });

  useEffect(() => {
    if (session.data?.status === "done" && sp.get("tour") === "1") setRunTour(true);
  }, [session.data?.status, sp]);

  if (session.isLoading || !session.data) return <Loading />;
  const s = session.data;

  if (s.status === "failed") {
    return (
      <div className="max-w-3xl mx-auto px-6 py-8">
        <Link to={`/participants/${s.participant_id}`} className="btn-ghost text-sm -ml-3">
          <ArrowLeft className="w-4 h-4" /> Back
        </Link>
        <div className="card p-6 mt-4 border-rose-200">
          <h1 className="text-2xl font-semibold text-rose-700">Processing failed</h1>
          <pre className="text-xs text-ink-600 dark:text-ink-300 mt-3 whitespace-pre-wrap bg-ink-50 dark:bg-ink-900 p-3 rounded-lg overflow-auto max-h-96">
            {s.error}
          </pre>
        </div>
      </div>
    );
  }

  if (s.status !== "done" || !score.data) {
    return (
      <div className="max-w-3xl mx-auto px-6 py-8">
        <Link to={`/participants/${s.participant_id}`} className="btn-ghost text-sm -ml-3">
          <ArrowLeft className="w-4 h-4" /> Back
        </Link>
        <div className="card p-10 mt-4 text-center">
          <Activity className="w-10 h-10 mx-auto text-teal-600 animate-pulse" />
          <h2 className="mt-3 text-xl font-semibold">Processing your recording</h2>
          <p className="text-ink-500 text-sm mt-1">{s.task} · {s.original_filename}</p>
          <div className="mt-4 max-w-md mx-auto h-2 bg-ink-100 dark:bg-ink-800 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-teal-600"
              initial={{ width: 0 }}
              animate={{ width: `${s.progress}%` }}
              transition={{ duration: 0.4 }}
            />
          </div>
          <div className="text-xs text-ink-500 mt-2">{s.progress}%</div>
        </div>
      </div>
    );
  }

  const sc = score.data;
  const rehabMeta = sc.score_meta?.rehab_proxy as
    | { rehab_proxy_index?: number; mes_paretic_mean?: number; capacity_weight?: number }
    | undefined;
  const headlineMes =
    typeof rehabMeta?.rehab_proxy_index === "number"
      ? rehabMeta.rehab_proxy_index
      : sc.mes_mean;
  const band = mesBand(headlineMes);
  const model = formatModelSha(sc.model_sha);

  const tourSteps: Step[] = [
    { target: "[data-tour=gauge]", content: "This gauge shows the participant's mean MES across all trials. Higher = stronger motor-cortical engagement.", disableBeacon: true },
    { target: "[data-tour=trace]", content: "Per-trial MES values. Watch for engagement that grows over the session (a positive sign in rehab)." },
    { target: "[data-tour=topo]", content: "ERD topomap — red over the contralateral motor strip means the target hemisphere is activating." },
    { target: "[data-tour=lat]",  content: "Lateralization index in [-1, +1]. Positive = contralateral dominance (expected for the target task)." },
  ];

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 space-y-6">
      <Joyride
        steps={tourSteps}
        run={runTour}
        continuous
        showProgress
        showSkipButton
        styles={{ options: { primaryColor: "#0d9488" } }}
        callback={(data) => { if (["finished","skipped"].includes(data.status)) setRunTour(false); }}
      />

      <div className="flex items-start justify-between flex-wrap gap-3">
        <div>
          <Link to={`/participants/${s.participant_id}`} className="btn-ghost text-sm -ml-3 mb-2">
            <ArrowLeft className="w-4 h-4" /> Back to participant
          </Link>
          <h1 className="text-3xl font-semibold">Session report</h1>
          <p className="text-ink-500 text-sm mt-1">
            {s.task} · {s.target_limb} · {s.headset} · {fmtDate(s.created_at)}
          </p>
          <p className="text-xs text-ink-500 mt-2" title={model.detail}>
            Classifier:{" "}
            <span
              className={
                model.isHeuristic
                  ? "text-amber-700 dark:text-amber-400 font-medium"
                  : model.isEnsemble
                    ? "text-teal-700 dark:text-teal-400 font-medium"
                    : "text-ink-600 dark:text-ink-300 font-medium"
              }
            >
              {model.label}
            </span>
            {sc.reliability && (
              <>
                {" · "}
                <ReliabilityBadge tier={sc.reliability} />
              </>
            )}
          </p>
          {sc.mes_recovery_z != null && (
            <p className="text-xs text-ink-500 mt-1">
              Recovery index (vs prior sessions):{" "}
              <span className="font-medium text-ink-700 dark:text-ink-200">
                {sc.mes_recovery_z >= 0 ? "+" : ""}
                {sc.mes_recovery_z.toFixed(2)} σ
              </span>
              {sc.score_meta?.recovery_label != null && (
                <span className="text-ink-400"> ({String(sc.score_meta.recovery_label)})</span>
              )}
            </p>
          )}
        </div>
        <div className="flex gap-2">
          <button className="btn-ghost" onClick={() => setRunTour(true)} aria-label="Guided tour">
            <Info className="w-4 h-4" /> Tour
          </button>
          <a className="btn-primary" href={api.reportUrl(s.id)} target="_blank" rel="noreferrer">
            <Download className="w-4 h-4" /> Download PDF
          </a>
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        <div className="card p-6 lg:col-span-1" data-tour="gauge">
          <Header
            icon={<Activity className="w-4 h-4" />}
            title={rehabMeta ? "Rehab proxy (RPI)" : "Headline MES"}
            hint={
              rehabMeta
                ? "Stroke cohort: paretic-hand MES weighted by clinical capacity (MBI/NIHSS when available). Research index only."
                : "The Motor Engagement Signal is a 0–100 score that combines ERD strength, lateralization, MRCP amplitude, and the classifier posterior. 100 means maximum task-related neural drive."
            }
          />
          <MesGauge value={headlineMes} />
          {rehabMeta && (
            <p className="text-xs text-ink-500 mt-2 text-center">
              Raw MES mean {sc.mes_mean.toFixed(1)}
              {rehabMeta.mes_paretic_mean != null && (
                <> · Paretic-hand MES {Number(rehabMeta.mes_paretic_mean).toFixed(1)}</>
              )}
            </p>
          )}
          <div className="grid grid-cols-3 gap-3 mt-4 text-center">
            <Stat label="Median" value={sc.mes_median.toFixed(1)} />
            <Stat label="Std" value={sc.mes_std.toFixed(1)} />
            <Stat label="Trials" value={String(sc.n_trials)} />
          </div>
          <div className="mt-4 text-center">
            <span className={band.cls}>{band.label}</span>
          </div>
        </div>

        <div className="card p-6 lg:col-span-2" data-tour="trace">
          <Header icon={<BarChart3 className="w-4 h-4" />} title="MES across trials" hint="One MES value per epoch in chronological order. Stable engagement around 50+ is the usual goal." />
          <MesTrace values={sc.mes_per_trial} />
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        <div className="card p-6" data-tour="topo">
          <Header icon={<Compass className="w-4 h-4" />} title="ERD scalp topomap" hint="Event-Related Desynchronization (mu band, 8–13 Hz). Power drop during task vs baseline. Red over the contralateral hemisphere is the canonical motor activation signature." />
          <Topomap payload={sc.erd_topomap} />
        </div>
        <div className="card p-6" data-tour="lat">
          <Header icon={<Compass className="w-4 h-4 rotate-90" />} title="Lateralization" hint="LI = (ERD_contra − ERD_ipsi) / (|ERD_contra| + |ERD_ipsi|). Range [−1, +1]. Positive values indicate stronger contralateral engagement (expected for the target task)." />
          <div className="text-6xl font-bold text-teal-600 my-4">
            {(sc.lateralization >= 0 ? "+" : "") + sc.lateralization.toFixed(2)}
          </div>
          <LateralBar v={sc.lateralization} />
        </div>
      </div>

      <div className="card p-6">
        <Header icon={<Info className="w-4 h-4" />} title="Per-trial detail" />
        <table className="w-full text-sm mt-2">
          <thead className="text-xs uppercase text-ink-500">
            <tr><th className="text-left p-2">#</th>
              <th className="text-right p-2">MES</th>
              <th className="text-right p-2">z(mu)</th>
              <th className="text-right p-2">z(beta)</th>
              <th className="text-right p-2">z(LI)</th>
              <th className="text-right p-2">z(MRCP)</th>
            </tr>
          </thead>
          <tbody>
            {sc.mes_per_trial.map((v, i) => (
              <tr key={i} className="border-t border-ink-100 dark:border-ink-800">
                <td className="p-2 text-ink-500">{i + 1}</td>
                <td className="p-2 text-right font-medium">{v.toFixed(1)}</td>
                <td className="p-2 text-right tabular-nums">{(sc.raw_features.z_mu?.[i] ?? 0).toFixed(2)}</td>
                <td className="p-2 text-right tabular-nums">{(sc.raw_features.z_beta?.[i] ?? 0).toFixed(2)}</td>
                <td className="p-2 text-right tabular-nums">{(sc.raw_features.z_li?.[i] ?? 0).toFixed(2)}</td>
                <td className="p-2 text-right tabular-nums">{(sc.raw_features.z_mrcp?.[i] ?? 0).toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Header({ icon, title, hint }: { icon: React.ReactNode; title: string; hint?: string }) {
  return (
    <div className="flex items-center justify-between mb-3">
      <div className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wider text-ink-600 dark:text-ink-300">
        {icon} {title}
      </div>
      {hint && <Tooltip text={hint} />}
    </div>
  );
}
function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-lg font-semibold">{value}</div>
      <div className="text-xs text-ink-500 uppercase tracking-wider">{label}</div>
    </div>
  );
}
function LateralBar({ v }: { v: number }) {
  const pct = Math.max(-1, Math.min(1, v)) * 50 + 50;
  return (
    <div>
      <div className="relative h-3 bg-ink-100 dark:bg-ink-800 rounded-full overflow-hidden">
        <div className="absolute top-0 bottom-0 bg-teal-500" style={{ left: "50%", width: `${Math.abs(pct - 50)}%`, transform: v < 0 ? "translateX(-100%)" : undefined }} />
        <div className="absolute top-0 bottom-0 left-1/2 w-px bg-ink-400" />
      </div>
      <div className="flex justify-between text-xs text-ink-500 mt-1">
        <span>ipsilateral (-1)</span><span>0</span><span>contralateral (+1)</span>
      </div>
    </div>
  );
}
function ReliabilityBadge({ tier }: { tier: string }) {
  const cls =
    tier === "High"
      ? "text-teal-700 dark:text-teal-400"
      : tier === "Medium"
        ? "text-amber-700 dark:text-amber-400"
        : "text-rose-700 dark:text-rose-400";
  return <span className={`font-medium ${cls}`}>{tier} reliability</span>;
}

function Loading() {
  return <div className="max-w-7xl mx-auto px-6 py-8 text-ink-500">Loading…</div>;
}

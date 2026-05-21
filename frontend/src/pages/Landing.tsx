import { Link, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { Activity, ArrowRight, Brain, LineChart, FileBadge2, Sparkles, ShieldCheck, Upload, BarChart3 } from "lucide-react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";

export function Landing() {
  const nav = useNavigate();
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const seed = useMutation({
    mutationFn: api.seedDemo,
    onSuccess: (rows) => {
      if (rows.length) nav(`/sessions/${rows[0].id}?tour=1`);
    },
  });

  return (
    <div className="bg-gradient-to-b from-white via-white to-ink-50 dark:from-ink-950 dark:via-ink-900 dark:to-ink-900">
      <header className="max-w-7xl mx-auto px-6 pt-8 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2 font-semibold">
          <span className="grid place-items-center w-9 h-9 rounded-xl bg-teal-600 text-white shadow-card">
            <Activity className="w-5 h-5" />
          </span>
          <span className="text-lg">MES</span>
        </Link>
        <nav className="flex items-center gap-2 text-sm">
          <Link to="/dashboard" className="btn-ghost">Dashboard</Link>
          <Link to="/about" className="btn-ghost">About</Link>
          <a
            href="https://github.com"
            target="_blank"
            rel="noreferrer"
            className="btn-secondary"
          >
            GitHub
          </a>
        </nav>
      </header>

      <section className="max-w-7xl mx-auto px-6 pt-16 pb-20 grid lg:grid-cols-2 gap-12 items-center">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <span className="pill-info inline-flex">
            <Sparkles className="w-3.5 h-3.5" /> Open-source · EEG · Motor rehab
          </span>
          <h1 className="mt-4 text-5xl sm:text-6xl font-bold tracking-tight text-ink-900 dark:text-ink-50">
            Motor Engagement Signal
          </h1>
          <p className="mt-4 text-xl text-ink-600 dark:text-ink-300 max-w-xl">
            Quantifying neural drive for movement recovery.{" "}
            A single, calibrated EEG-derived score that tracks how strongly a
            patient's motor cortex engages — built for stroke and SCI rehabilitation.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <button
              className="btn-primary text-base px-5 py-2.5"
              onClick={() => seed.mutate()}
              disabled={seed.isPending}
            >
              {seed.isPending ? "Loading demo…" : "Try the live demo"} <ArrowRight className="w-4 h-4" />
            </button>
            <Link to="/dashboard" className="btn-secondary text-base px-5 py-2.5">
              Go to dashboard
            </Link>
          </div>
          <div className="mt-10 grid grid-cols-3 gap-4 max-w-md">
            <Stat label="Datasets" value="4+" sub="PhysioNet · BCI IV · Liu2024/25" />
            <Stat label="Channels" value="16" sub="OpenBCI Cyton+Daisy @125 Hz" />
            <Stat label="Models" value={models.data ? String((models.data as any).available?.length ?? 2) : "—"} sub="ONNX · CPU inference" />
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.96 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="relative"
        >
          <HeroVisualization />
        </motion.div>
      </section>

      <section className="max-w-7xl mx-auto px-6 pb-20">
        <h2 className="text-2xl font-semibold text-center mb-12">How it works</h2>
        <div className="grid md:grid-cols-3 gap-6">
          <Step
            n={1}
            icon={<Upload className="w-5 h-5" />}
            title="Upload a recording"
            body="Drop in an EDF, BDF, or OpenBCI .txt file of a participant performing motor imagery or movement."
          />
          <Step
            n={2}
            icon={<Brain className="w-5 h-5" />}
            title="Extract neural signatures"
            body="Bandpower in mu/beta, ERD%, MRCP, lateralization, and Riemannian covariances on a 16-channel sensorimotor montage."
          />
          <Step
            n={3}
            icon={<BarChart3 className="w-5 h-5" />}
            title="Get a calibrated MES score"
            body="Per-trial MES (0–100) plus topomaps, longitudinal tracking, and a downloadable PDF report."
          />
        </div>
      </section>

      <section className="max-w-7xl mx-auto px-6 pb-24">
        <div className="card p-8 grid md:grid-cols-3 gap-8 items-center">
          <div>
            <span className="pill-info"><ShieldCheck className="w-3.5 h-3.5" /> Validated on open data</span>
            <h3 className="mt-3 text-2xl font-semibold">Honest validation</h3>
          </div>
          <p className="md:col-span-2 text-ink-600 dark:text-ink-300">
            Trained on 100+ healthy controls (PhysioNet eegmmidb), benchmarked on BCI Competition IV 2a/2b,
            with open benchmarks on PhysioNet MI and Liu2024 acute stroke (hand MI vs rest/break).
            Stroke discrimination is tracked separately from healthy MI — see{" "}
            <a className="text-teal-600 hover:underline" href="https://huggingface.co/abachu2005/mes-models">benchmarks.md</a>{" "}
            and the committed benchmarks JSON in the repository.
          </p>
        </div>
      </section>
    </div>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div>
      <div className="text-3xl font-bold text-teal-600">{value}</div>
      <div className="text-xs uppercase tracking-wider text-ink-500 mt-0.5">{label}</div>
      <div className="text-xs text-ink-500 mt-0.5">{sub}</div>
    </div>
  );
}

function Step({ n, icon, title, body }: { n: number; icon: React.ReactNode; title: string; body: string }) {
  return (
    <div className="card p-6">
      <div className="flex items-center gap-3">
        <span className="w-8 h-8 grid place-items-center rounded-lg bg-teal-50 dark:bg-teal-900/40 text-teal-700 dark:text-teal-300">
          {icon}
        </span>
        <span className="pill-muted">Step {n}</span>
      </div>
      <h3 className="mt-3 font-semibold text-lg">{title}</h3>
      <p className="mt-1 text-sm text-ink-600 dark:text-ink-300">{body}</p>
    </div>
  );
}

function HeroVisualization() {
  // Stylized MES gauge + trace for the hero (pure SVG / motion, no API needed).
  const mes = 72;
  return (
    <div className="card p-6">
      <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-ink-500">
        <LineChart className="w-3.5 h-3.5" /> Live demo preview
      </div>
      <div className="flex items-end gap-6 mt-2">
        <div className="relative w-40 h-40">
          <svg viewBox="0 0 100 100" className="absolute inset-0">
            <circle cx="50" cy="50" r="42" fill="none" stroke="rgb(226,232,240)" strokeWidth="9" />
            <motion.circle
              cx="50" cy="50" r="42" fill="none" stroke="#0d9488" strokeWidth="9" strokeLinecap="round"
              strokeDasharray={`${(mes / 100) * 264} 264`}
              transform="rotate(-90 50 50)"
              initial={{ strokeDasharray: "0 264" }}
              animate={{ strokeDasharray: `${(mes / 100) * 264} 264` }}
              transition={{ duration: 1.2, ease: "easeOut" }}
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <div className="text-4xl font-bold text-ink-900 dark:text-ink-50">{mes}</div>
            <div className="text-xs text-ink-500 uppercase tracking-wider mt-1">MES</div>
          </div>
        </div>
        <div className="flex-1 h-32">
          <svg viewBox="0 0 200 80" className="w-full h-full">
            <polyline
              fill="none" stroke="#14b8a6" strokeWidth="2.5" strokeLinecap="round"
              strokeLinejoin="round"
              points="0,60 20,55 40,48 60,42 80,38 100,30 120,32 140,26 160,22 180,18 200,15"
            />
            <line x1="0" y1="40" x2="200" y2="40" stroke="rgb(203,213,225)" strokeDasharray="3 3" strokeWidth="0.8" />
          </svg>
          <div className="text-xs text-ink-500 mt-1 flex justify-between">
            <span>Session 1</span>
            <span>Session 8</span>
          </div>
        </div>
      </div>
      <div className="mt-4 grid grid-cols-3 gap-3 text-xs">
        <Mini label="Mu ERD" value="+42%" />
        <Mini label="Beta ERD" value="+28%" />
        <Mini label="Lateralization" value="+0.48" />
      </div>
      <div className="mt-3 inline-flex items-center gap-2 text-xs text-ink-500">
        <FileBadge2 className="w-3.5 h-3.5" /> Mock data shown — click "Try the live demo" for real model.
      </div>
    </div>
  );
}

function Mini({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-ink-50 dark:bg-ink-700/50 rounded-lg px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-ink-500">{label}</div>
      <div className="font-semibold text-ink-900 dark:text-ink-100">{value}</div>
    </div>
  );
}

import { useQuery } from "@tanstack/react-query";
import { ExternalLink } from "lucide-react";
import { api } from "../lib/api";

export function About() {
  const health = useQuery({ queryKey: ["health"], queryFn: api.health });
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });

  return (
    <div className="max-w-4xl mx-auto px-6 py-10 space-y-8">
      <div>
        <h1 className="text-3xl font-semibold">About MES</h1>
        <p className="mt-2 text-ink-600 dark:text-ink-300">
          The Motor Engagement Signal (MES) is an open-source EEG analysis pipeline
          that produces a single calibrated score (0–100) reflecting the strength
          of motor-cortical engagement during a movement or motor-imagery task.
        </p>
      </div>

      <Section title="The MES formula">
        <code className="block bg-ink-900 text-ink-50 p-4 rounded-xl text-sm overflow-x-auto">
{`raw = w1·z(ERD_mu) + w2·z(ERD_beta) + w3·z(LI) + w4·z(MRCP) + w5·logit(p_model)
MES = 100 · sigmoid(raw)`}
        </code>
        <p className="text-sm text-ink-600 dark:text-ink-300 mt-3">
          Where ERD is Event-Related Desynchronization (mu and beta), LI is the
          lateralization index, MRCP is the movement-related cortical potential
          amplitude, and p_model is the classifier's posterior for the target task.
          Weights are fit by logistic regression of these features against
          dataset-given task-vs-rest labels.
        </p>
      </Section>

      <Section title="Hardware target">
        <p className="text-sm text-ink-600 dark:text-ink-300">
          OpenBCI Cyton + Daisy. 16 channels @ 125 Hz, sensorimotor-centered montage
          (Fpz, Fz, FC3/FCz/FC4, C3/C1/Cz/C2/C4, CP3/CPz/CP4, T7/T8, Pz). Recordings
          from any standard 10-20 cap are spatially mapped to this montage via
          spherical-spline interpolation.
        </p>
      </Section>

      <Section title="Data">
        <ul className="text-sm space-y-1">
          <li>· PhysioNet EEG Motor Movement/Imagery (~109 subjects, healthy)</li>
          <li>· BCI Competition IV 2a + 2b (held-out benchmark)</li>
          <li>· Liu2024 — 50 acute stroke patients (hand MI)</li>
          <li>· Liu2025 — 27 stroke patients (gait MI, longitudinal)</li>
        </ul>
      </Section>

      <Section title="Models available">
        {!models.data ? <div className="text-ink-500 text-sm">Loading…</div> : (
          <div className="space-y-2 text-sm">
            <div>Repo: <code>{(models.data as any).model_repo}</code></div>
            <div>Files:</div>
            <ul className="list-disc list-inside text-ink-600 dark:text-ink-300">
              {((models.data as any).available ?? []).map((f: string) => (
                <li key={f}><code>{f}</code></li>
              ))}
              {((models.data as any).available ?? []).length === 0 && (
                <li className="italic text-ink-500">No models published yet — training in progress on Kaggle.</li>
              )}
            </ul>
          </div>
        )}
      </Section>

      <Section title="Limitations">
        <ul className="text-sm space-y-1 text-ink-600 dark:text-ink-300">
          <li>· 16-channel mapped data underperforms 64-channel research caps for cross-subject MI.</li>
          <li>· Stroke validation limited to two open datasets. Clinical use would need IRB-approved prospective data.</li>
          <li>· Research use only — not FDA / CE cleared.</li>
          <li>· HF Space free-tier sleeps after 48 h inactivity (1-2 min cold start).</li>
        </ul>
      </Section>

      <Section title="System status">
        {!health.data ? <div className="text-ink-500 text-sm">Loading…</div> : (
          <pre className="text-xs bg-ink-100 dark:bg-ink-800 p-3 rounded-lg">
{JSON.stringify(health.data, null, 2)}
          </pre>
        )}
      </Section>

      <div className="text-sm text-ink-500">
        <a className="hover:underline text-teal-600" href="https://huggingface.co/abachu2005/mes-models" target="_blank" rel="noreferrer">
          Model repo <ExternalLink className="inline w-3 h-3" />
        </a>
        <span className="mx-2">·</span>
        <a className="hover:underline text-teal-600" href="https://huggingface.co/datasets/abachu2005/mes-eeg-processed" target="_blank" rel="noreferrer">
          Dataset repo <ExternalLink className="inline w-3 h-3" />
        </a>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="card p-6">
      <h2 className="font-semibold text-lg mb-3">{title}</h2>
      {children}
    </section>
  );
}

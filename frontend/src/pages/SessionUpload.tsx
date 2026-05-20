import { useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Upload, ArrowLeft, AlertCircle, FileAudio2 } from "lucide-react";
import { api } from "../lib/api";
import { Topomap } from "../components/Topomap";

const TASKS = [
  { value: "right_hand", label: "Right hand — motor imagery" },
  { value: "left_hand", label: "Left hand — motor imagery" },
  { value: "feet", label: "Feet — motor imagery" },
  { value: "gait", label: "Gait — motor imagery" },
];

export function SessionUpload() {
  const { id } = useParams<{ id: string }>();
  const nav = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [task, setTask] = useState("right_hand");
  const [target, setTarget] = useState("Right hand");
  const [headset, setHeadset] = useState("OpenBCI Cyton+Daisy");

  const p = useQuery({ queryKey: ["participant", id], queryFn: () => api.getParticipant(id!), enabled: !!id });

  const submit = useMutation({
    mutationFn: () => {
      if (!file) throw new Error("Pick a file first");
      const f = new FormData();
      f.append("file", file);
      f.append("participant_id", id!);
      f.append("task", task);
      f.append("target_limb", target);
      f.append("headset", headset);
      return api.uploadSession(f);
    },
    onSuccess: (s) => nav(`/sessions/${s.id}`),
  });

  return (
    <div className="max-w-3xl mx-auto px-6 py-8 space-y-6">
      <Link to={`/participants/${id}`} className="btn-ghost text-sm w-fit -ml-3">
        <ArrowLeft className="w-4 h-4" /> Back to {p.data?.code ?? "participant"}
      </Link>
      <div>
        <h1 className="text-3xl font-semibold">Upload session</h1>
        <p className="text-ink-500 text-sm mt-1">
          Drop in an EDF, BDF, GDF, FIF, BrainVision .vhdr, EEGLAB .set, or
          OpenBCI .txt/.csv file.
        </p>
      </div>

      <div className="card p-6 space-y-5">
        <div
          className="border-2 border-dashed border-ink-300 dark:border-ink-700 rounded-xl p-8 text-center cursor-pointer hover:border-teal-500 hover:bg-teal-50/30 dark:hover:bg-teal-900/10 transition-colors"
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            const f = e.dataTransfer.files[0];
            if (f) setFile(f);
          }}
        >
          <input
            type="file"
            ref={inputRef}
            className="hidden"
            accept=".edf,.bdf,.gdf,.fif,.vhdr,.set,.txt,.csv"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
          {file ? (
            <div>
              <FileAudio2 className="w-8 h-8 mx-auto text-teal-600" />
              <div className="font-medium mt-2">{file.name}</div>
              <div className="text-xs text-ink-500">{(file.size / 1024 / 1024).toFixed(2)} MB</div>
              <button className="btn-ghost text-xs mt-2" onClick={(e) => { e.stopPropagation(); setFile(null); }}>
                Remove
              </button>
            </div>
          ) : (
            <div>
              <Upload className="w-8 h-8 mx-auto text-ink-400" />
              <div className="font-medium mt-2 text-ink-700 dark:text-ink-200">Click to choose, or drag a file here</div>
              <div className="text-xs text-ink-500 mt-1">Max 200 MB · stays on the server</div>
            </div>
          )}
        </div>

        <div className="grid sm:grid-cols-2 gap-4">
          <label className="block text-sm">
            <div className="font-medium mb-1">Task</div>
            <select className="input" value={task} onChange={(e) => setTask(e.target.value)}>
              {TASKS.map((t) => <option key={t.value} value={t.value}>{t.label}</option>)}
            </select>
          </label>
          <label className="block text-sm">
            <div className="font-medium mb-1">Target limb</div>
            <input className="input" value={target} onChange={(e) => setTarget(e.target.value)} />
          </label>
          <label className="block text-sm sm:col-span-2">
            <div className="font-medium mb-1">Headset</div>
            <input className="input" value={headset} onChange={(e) => setHeadset(e.target.value)} />
            <div className="text-xs text-ink-500 mt-1">
              Data will be spatially mapped to the 16-channel OpenBCI sensorimotor montage @ 125 Hz.
            </div>
          </label>
        </div>

        <div>
          <div className="font-medium text-sm mb-2">Production electrode placement</div>
          <Topomap
            payload={{
              vmin: 0, vmax: 1,
              points: [
                "Fpz","Fz","FC3","FCz","FC4","C3","C1","Cz","C2","C4","CP3","CPz","CP4","T7","T8","Pz",
              ].map((ch) => ({ channel: ch, x: 0, y: 0, value: 0.5 })),
            }}
            labelOnly
          />
        </div>

        {submit.error && (
          <div className="flex items-start gap-2 text-rose-700 text-sm">
            <AlertCircle className="w-4 h-4 mt-0.5" />
            <div>{(submit.error as Error).message}</div>
          </div>
        )}

        <div className="flex justify-end gap-2">
          <Link to={`/participants/${id}`} className="btn-ghost">Cancel</Link>
          <button
            className="btn-primary"
            disabled={!file || submit.isPending}
            onClick={() => submit.mutate()}
          >
            <Upload className="w-4 h-4" /> {submit.isPending ? "Uploading…" : "Upload & process"}
          </button>
        </div>
      </div>
    </div>
  );
}

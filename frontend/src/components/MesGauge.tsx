import { motion } from "framer-motion";
import { mesBand } from "../lib/format";

export function MesGauge({ value }: { value: number }) {
  const v = Math.max(0, Math.min(100, value));
  const band = mesBand(v);
  const fill = v >= 70 ? "#0d9488" : v >= 40 ? "#14b8a6" : v >= 20 ? "#f59e0b" : "#e11d48";
  const circumference = 264; // 2 * pi * 42

  return (
    <div className="relative w-full max-w-[260px] aspect-square mx-auto">
      <svg viewBox="0 0 100 100" className="w-full h-full">
        <circle cx="50" cy="50" r="42" fill="none" stroke="rgb(226,232,240)" strokeWidth="9" className="dark:stroke-ink-700" />
        <motion.circle
          cx="50" cy="50" r="42" fill="none" stroke={fill} strokeWidth="9" strokeLinecap="round"
          transform="rotate(-90 50 50)"
          strokeDasharray={`0 ${circumference}`}
          animate={{ strokeDasharray: `${(v / 100) * circumference} ${circumference}` }}
          transition={{ duration: 1.0, ease: "easeOut" }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <motion.div
          initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.6 }}
          className="text-5xl font-bold tabular-nums text-ink-900 dark:text-ink-50"
        >
          {v.toFixed(1)}
        </motion.div>
        <div className="text-xs uppercase tracking-widest text-ink-500 mt-1">MES</div>
        <div className={`mt-2 ${band.cls}`}>{band.label}</div>
      </div>
    </div>
  );
}

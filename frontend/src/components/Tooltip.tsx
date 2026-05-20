import { useState } from "react";
import { Info } from "lucide-react";

export function Tooltip({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="relative">
      <button
        type="button"
        className="text-ink-400 hover:text-teal-600 focus:outline-none focus:ring-2 focus:ring-teal-500 rounded-full"
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        aria-label="explanation"
      >
        <Info className="w-4 h-4" />
      </button>
      {open && (
        <div className="absolute right-0 top-6 w-64 bg-ink-900 text-ink-50 text-xs p-3 rounded-lg shadow-card-hover z-20 leading-relaxed">
          {text}
        </div>
      )}
    </div>
  );
}

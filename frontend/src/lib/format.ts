export function fmtPct(v: number, digits = 1): string {
  return `${v.toFixed(digits)}`;
}

export function fmtDate(s: string): string {
  try {
    return new Date(s).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
  } catch {
    return s;
  }
}

export function mesBand(v: number): { label: string; cls: string } {
  if (v >= 70) return { label: "Strong engagement", cls: "pill-good" };
  if (v >= 40) return { label: "Moderate", cls: "pill-info" };
  if (v >= 20) return { label: "Weak", cls: "pill-warn" };
  return { label: "Minimal", cls: "pill-bad" };
}

/** Human-readable labels for backend `model_sha` strings. */

export function formatModelSha(sha: string | null | undefined): {
  label: string;
  detail: string;
  isHeuristic: boolean;
  usesEegnet: boolean;
  isEnsemble: boolean;
} {
  const raw = (sha ?? "").trim();
  if (!raw || raw === "unknown") {
    return {
      label: "Unknown",
      detail: raw || "No model identifier recorded",
      isHeuristic: false,
      usesEegnet: false,
      isEnsemble: false,
    };
  }
  if (raw === "demo") {
    return {
      label: "Demo (synthetic)",
      detail: "Preloaded demonstration scores",
      isHeuristic: false,
      usesEegnet: false,
      isEnsemble: false,
    };
  }
  if (raw === "heuristic") {
    return {
      label: "Heuristic fallback",
      detail: "ONNX models unavailable; mu-band ERD logistic used for p_model",
      isHeuristic: true,
      usesEegnet: false,
      isEnsemble: false,
    };
  }
  const isEnsemble = raw.startsWith("ensemble(");
  const usesEegnet = /eegnet/i.test(raw);
  if (isEnsemble) {
    return {
      label: "Riemannian + EEGNet ensemble",
      detail: raw,
      isHeuristic: false,
      usesEegnet: true,
      isEnsemble: true,
    };
  }
  if (usesEegnet) {
    return {
      label: "EEGNet",
      detail: raw,
      isHeuristic: false,
      usesEegnet: true,
      isEnsemble: false,
    };
  }
  if (/riemannian/i.test(raw)) {
    return {
      label: "Riemannian logistic regression",
      detail: raw,
      isHeuristic: false,
      usesEegnet: false,
      isEnsemble: false,
    };
  }
  return {
    label: raw.slice(0, 48),
    detail: raw,
    isHeuristic: false,
    usesEegnet: usesEegnet,
    isEnsemble: isEnsemble,
  };
}

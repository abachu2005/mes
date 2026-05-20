#!/usr/bin/env bash
# Run the full Kaggle pipeline end-to-end, sequentially, with polling.
# Logs everything to scripts/run_all_kaggle.log
set -u
cd "$(dirname "$0")/.."

LOG="scripts/run_all_kaggle.log"
exec >>"$LOG" 2>&1

set -a; . ./.env.local; set +a
export KAGGLE_API_TOKEN
KAGGLE="$(pwd)/.venv/bin/kaggle"
PY="$(pwd)/.venv/bin/python"
SUBMIT="$PY scripts/kaggle_submit.py"

ts() { date "+%Y-%m-%d %H:%M:%S"; }

poll() {
  local kid="$1"
  local max_min="${2:-360}"
  local start=$(date +%s)
  while :; do
    local status
    status=$("$KAGGLE" kernels status "$kid" 2>&1 | tail -1)
    echo "[$(ts)] $status"
    case "$status" in
      *COMPLETE*|*SUCCESS*) return 0 ;;
      *ERROR*|*FAILED*|*CANCEL*) echo "[$(ts)] FAIL: $status"; return 2 ;;
    esac
    local now=$(date +%s)
    if (( (now-start)/60 >= max_min )); then
      echo "[$(ts)] TIMEOUT after ${max_min}m"; return 3
    fi
    sleep 60
  done
}

submit() {
  local src="$1" kid="$2" extra="$3"
  echo "[$(ts)] === submit $kid ($src) extra='$extra' ==="
  KAGGLE_CLI="$KAGGLE" $SUBMIT "$src" --kernel-id "$kid" $extra --no-poll || return $?
}

echo "[$(ts)] ============ run_all_kaggle.sh START ============"

# 1) Preprocess (already submitted v4 + running). Just poll.
echo "[$(ts)] step 1: poll preprocess"
poll "abachu2005/mes-00-preprocess" 180 || {
  echo "[$(ts)] preprocess failed; re-submitting v5"
  submit notebooks/kaggle/00_preprocess.py abachu2005/mes-00-preprocess "--no-gpu --internet" || exit 10
  poll "abachu2005/mes-00-preprocess" 180 || exit 11
}

# 2) Riemannian baseline (CPU). Submit + poll.
echo "[$(ts)] step 2: train_riemannian"
submit notebooks/kaggle/01_train_riemannian.py abachu2005/mes-01-train-riemannian "--no-gpu --internet" || exit 20
poll "abachu2005/mes-01-train-riemannian" 180 || echo "[$(ts)] riemannian failed - continuing"

# 3) EEGNet (GPU). Submit + poll.
echo "[$(ts)] step 3: train_eegnet (GPU)"
submit notebooks/kaggle/02_train_eegnet.py abachu2005/mes-02-train-eegnet "--gpu --internet" || exit 30
poll "abachu2005/mes-02-train-eegnet" 360 || echo "[$(ts)] eegnet failed - continuing"

# 4) Validate. Submit + poll.
echo "[$(ts)] step 4: validate"
submit notebooks/kaggle/03_validate.py abachu2005/mes-03-validate "--no-gpu --internet" || exit 40
poll "abachu2005/mes-03-validate" 90 || echo "[$(ts)] validate failed"

echo "[$(ts)] ============ run_all_kaggle.sh DONE ============"
echo "[$(ts)] Models: https://huggingface.co/abachu2005/mes-models"
echo "[$(ts)] Benchmarks: https://huggingface.co/abachu2005/mes-models/blob/main/benchmarks.md"

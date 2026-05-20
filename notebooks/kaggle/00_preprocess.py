# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # MES — 00_preprocess
#
# Downloads MOABB datasets, preprocesses to the 16-channel OpenBCI montage @ 125 Hz,
# writes per-dataset parquet, and uploads to the HF dataset repo.
#
# Runs on Kaggle CPU. Internet required. HF_TOKEN must be set as a Kaggle secret.

# +
import os, sys, json, gc, shutil
from pathlib import Path

# Install deps
os.system("pip install -q --upgrade pip")
# Retry-friendly install: split into groups, retry up to 5 times for flakes.
def _pip(pkgs):
    for attempt in range(5):
        rc = os.system(
            "pip install -q --default-timeout=60 --retries 5 " + pkgs
        )
        if rc == 0:
            return
    raise RuntimeError(f"pip failed: {pkgs}")

_pip("'huggingface_hub>=0.24' pyarrow pandas scipy")
_pip("mne")
# NOTE: skip moabb + mne-icalabel + autoreject - Kaggle's network blocks these.
# We use mne.datasets.eegbci directly for PhysioNet; ICA fallback uses kurtosis.
# -

# +
HF_TOKEN = os.environ["HF_TOKEN"]
HF_USER  = os.environ.get("HF_USERNAME", "abachu2005")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", f"{HF_USER}/mes-eeg-processed")
assert HF_TOKEN.startswith("hf_"), "HF_TOKEN missing or malformed"
# -

# +
import time
import numpy as np
import pandas as pd
import mne
mne.set_log_level("ERROR")
from huggingface_hub import HfApi, create_repo, login

def _retry(fn, what, tries=8, sleep_s=15):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            print(f"  HF {what} attempt {i+1}/{tries} failed: {type(e).__name__}: {e}")
            time.sleep(sleep_s)
    raise RuntimeError(f"HF {what} failed after {tries} retries: {last}")

_retry(lambda: login(token=HF_TOKEN, add_to_git_credential=False), "login")
api = HfApi(token=HF_TOKEN)
_retry(lambda: create_repo(repo_id=HF_DATASET_REPO, repo_type="dataset",
                            exist_ok=True, token=HF_TOKEN),
        "create_repo")
# -

# +
# Inline the channel constants + preprocessing helpers so the notebook is self-contained.
OPENBCI_MONTAGE_16 = (
    "Fpz","Fz","FC3","FCz","FC4","C3","C1","Cz","C2","C4",
    "CP3","CPz","CP4","T7","T8","Pz",
)
TARGET_SFREQ = 125.0
EPOCH_TMIN, EPOCH_TMAX = -2.0, 4.0
BASELINE = (-1.5, -0.5)

def map_to_openbci_16(raw, target=OPENBCI_MONTAGE_16):
    raw = raw.copy()
    try:
        raw.set_montage("standard_1020", on_missing="ignore", match_case=False)
    except Exception:
        pass
    present = set(raw.info["ch_names"])
    missing = [c for c in target if c not in present]
    if not missing:
        return raw.pick(list(target))
    target_present = [c for c in target if c in present]
    others = [c for c in raw.info["ch_names"] if c not in target]
    raw_p = raw.copy().pick(target_present + others)
    zero = np.zeros((len(missing), raw.n_times))
    info_m = mne.create_info(missing, sfreq=raw.info["sfreq"], ch_types="eeg")
    raw_m = mne.io.RawArray(zero, info_m)
    raw_m.info["bads"] = list(missing)
    raw_p.add_channels([raw_m], force_update_info=True)
    raw_p.set_montage("standard_1020", on_missing="ignore", match_case=False)
    try:
        raw_p.interpolate_bads(reset_bads=True, mode="accurate")
    except Exception:
        pass
    return raw_p.pick(list(target))

def preprocess_one(raw, do_ica=True):
    raw = raw.copy()
    raw.filter(0.5, 40.0, fir_design="firwin")
    nyq = raw.info["sfreq"] / 2.0
    notches = [f for f in (50.0, 60.0) if f < nyq - 1]
    if notches:
        raw.notch_filter(notches)
    raw.set_eeg_reference("average", projection=False)
    if do_ica and len(raw.info["ch_names"]) >= 32:
        try:
            ica = mne.preprocessing.ICA(
                n_components=min(20, len(raw.info["ch_names"]) - 1),
                method="fastica", random_state=1729, max_iter="auto",
            )
            ica.fit(raw)
            try:
                from mne_icalabel import label_components
                labels = label_components(raw, ica, method="iclabel")
                bad = [
                    i for i, (lab, p) in enumerate(
                        zip(labels["labels"], labels["y_pred_proba"])
                    )
                    if lab in {"eye blink","muscle artifact","heart beat","line noise"} and p > 0.7
                ]
                ica.exclude = bad
                raw = ica.apply(raw)
            except Exception:
                # Fallback: kurtosis-based rejection.
                src = ica.get_sources(raw).get_data()
                k = ((src - src.mean(1, keepdims=True))**4).mean(1) / (src.var(1)**2 + 1e-12) - 3.0
                ica.exclude = [int(i) for i in np.where(k > 5.0)[0]]
                raw = ica.apply(raw)
        except Exception as e:
            print("ICA skipped:", e)
    raw = map_to_openbci_16(raw)
    if abs(raw.info["sfreq"] - TARGET_SFREQ) > 0.5:
        raw.resample(TARGET_SFREQ, npad="auto")
    return raw

def epoch_one(raw, event_id):
    events, ev = mne.events_from_annotations(raw, event_id=event_id)
    if events.size == 0:
        return None
    epochs = mne.Epochs(
        raw, events, ev, tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
        baseline=BASELINE, preload=True, reject_by_annotation=True,
    )
    return epochs
# -

# +
OUT = Path("/kaggle/working/processed")
OUT.mkdir(exist_ok=True, parents=True)

def epochs_to_parquet(epochs, dataset_name, subject_id, label_map, out_dir):
    if epochs is None or len(epochs) == 0:
        return None
    data = epochs.get_data()  # (n, ch, time)
    ev = epochs.events[:, 2]
    labels = pd.Series(ev).map({v: k for k, v in label_map.items()}).fillna("unknown").tolist()
    rows = []
    for i, (arr, lab) in enumerate(zip(data, labels)):
        rows.append({
            "dataset": dataset_name,
            "subject": str(subject_id),
            "trial": i,
            "label": lab,
            "sfreq": float(epochs.info["sfreq"]),
            "ch_names": json.dumps(epochs.info["ch_names"]),
            "data": arr.astype("float32").tobytes(),
            "n_channels": int(arr.shape[0]),
            "n_times": int(arr.shape[1]),
        })
    df = pd.DataFrame(rows)
    out = out_dir / f"{dataset_name}_S{subject_id}.parquet"
    df.to_parquet(out, compression="zstd")
    return out
# -

# +
# Dataset: PhysioNet EEG Motor Movement/Imagery (eegmmidb) via mne.datasets.eegbci.
# 109 subjects, 14 runs each. Runs 4/8/12 = left/right hand MI;
# runs 6/10/14 = both hands/feet MI; runs 1/2 = baseline rest.
# Event codes in PhysioNet annotations: T0=rest, T1=left/both-hands, T2=right/feet.
from mne.datasets import eegbci

print("=== PhysioNet eegmmidb (via mne) ===")
N_SUBJECTS = 30   # caps Kaggle wall-time around ~30-45 min total

# Hand motor imagery runs only (T1 = left hand MI, T2 = right hand MI).
MI_HAND_RUNS = [4, 8, 12]
# Rest baseline (eyes open).
REST_RUNS = [1]

label_map_pn = {"rest": 0, "left_hand": 1, "right_hand": 2}

def load_pn_raws(subject, runs):
    """Download (cached) + load + concat raws for one subject."""
    files = eegbci.load_data(subject, runs, update_path=True, verbose="ERROR")
    raws = [mne.io.read_raw_edf(f, preload=True, verbose="ERROR") for f in files]
    raw = mne.concatenate_raws(raws, verbose="ERROR")
    # PhysioNet's channel names have trailing dots ("Fc3.."); normalize.
    eegbci.standardize(raw)
    raw.set_montage("standard_1005", on_missing="ignore", verbose="ERROR")
    return raw

for sid in range(1, N_SUBJECTS + 1):
    try:
        # MI hand runs
        raw_mi = load_pn_raws(sid, MI_HAND_RUNS)
        raw_mi_pp = preprocess_one(raw_mi, do_ica=False)
        # Map PhysioNet event codes (T0/T1/T2 -> 1/2/3 in mne annotations)
        # to our integer labels. mne.events_from_annotations enumerates alphabetically.
        events, ev_id = mne.events_from_annotations(raw_mi_pp, verbose="ERROR")
        # ev_id is typically {'T0':1, 'T1':2, 'T2':3}. Map: T0->rest, T1->left_hand, T2->right_hand.
        wanted = {}
        if "T1" in ev_id: wanted["left_hand"]  = ev_id["T1"]
        if "T2" in ev_id: wanted["right_hand"] = ev_id["T2"]
        # Epoch task trials.
        if wanted:
            ep = mne.Epochs(raw_mi_pp, events, event_id=wanted,
                            tmin=EPOCH_TMIN, tmax=EPOCH_TMAX, baseline=BASELINE,
                            preload=True, reject_by_annotation=True, verbose="ERROR")
            if len(ep) > 0:
                epochs_to_parquet(ep, "physionet", f"{sid}_mi", wanted, OUT)

        # Rest baseline: window the resting run into pseudo-trials.
        try:
            raw_rest = load_pn_raws(sid, REST_RUNS)
            raw_rest_pp = preprocess_one(raw_rest, do_ica=False)
            arr = raw_rest_pp.get_data()
            sfreq = raw_rest_pp.info["sfreq"]
            ch_names = raw_rest_pp.info["ch_names"]
            n_win = int((EPOCH_TMAX - EPOCH_TMIN) * sfreq)
            n_trials = arr.shape[1] // n_win
            if n_trials > 0:
                data = np.stack([arr[:, i*n_win:(i+1)*n_win] for i in range(n_trials)])
                # Wrap as a faux Epochs-like for parquet writer
                rows = []
                for i, A in enumerate(data):
                    rows.append({
                        "dataset": "physionet", "subject": f"{sid}_rest", "trial": i,
                        "label": "rest", "sfreq": float(sfreq),
                        "ch_names": json.dumps(ch_names),
                        "data": A.astype("float32").tobytes(),
                        "n_channels": int(A.shape[0]),
                        "n_times": int(A.shape[1]),
                    })
                pd.DataFrame(rows).to_parquet(OUT / f"physionet_S{sid}_rest.parquet",
                                              compression="zstd")
        except Exception as e:
            print(f"  subject {sid} rest FAIL: {e}")

        if sid % 5 == 0:
            print(f"  subject {sid}/{N_SUBJECTS} done")
    except Exception as e:
        print(f"  subject {sid} FAIL: {e}")
    gc.collect()
# -

# +
# Upload processed parquet to HF dataset repo (with retries).
print("Uploading to", HF_DATASET_REPO)
_retry(lambda: api.upload_folder(
    folder_path=str(OUT),
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
    commit_message="kaggle 00_preprocess publish",
), "upload_folder", tries=10, sleep_s=20)
print("DONE")
# -

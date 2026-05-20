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
_pip("pyriemann")
_pip("moabb")
_pip("mne-icalabel autoreject")
# -

# +
HF_TOKEN = os.environ["HF_TOKEN"]
HF_USER  = os.environ.get("HF_USERNAME", "abachu2005")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", f"{HF_USER}/mes-eeg-processed")
assert HF_TOKEN.startswith("hf_"), "HF_TOKEN missing or malformed"
# -

# +
import numpy as np
import pandas as pd
import mne
mne.set_log_level("ERROR")
from huggingface_hub import HfApi, create_repo, login
login(token=HF_TOKEN)
api = HfApi(token=HF_TOKEN)
create_repo(repo_id=HF_DATASET_REPO, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
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
# Dataset 1: BCI Competition IV 2a (small, sanity check) - load via MOABB.
from moabb.datasets import BNCI2014001
from moabb.paradigms import LeftRightImagery, MotorImagery

print("=== BCI IV 2a ===")
ds = BNCI2014001()
paradigm = MotorImagery(events=["left_hand","right_hand","feet","tongue"])
label_map_2a = {"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4}
for sid in ds.subject_list[:9]:
    try:
        X, y, meta = paradigm.get_data(dataset=ds, subjects=[sid])
        # Reconstruct as continuous raw per session via meta...
        # Simpler: use the epochs directly.
        # MOABB returns numpy arrays already epoched. We need to manually preprocess raws,
        # so use ds.get_data() to access raw mne objects:
        data_dict = ds.get_data([sid])
        for sess_name, runs in data_dict[sid].items():
            for run_name, raw in runs.items():
                raw.load_data()
                raw_pp = preprocess_one(raw)
                # Use built-in dataset event_id mapping; MOABB uses string codes
                ep = epoch_one(raw_pp, event_id=label_map_2a)
                if ep is not None and len(ep) > 0:
                    epochs_to_parquet(
                        ep, "bci_iv_2a", f"{sid}_{sess_name}_{run_name}",
                        label_map_2a, OUT,
                    )
        print(f"  subject {sid} done")
    except Exception as e:
        print(f"  subject {sid} FAIL: {e}")
    gc.collect()
# -

# +
# Dataset 2: PhysioNet eegmmidb - the heaviest healthy set.
from moabb.datasets import PhysionetMI

print("=== PhysioNet eegmmidb ===")
ds = PhysionetMI()
label_map_pn = {"left_hand": 1, "right_hand": 2, "feet": 3, "rest": 0}
for sid in ds.subject_list[:30]:  # first 30 subjects to fit Kaggle time budget
    try:
        data_dict = ds.get_data([sid])
        for sess_name, runs in data_dict[sid].items():
            for run_name, raw in runs.items():
                raw.load_data()
                raw_pp = preprocess_one(raw, do_ica=False)  # 64ch but ICLabel slow
                ep = epoch_one(raw_pp, event_id=label_map_pn)
                if ep is not None and len(ep) > 0:
                    epochs_to_parquet(
                        ep, "physionet", f"{sid}_{sess_name}_{run_name}",
                        label_map_pn, OUT,
                    )
        if sid % 5 == 0:
            print(f"  subject {sid} done")
    except Exception as e:
        print(f"  subject {sid} FAIL: {e}")
    gc.collect()
# -

# +
# Dataset 3: Liu2024 (stroke hand MI) - may fail if not yet in installed MOABB.
print("=== Liu2024 (stroke) ===")
try:
    from moabb.datasets import Liu2024
    ds = Liu2024()
    label_map_l = {"left_hand": 1, "right_hand": 2}
    for sid in ds.subject_list[:20]:  # first 20 to fit time budget
        try:
            data_dict = ds.get_data([sid])
            for sess_name, runs in data_dict[sid].items():
                for run_name, raw in runs.items():
                    raw.load_data()
                    raw_pp = preprocess_one(raw, do_ica=True)
                    ep = epoch_one(raw_pp, event_id=label_map_l)
                    if ep is not None and len(ep) > 0:
                        epochs_to_parquet(
                            ep, "liu2024", f"{sid}_{sess_name}_{run_name}",
                            label_map_l, OUT,
                        )
        except Exception as e:
            print(f"  subject {sid} FAIL: {e}")
        gc.collect()
except Exception as e:
    print("Liu2024 not available in this MOABB version, skipping:", e)
# -

# +
# Upload processed parquet to HF dataset repo.
print("Uploading to", HF_DATASET_REPO)
api.upload_folder(
    folder_path=str(OUT),
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
    commit_message="kaggle 00_preprocess publish",
)
print("DONE")
# -

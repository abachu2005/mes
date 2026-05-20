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

# # MES — 02_train_eegnet
#
# Trains EEGNet v4 via braindecode on the processed parquet data, exports to
# ONNX, and pushes to the HF model repo.
# Runs on Kaggle GPU (P100). ~2-4 hours.

# +
import os, json
from pathlib import Path

os.system("pip install -q --upgrade pip")

def _pip(pkgs):
    for _ in range(5):
        if os.system("pip install -q --default-timeout=60 --retries 5 " + pkgs) == 0:
            return
    raise RuntimeError(f"pip failed: {pkgs}")

_pip("'huggingface_hub>=0.24' pyarrow pandas onnx onnxruntime")
_pip("'braindecode>=0.8'")
# -

# +
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold
from huggingface_hub import HfApi, snapshot_download, login

HF_TOKEN = os.environ["HF_TOKEN"]
HF_USER  = os.environ.get("HF_USERNAME", "abachu2005")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", f"{HF_USER}/mes-eeg-processed")
HF_MODEL_REPO   = os.environ.get("HF_MODEL_REPO",   f"{HF_USER}/mes-models")

def _retry(fn, what, tries=8, sleep_s=15):
    for i in range(tries):
        try: return fn()
        except Exception as e:
            print(f"  HF {what} attempt {i+1}/{tries} failed: {e}")
            time.sleep(sleep_s)
    raise RuntimeError(f"HF {what} failed after {tries} retries")

_retry(lambda: login(token=HF_TOKEN, add_to_git_credential=False), "login")
api = HfApi(token=HF_TOKEN)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
# -

# +
ds_dir = Path(_retry(lambda: snapshot_download(repo_id=HF_DATASET_REPO,
                                                 repo_type="dataset", token=HF_TOKEN),
                      "snapshot_download", 10, 20))
files = sorted(ds_dir.glob("*.parquet"))

TARGET_T = 750  # 6 s @ 125 Hz
N_CH = 16
target_classes = {"right_hand": 1, "rest": 0}

def load_subset(prefix):
    rows = []
    for f in files:
        if not f.name.startswith(prefix):
            continue
        df = pd.read_parquet(f)
        for _, r in df.iterrows():
            if r["label"] not in target_classes:
                continue
            X = np.frombuffer(r["data"], dtype="float32").reshape(r["n_channels"], r["n_times"])
            n = X.shape[1]
            if n >= TARGET_T:
                X = X[:, :TARGET_T]
            else:
                X = np.concatenate([X, np.zeros((X.shape[0], TARGET_T-n), dtype="float32")], 1)
            rows.append({"X": X, "y": target_classes[r["label"]], "subject": r["subject"]})
    return rows

train_rows = load_subset("physionet_") or load_subset("bci_iv_2a_")
X = np.stack([r["X"] for r in train_rows])
y = np.array([r["y"] for r in train_rows])
subj = np.array([r["subject"] for r in train_rows])
print("X:", X.shape, "y dist:", dict(zip(*np.unique(y, return_counts=True))))
# -

# +
from braindecode.models import EEGNetv4

def make_model(n_classes=2):
    m = EEGNetv4(n_chans=N_CH, n_outputs=n_classes, n_times=TARGET_T, drop_prob=0.5)
    return m.to(device)

def train_one_fold(Xtr, ytr, Xte, yte, n_epochs=80, batch=64, lr=1e-3):
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    Xv = torch.tensor(Xte, dtype=torch.float32)
    yv = torch.tensor(yte, dtype=torch.long)
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=batch, shuffle=True)
    m  = make_model()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(n_epochs):
        m.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = m(xb)
            loss_fn(out, yb).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        pred = m(Xv.to(device)).argmax(1).cpu().numpy()
    acc = float((pred == yte).mean())
    return m, acc

gkf = GroupKFold(n_splits=min(5, len(np.unique(subj))))
fold_accs = []
final_model = None
best_acc = -1
for i, (tr, te) in enumerate(gkf.split(X, y, groups=subj)):
    m, a = train_one_fold(X[tr], y[tr], X[te], y[te])
    fold_accs.append(a)
    print(f"  fold {i}: {a:.3f}")
    if a > best_acc:
        best_acc = a
        final_model = m
print("LOSO mean acc:", float(np.mean(fold_accs)))
# -

# +
# Refit on all data + export.
final_model = make_model()
Xt = torch.tensor(X, dtype=torch.float32)
yt = torch.tensor(y, dtype=torch.long)
dl = DataLoader(TensorDataset(Xt, yt), batch_size=64, shuffle=True)
opt = torch.optim.Adam(final_model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
for ep in range(80):
    final_model.train()
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss_fn(final_model(xb), yb).backward()
        opt.step()

final_model.eval()
dummy = torch.zeros(1, N_CH, TARGET_T, dtype=torch.float32, device=device)
out_path = Path("eegnet_right_hand.onnx")
torch.onnx.export(
    final_model, dummy, str(out_path),
    input_names=["X"], output_names=["proba"],
    dynamic_axes={"X": {0: "batch"}, "proba": {0: "batch"}},
    opset_version=17,
)
meta = {
    "model": "eegnet_v4",
    "task": "right_hand_vs_rest",
    "input_kind": "raw_epochs",
    "n_channels": N_CH, "n_times": TARGET_T,
    "sfreq": 125.0, "n_classes": 2,
    "loso_acc_mean": float(np.mean(fold_accs)),
    "fold_accs": [float(a) for a in fold_accs],
    "n_train": int(X.shape[0]),
}
Path("eegnet_right_hand.json").write_text(json.dumps(meta, indent=2))

_retry(lambda: api.upload_file(path_or_fileobj=str(out_path), path_in_repo=out_path.name,
                repo_id=HF_MODEL_REPO, commit_message="eegnet v0.1"), "upload onnx", 10, 20)
_retry(lambda: api.upload_file(path_or_fileobj="eegnet_right_hand.json", path_in_repo="eegnet_right_hand.json",
                repo_id=HF_MODEL_REPO, commit_message="eegnet meta"), "upload meta", 10, 20)
print("DONE — pushed to", HF_MODEL_REPO)
# -

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

# # MES — 01_train_riemannian
#
# Trains a Riemannian + Logistic Regression baseline on the processed parquet
# data from HF, exports to ONNX, and pushes to the HF model repo.
# Runs on Kaggle CPU. ~30-60 min.

# +
import os, json, io
from pathlib import Path

os.system("pip install -q --upgrade pip")
os.system("pip install -q numpy scipy scikit-learn pyriemann 'huggingface_hub>=0.24' pyarrow pandas onnx skl2onnx onnxruntime")
# -

# +
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, snapshot_download, login

HF_TOKEN = os.environ["HF_TOKEN"]
HF_USER  = os.environ.get("HF_USERNAME", "abachu2005")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", f"{HF_USER}/mes-eeg-processed")
HF_MODEL_REPO   = os.environ.get("HF_MODEL_REPO",   f"{HF_USER}/mes-models")
login(token=HF_TOKEN)
api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=HF_MODEL_REPO, exist_ok=True)
# -

# +
ds_dir = Path(snapshot_download(repo_id=HF_DATASET_REPO, repo_type="dataset", token=HF_TOKEN))
files = sorted(ds_dir.glob("*.parquet"))
print(f"Found {len(files)} parquet files")

def load_subset(prefix, label_pairs):
    """Load all parquets whose name starts with `prefix`; keep only labels in `label_pairs`."""
    rows = []
    for f in files:
        if not f.name.startswith(prefix):
            continue
        df = pd.read_parquet(f)
        for _, r in df.iterrows():
            if r["label"] not in label_pairs:
                continue
            arr = np.frombuffer(r["data"], dtype="float32").reshape(r["n_channels"], r["n_times"])
            rows.append({"X": arr, "y": label_pairs[r["label"]], "subject": r["subject"]})
    return rows
# -

# +
# Build a binary classifier: right_hand vs rest (extendable to other pairs).
# Use PhysioNet for training (lots of subjects), keep BCI IV 2a held out for benchmark.
target_classes = {"right_hand": 1, "rest": 0}
train_rows = load_subset("physionet_", target_classes)
print(f"physionet trials: {len(train_rows)}")

if len(train_rows) < 20:
    # Fall back to BCI IV 2a if PhysioNet was empty
    target_classes = {"right_hand": 1, "left_hand": 0}
    train_rows = load_subset("bci_iv_2a_", target_classes)
    print(f"fallback to bci_iv_2a trials: {len(train_rows)}")

# Trim time to a common length (6 s at 125 Hz = 750 samples).
TARGET_T = 750
def crop_or_pad(X):
    n = X.shape[1]
    if n >= TARGET_T:
        return X[:, :TARGET_T]
    pad = np.zeros((X.shape[0], TARGET_T - n), dtype="float32")
    return np.concatenate([X, pad], axis=1)

X = np.stack([crop_or_pad(r["X"]) for r in train_rows])
y = np.array([r["y"] for r in train_rows])
subjects = np.array([r["subject"] for r in train_rows])
print("X shape:", X.shape, "y dist:", dict(zip(*np.unique(y, return_counts=True))))
# -

# +
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

def build_pipe():
    return Pipeline([
        ("cov", Covariances(estimator="oas")),
        ("tan", TangentSpace(metric="riemann")),
        ("lr",  LogisticRegression(C=1.0, max_iter=1000)),
    ])

gkf = GroupKFold(n_splits=min(5, len(np.unique(subjects))))
accs = []
for tr, te in gkf.split(X, y, groups=subjects):
    p = build_pipe()
    p.fit(X[tr], y[tr])
    accs.append(p.score(X[te], y[te]))
    print(f"  fold acc = {accs[-1]:.3f}")
mean_acc = float(np.mean(accs))
print(f"LOSO mean acc: {mean_acc:.3f}")
# -

# +
# Refit on all data with Platt-scaled calibration, then export to ONNX.
final = CalibratedClassifierCV(build_pipe(), method="sigmoid", cv=3)
final.fit(X, y)

# skl2onnx doesn't natively support pyriemann; manually compute the tangent-space
# features then export a scikit-learn pipeline over those features.
cov_tr = Covariances(estimator="oas").fit(X)
ts = TangentSpace(metric="riemann").fit(cov_tr.transform(X))
ts_X = ts.transform(cov_tr.transform(X))

post_pipe = Pipeline([("lr", LogisticRegression(C=1.0, max_iter=1000))])
post_pipe.fit(ts_X, y)
final_post = CalibratedClassifierCV(post_pipe, method="sigmoid", cv=3).fit(ts_X, y)

# Save covariance + tangent space transformers as a numpy bundle.
np.savez(
    "rieman_frontend.npz",
    cov_estimator=np.array([cov_tr.estimator_], dtype=object),
    ts_reference=ts.reference_,
)

from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
onx = to_onnx(
    final_post,
    initial_types=[("X", FloatTensorType([None, ts_X.shape[1]]))],
    target_opset={"": 17, "ai.onnx.ml": 3},
    options={id(final_post): {"zipmap": False}},
)
out_path = Path("riemannian_lr_right_hand.onnx")
out_path.write_bytes(onx.SerializeToString())

meta = {
    "model": "riemannian_lr",
    "task": "right_hand_vs_rest",
    "input_kind": "tangent_space",
    "n_features": int(ts_X.shape[1]),
    "n_classes": 2,
    "loso_acc": mean_acc,
    "n_train": int(X.shape[0]),
    "channels": 16,
    "sfreq": 125.0,
    "epoch_samples": int(X.shape[2]),
}
Path("riemannian_lr_right_hand.json").write_text(json.dumps(meta, indent=2))
print("ONNX written, meta:", meta)
# -

# +
# Push to HF model repo.
api.upload_file(path_or_fileobj=str(out_path),
                path_in_repo=out_path.name,
                repo_id=HF_MODEL_REPO,
                commit_message="riemannian baseline v0.1")
api.upload_file(path_or_fileobj="riemannian_lr_right_hand.json",
                path_in_repo="riemannian_lr_right_hand.json",
                repo_id=HF_MODEL_REPO,
                commit_message="riemannian baseline meta")
api.upload_file(path_or_fileobj="rieman_frontend.npz",
                path_in_repo="riemannian_lr_frontend.npz",
                repo_id=HF_MODEL_REPO,
                commit_message="riemannian frontend (cov+ts)")
print("DONE — pushed to", HF_MODEL_REPO)
# -

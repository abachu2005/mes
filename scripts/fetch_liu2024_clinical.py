#!/usr/bin/env python3
"""Fetch Liu2024 per-patient clinical table (NIHSS, MBI, paralysis side).

The full Figshare ``sourcedata.zip`` (~1.8 GB) contains ``participants.tsv``.
This script extracts only that file when the zip is present or downloaded.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LIU2024_SOURCEDATA_URL = "https://ndownloader.figshare.com/files/38516555"
DEFAULT_OUT = ROOT / "mes_core/data/liu2024_clinical.tsv"


def _find_participants_tsv(z: zipfile.ZipFile) -> str | None:
    for name in z.namelist():
        low = name.lower()
        if low.endswith("participants.tsv") and "participants" in low:
            return name
    return None


def extract_from_zip(zip_path: Path, out: Path) -> bool:
    with zipfile.ZipFile(zip_path) as z:
        member = _find_participants_tsv(z)
        if not member:
            return False
        raw = z.read(member).decode("utf-8", errors="replace")
    lines = raw.strip().splitlines()
    if len(lines) < 3:
        return False
    # Normalize to MES clinical TSV format.
    import io

    import pandas as pd

    df = pd.read_csv(io.StringIO(raw), sep="\t")
    rename = {
        "participant_id": "participant_id",
        "Participant_ID": "participant_id",
        "ParalysisSide": "paralysis_side",
        "paralysis_side": "paralysis_side",
        "NIHSS": "nihss",
        "MBI": "mbi",
        "mRS": "mrs",
    }
    cols = {c: rename.get(c, c.lower()) for c in df.columns}
    df = df.rename(columns=cols)
    if "participant_id" not in df.columns:
        return False
    df["participant_id"] = df["participant_id"].astype(str).str.replace("sub-", "", regex=False)
    df["participant_id"] = df["participant_id"].apply(
        lambda x: f"S{int(x)}" if x.isdigit() else x
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--zip-path",
        type=Path,
        default=Path.home() / "mne_data/MNE-liu2024-data/files/38516555",
        help="Local sourcedata.zip from MOABB/Figshare",
    )
    ap.add_argument("--download", action="store_true", help="Download sourcedata.zip first")
    args = ap.parse_args()

    zip_path = args.zip_path
    if args.download or not zip_path.exists():
        print("Downloading sourcedata.zip (~1.8 GB) — one-time…")
        from huggingface_hub import hf_hub_download  # noqa: F401 — optional

        import urllib.request

        cache = zip_path.parent
        cache.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(LIU2024_SOURCEDATA_URL, zip_path)

    if not zip_path.exists():
        print("Missing zip:", zip_path, file=sys.stderr)
        return 1
    if extract_from_zip(zip_path, args.output):
        print("Wrote", args.output)
        return 0
    print("participants.tsv not found inside zip", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

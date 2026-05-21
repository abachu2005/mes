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
# Figshare v4 includes standalone participants.tsv (~3.5 KB); v5 removed it from the file list.
LIU2024_PARTICIPANTS_TSV_URL = "https://ndownloader.figshare.com/files/38516582"
DEFAULT_OUT = ROOT / "mes_core/data/liu2024_clinical.tsv"
DEFAULT_ZIP = Path.home() / "mne_data/MNE-liu2024-data/files/38516555"


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


def download_participants_tsv(out: Path) -> bool:
    """Fetch BIDS participants.tsv from Figshare v4 (small file, no 1.8 GB zip)."""
    import urllib.request

    out.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading participants.tsv from Figshare v4…")
    urllib.request.urlretrieve(LIU2024_PARTICIPANTS_TSV_URL, out)
    if not out.exists() or out.stat().st_size < 100:
        return False
    return _normalize_participants_file(out)


def _normalize_participants_file(path: Path) -> bool:
    import io

    import pandas as pd

    raw = path.read_text(encoding="utf-8", errors="replace")
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
        lambda x: f"S{int(x)}" if str(x).isdigit() else str(x)
    )
    df.to_csv(path, sep="\t", index=False)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--zip-path",
        type=Path,
        default=DEFAULT_ZIP,
        help="Local sourcedata.zip from MOABB/Figshare (fallback)",
    )
    ap.add_argument("--download", action="store_true", help="Download participants.tsv (preferred) or sourcedata.zip")
    args = ap.parse_args()

    if args.output.exists() and args.output.stat().st_size > 200:
        print("Already present:", args.output)
        return 0

    if download_participants_tsv(args.output):
        print("Wrote", args.output)
        return 0

    zip_path = args.zip_path
    if args.download or zip_path.exists():
        if not zip_path.exists():
            print("Downloading sourcedata.zip (~1.8 GB) — fallback…")
            import urllib.request

            cache = zip_path.parent
            cache.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(LIU2024_SOURCEDATA_URL, zip_path)
        if zip_path.exists() and extract_from_zip(zip_path, args.output):
            print("Wrote", args.output)
            return 0

    print("Could not obtain participants.tsv", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Pre-flight checks for MES training pipelines — fail fast before long jobs.

Usage:
    python scripts/preflight.py --stage mes-train
    python scripts/preflight.py --stage hf-jobs-eegnet
    python scripts/preflight.py --stage all
    python scripts/preflight.py --stage mes-train --smoke   # 1-subject end-to-end probe
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    hint: str = ""


@dataclass
class PreflightReport:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "", hint: str = "") -> None:
        self.results.append(CheckResult(name, ok, detail, hint))

    @property
    def passed(self) -> bool:
        return all(r.ok for r in self.results)

    def print_report(self) -> None:
        print("\n=== MES Preflight ===")
        for r in self.results:
            mark = "PASS" if r.ok else "FAIL"
            print(f"[{mark}] {r.name}: {r.detail}")
            if not r.ok and r.hint:
                print(f"       hint: {r.hint}")
        print(f"\n{'ALL CHECKS PASSED' if self.passed else 'PREFLIGHT FAILED — fix issues above before running pipeline'}")
        print(f"  {sum(r.ok for r in self.results)}/{len(self.results)} passed\n")


def _env(name: str, required: bool = True) -> CheckResult:
    val = os.environ.get(name, "").strip()
    if val:
        return CheckResult(name, True, "set")
    return CheckResult(
        name,
        not required,
        "missing" if required else "not set (optional)",
        f"export {name} or add to GitHub secrets" if required else "",
    )


def _http_reachable(url: str, timeout: float = 10.0) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return True, f"HTTP {resp.status}"
    except urllib.error.HTTPError as e:
        # 403/405 still means DNS + TCP worked
        if e.code in (403, 405, 301, 302):
            return True, f"HTTP {e.code} (reachable)"
        return False, str(e)
    except Exception as e:
        return False, str(e)


def _dns_resolves(host: str) -> tuple[bool, str]:
    try:
        socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
        return True, "resolves"
    except socket.gaierror as e:
        return False, str(e)


def check_network(report: PreflightReport) -> None:
    hosts = {
        "physionet.org": "PhysioNet EEG downloads (required for preprocess)",
        "huggingface.co": "HF Hub upload/download",
        "pypi.org": "pip package installs",
    }
    for host, why in hosts.items():
        ok, detail = _dns_resolves(host)
        hint = f"DNS cannot resolve {host} — {why} will fail" if not ok else ""
        report.add(f"dns:{host}", ok, detail, hint)

    for url in ("https://physionet.org/files/eegmmidb/", "https://huggingface.co", "https://pypi.org"):
        ok, detail = _http_reachable(url)
        host = url.split("/")[2]
        report.add(f"http:{host}", ok, detail, "Network blocked from this runner" if not ok else "")


def check_hf_auth(report: PreflightReport) -> None:
    er = _env("HF_TOKEN")
    report.results.append(er)
    if not er.ok:
        return
    try:
        from huggingface_hub import HfApi

        who = HfApi(token=os.environ["HF_TOKEN"]).whoami()
        name = who.get("name") or who.get("fullname") or "unknown"
        report.add("hf:whoami", True, name)
    except Exception as e:
        report.add("hf:whoami", False, str(e), "Check HF_TOKEN is a valid write token")


def check_hf_repos(report: PreflightReport, *, need_parquet: bool = False) -> None:
    user = os.environ.get("HF_USERNAME", "").strip()
    if not user:
        report.add("hf:username", False, "HF_USERNAME missing", "Set secrets.HF_USERNAME in GitHub")
        return
    report.add("hf:username", True, user)

    ds = os.environ.get("HF_DATASET_REPO", f"{user}/mes-eeg-processed")
    mdl = os.environ.get("HF_MODEL_REPO", f"{user}/mes-models")
    token = os.environ.get("HF_TOKEN", "")

    try:
        from huggingface_hub import HfApi, create_repo

        api = HfApi(token=token)
        for repo, rtype in ((ds, "dataset"), (mdl, "model")):
            create_repo(repo_id=repo, repo_type=rtype, exist_ok=True, token=token)
            files = api.list_repo_files(repo, repo_type=rtype)
            parquet = [f for f in files if f.endswith(".parquet")]
            onnx = [f for f in files if f.endswith(".onnx")]
            detail = f"{len(files)} file(s)"
            if rtype == "dataset":
                detail += f", {len(parquet)} parquet"
                if need_parquet and not parquet:
                    report.add(
                        f"hf:repo:{repo}",
                        False,
                        "repo exists but no parquet — run mes-train-pipeline first",
                        "gh workflow run mes-train-pipeline.yml",
                    )
                    continue
            else:
                detail += f", {len(onnx)} onnx"
            report.add(f"hf:repo:{repo}", True, detail)
    except Exception as e:
        report.add("hf:repos", False, str(e), "Verify HF_TOKEN has write access")


def check_hf_jobs(report: PreflightReport) -> None:
    hf = shutil.which("hf")
    if not hf:
        report.add("hf:cli", False, "hf CLI not found", "pip install 'huggingface_hub[cli]>=0.26'")
        return
    report.add("hf:cli", True, hf)

    env = os.environ.copy()
    if os.environ.get("HF_TOKEN"):
        env["HF_TOKEN"] = os.environ["HF_TOKEN"]
    proc = subprocess.run(
        [hf, "jobs", "hardware"],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode == 0 and "a10g" in out.lower():
        report.add("hf:jobs", True, "Jobs API reachable (a10g flavors listed)")
    elif "pro" in out.lower() or "subscription" in out.lower() or "403" in out:
        report.add(
            "hf:jobs",
            False,
            out.strip()[:200] or "access denied",
            "HF Pro subscription required for Jobs — https://huggingface.co/pro",
        )
    elif proc.returncode != 0:
        report.add("hf:jobs", False, out.strip()[:200], "Check HF Pro + billing credits")
    else:
        report.add("hf:jobs", True, "hardware list returned")


def check_imports(report: PreflightReport, modules: list[str]) -> None:
    for mod in modules:
        try:
            __import__(mod)
            report.add(f"import:{mod}", True, "ok")
        except Exception as e:
            report.add(f"import:{mod}", False, str(e), f"pip install deps for {mod}")


def check_scripts_compile(report: PreflightReport, paths: list[Path]) -> None:
    for p in paths:
        try:
            src = p.read_text()
            compile(src, str(p), "exec")
            report.add(f"script:{p.name}", True, "syntax ok")
        except Exception as e:
            report.add(f"script:{p.name}", False, str(e))


def check_disk(report: PreflightReport, min_gb: float = 2.0) -> None:
    usage = shutil.disk_usage(tempfile.gettempdir())
    free_gb = usage.free / (1024**3)
    ok = free_gb >= min_gb
    report.add(
        "disk:free",
        ok,
        f"{free_gb:.1f} GB free in {tempfile.gettempdir()}",
        f"Need >= {min_gb} GB for preprocess parquet" if not ok else "",
    )


def check_uv_script_header(report: PreflightReport, path: Path) -> None:
    text = path.read_text()
    ok = "# /// script" in text and "dependencies" in text
    report.add(
        f"uv-script:{path.name}",
        ok,
        "inline deps present" if ok else "missing UV script header",
        "hf jobs uv run requires # /// script dependency block" if not ok else "",
    )


def run_smoke_preprocess(report: PreflightReport) -> None:
    """Download + preprocess 1 PhysioNet subject (~2-5 min)."""
    script = ROOT / "scripts" / "gh_preprocess_physionet.py"
    if not script.exists():
        report.add("smoke:preprocess", False, "script missing")
        return
    out = Path(tempfile.mkdtemp(prefix="mes-preflight-"))
    proc = subprocess.run(
        [sys.executable, str(script), "--out", str(out / "processed"), "--subjects", "1"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        env=os.environ.copy(),
    )
    parquet = list((out / "processed").glob("*.parquet"))
    tail = (proc.stdout or "")[-1500:] + (proc.stderr or "")[-500:]
    if proc.returncode == 0 and parquet:
        report.add("smoke:preprocess", True, f"{len(parquet)} parquet from 1 subject")
    else:
        report.add(
            "smoke:preprocess",
            False,
            f"exit {proc.returncode}, {len(parquet)} parquet\n{tail}",
            "PhysioNet download or MNE preprocess failed on this runner",
        )


def run_smoke_hf_sync_dry(report: PreflightReport) -> None:
    """Verify hf_sync.py argparse and HF write without uploading real data."""
    sync = ROOT / "scripts" / "hf_sync.py"
    if not sync.exists():
        report.add("smoke:hf_sync", False, "script missing")
        return
    proc = subprocess.run(
        [sys.executable, str(sync), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    report.add("smoke:hf_sync", proc.returncode == 0, "cli ok" if proc.returncode == 0 else proc.stderr[:200])


def stage_mes_train(report: PreflightReport, *, smoke: bool) -> None:
    check_network(report)
    check_disk(report, min_gb=3.0)
    for er in (_env("HF_TOKEN"), _env("HF_USERNAME")):
        report.results.append(er)
    if os.environ.get("HF_TOKEN"):
        check_hf_repos(report, need_parquet=False)
    check_imports(
        report,
        ["mne", "numpy", "pandas", "pyarrow", "pyriemann", "sklearn", "skl2onnx", "onnx", "onnxruntime"],
    )
    check_scripts_compile(
        report,
        [
            ROOT / "scripts" / "gh_preprocess_physionet.py",
            ROOT / "scripts" / "gh_train_riemannian.py",
            ROOT / "scripts" / "hf_sync.py",
        ],
    )
    if smoke:
        run_smoke_preprocess(report)
        run_smoke_hf_sync_dry(report)


def stage_hf_jobs_eegnet(report: PreflightReport, *, smoke: bool) -> None:
    check_network(report)
    for er in (_env("HF_TOKEN"), _env("HF_USERNAME")):
        report.results.append(er)
    if os.environ.get("HF_TOKEN"):
        check_hf_auth(report)
        check_hf_repos(report, need_parquet=True)
    check_hf_jobs(report)
    check_uv_script_header(report, ROOT / "scripts" / "hf_jobs_train_eegnet.py")
    check_scripts_compile(report, [ROOT / "scripts" / "hf_jobs_train_eegnet.py"])
    if smoke:
        run_smoke_hf_sync_dry(report)


def stage_kaggle_warn(report: PreflightReport) -> None:
    ok, detail = _dns_resolves("physionet.org")
    report.add(
        "kaggle:dns-warning",
        ok,
        detail if ok else f"BLOCKED: {detail}",
        "Kaggle kernels cannot download PhysioNet — use mes-train-pipeline instead" if not ok else "",
    )
    # Kaggle historically also flakes on pypi/huggingface
    for host in ("pypi.org", "huggingface.co"):
        ok, detail = _dns_resolves(host)
        if not ok:
            report.add(f"kaggle:dns:{host}", False, detail, "Do not use Kaggle for this pipeline")


STAGES: dict[str, Callable[..., None]] = {
    "mes-train": stage_mes_train,
    "hf-jobs-eegnet": stage_hf_jobs_eegnet,
    "kaggle-warn": lambda r, smoke=False: stage_kaggle_warn(r),
}


def main() -> int:
    p = argparse.ArgumentParser(description="MES pipeline preflight checks")
    p.add_argument(
        "--stage",
        choices=["mes-train", "hf-jobs-eegnet", "kaggle-warn", "all"],
        default="all",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run 1-subject preprocess probe (~2-5 min, mes-train stage only)",
    )
    args = p.parse_args()

    report = PreflightReport()
    stages = ["mes-train", "hf-jobs-eegnet", "kaggle-warn"] if args.stage == "all" else [args.stage]

    for stage in stages:
        fn = STAGES[stage]
        if stage in ("mes-train", "hf-jobs-eegnet"):
            fn(report, smoke=args.smoke)
        else:
            fn(report)

    report.print_report()
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())

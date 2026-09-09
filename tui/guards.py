"""The checks that stop a convenient launcher from producing a CPU run in an NPU costume.

Every check here shells out to the same tool scripts/lib.sh uses -- xrt-smi.exe and
tools/host_load.ps1 -- so the tool stays the single source of truth and only the thin
parse is duplicated. The EP report is read through npu.ep_report, the validated reader;
lib.sh's npu_verdict already re-implements that count inline and a third copy would be
one too many.

Nothing in this module opens a hardware context. It is safe to call from --selftest.
"""
import json
import os
import re
import shutil
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path

from npu import ep_report
from npu.paths import ROOT

XRT_SMI = Path(r"C:\Windows\System32\AMD\xrt-smi.exe")
HOST_LOAD_PS1 = ROOT / "tools" / "host_load.ps1"

# Which machine is this? CLAUDE.md: work has already been handed to the wrong box
# because a session guessed from what happened to be installed on it. An unrecorded
# hostname is reported as unknown, never inferred.
MACHINES = {
    "DESKTOP-CBL5NUA": ("Desktop 2", "Ryzen 7 8700G / Radeon 780M, Phoenix XDNA1, 32 GB", True),
}

OK, WARN, FAIL, UNKNOWN = "ok", "warn", "fail", "unknown"


@dataclass
class Check:
    name: str
    status: str
    detail: str

    @property
    def blocks_npu(self) -> bool:
        return self.status == FAIL


# --------------------------------------------------------------------------
# Startup checks
# --------------------------------------------------------------------------

def check_machine() -> Check:
    host = socket.gethostname().upper()
    if host in MACHINES:
        name, spec, has_npu = MACHINES[host]
        if has_npu:
            return Check("machine", OK, f"{host} = {name} ({spec})")
        return Check("machine", FAIL, f"{host} = {name} -- no XDNA device; NPU runs are impossible here")
    return Check("machine", UNKNOWN,
                 f"{host} is not in the machine table -- NPU capability unverified. "
                 f"Add it to tui/guards.MACHINES rather than guessing.")


def check_env() -> Check:
    env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if env == "resnet_env17":
        return Check("conda env", OK, "resnet_env17")
    if not env:
        return Check("conda env", FAIL,
                     "no conda env active. Only resnet_env17 can run on the NPU "
                     "(1.8.0 ships no Phoenix xclbin).")
    return Check("conda env", FAIL,
                 f"{env} is active, not resnet_env17. Do not run NPU inference from here.")


def _xclbin_dir() -> Path:
    return Path(os.environ.get("RYZEN_AI_INSTALLATION_PATH", "")) / "voe-4.0-win_amd64" / "xclbins" / "phoenix"


def check_xclbin(name="4x4.xclbin", label="xclbin (4x4)") -> Check:
    install = os.environ.get("RYZEN_AI_INSTALLATION_PATH", "")
    if not install:
        return Check(label, FAIL,
                     "RYZEN_AI_INSTALLATION_PATH is unset. In PowerShell:\n"
                     r"  $env:RYZEN_AI_INSTALLATION_PATH = 'C:\Program Files\RyzenAI\1.7.1'")
    candidate = _xclbin_dir() / name
    if candidate.is_file():
        warn = "" if "1.7.1" in install else "  (this path is not 1.7.1 -- check it)"
        return Check(label, OK if not warn else WARN, str(candidate) + warn)
    return Check(label, FAIL,
                 f"no {name} at {candidate}. 1.8.0 ships none; only 1.7.1 has it.")


def check_contention() -> Check:
    """Is another session already holding the AIE tiles? Mirrors lib.sh:check_npu_contention."""
    if not XRT_SMI.is_file():
        return Check("NPU contention", UNKNOWN, f"{XRT_SMI} not found -- cannot check")
    try:
        out = subprocess.run([str(XRT_SMI), "examine", "-r", "aie-partitions"],
                             capture_output=True, text=True, timeout=60).stdout
    except (subprocess.SubprocessError, OSError) as exc:
        return Check("NPU contention", UNKNOWN, f"xrt-smi failed: {exc}")
    if "No hardware contexts running" in out:
        return Check("NPU contention", OK, "no hardware contexts running")
    rows = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return Check("NPU contention", WARN,
                 "another process holds the device -- a run now measures contention, "
                 "not the model:\n  " + "\n  ".join(rows[:8]))


def check_host_load() -> Check:
    """CLEAR / BUSY / PEER. Mirrors lib.sh:check_host_load in warn mode."""
    if not HOST_LOAD_PS1.is_file() or not shutil.which("powershell"):
        return Check("host load", UNKNOWN, "no powershell, or tools/host_load.ps1 missing")
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(HOST_LOAD_PS1)],
            capture_output=True, text=True, cwd=str(ROOT), timeout=120).stdout
    except (subprocess.SubprocessError, OSError) as exc:
        return Check("host load", UNKNOWN, f"host_load.ps1 failed: {exc}")
    verdict = ""
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "HOST_LOAD_VERDICT":
            verdict = parts[1]
    if verdict == "CLEAR":
        return Check("host load", OK, "nothing else is competing for the CPU")
    if verdict == "BUSY":
        tops = [ln.split(" ", 1)[1] for ln in out.splitlines()
                if ln.startswith("HOST_LOAD_TOP ")][:3]
        return Check("host load", WARN, "host is busy: " + "; ".join(tops))
    if verdict == "PEER":
        peers = [ln.split(" ", 1)[1] for ln in out.splitlines()
                 if ln.startswith("HOST_LOAD_PEER ")]
        return Check("host load", WARN,
                     "another heavy job holds the CPU: " + "; ".join(peers))
    return Check("host load", UNKNOWN, "host_load.ps1 gave no verdict")


def startup_checks(need_1x4=False) -> list:
    checks = [check_machine(), check_env(), check_xclbin()]
    if need_1x4:
        checks.append(check_xclbin("1x4.xclbin", "xclbin (1x4)"))
    checks += [check_contention(), check_host_load()]
    return checks


# --------------------------------------------------------------------------
# Compile-cache staleness -- exact, from the cache's own record
# --------------------------------------------------------------------------
# The cache is keyed by NAME, not by model hash, so a different model in the same
# family silently reuses the previous compile. That is normally undetectable, but
# <cacheKey>/context.json records config.onnxPath -- the model that compile was
# actually built from. Present in all 31 caches on this machine, so the check is a
# pure read with no sidecar state to drift.

def cache_source_model(cache_key: str):
    """The model a cache was compiled from, or None if unknown/absent."""
    ctx = ROOT / cache_key / "context.json"
    if not ctx.is_file():
        return None
    try:
        return json.loads(ctx.read_text(encoding="utf-8")).get("config", {}).get("onnxPath")
    except (OSError, ValueError):
        return None


def _normalise(p: str):
    """context.json records paths three ways: relative posix, relative windows, and
    absolute. Resolve them all against ROOT so a comparison means something."""
    if not p:
        return None
    candidate = Path(p.replace("\\", "/"))
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    try:
        return candidate.resolve()
    except OSError:
        return candidate


def cache_status(cache_key: str, model_path) -> Check:
    """Would running `model_path` under `cache_key` reuse someone else's compile?"""
    recorded = cache_source_model(cache_key)
    if recorded is None:
        if (ROOT / cache_key).is_dir():
            return Check(f"cache {cache_key}", UNKNOWN,
                         "cache exists but records no source model -- compile fresh to be sure")
        return Check(f"cache {cache_key}", OK, "no cache yet; this run compiles from scratch")
    want, have = _normalise(str(model_path)), _normalise(recorded)
    if want == have:
        return Check(f"cache {cache_key}", OK, f"already compiled from {recorded}")
    return Check(f"cache {cache_key}", WARN,
                 f"STALE -- holds a compile of {recorded}, not the model you picked. "
                 f"Reusing it would run the wrong weights with no error. Use --fresh.")


def is_stale(cache_key: str, model_path) -> bool:
    return cache_status(cache_key, model_path).status == WARN


# --------------------------------------------------------------------------
# Did the EP actually engage?
# --------------------------------------------------------------------------

@dataclass
class Verdict:
    status: str          # ok | zero | missing | unreadable
    npu: int = 0
    total: int = 0
    sha256: str = ""
    detail: str = ""

    def __str__(self):
        if self.status == "ok":
            return f"{self.npu}/{self.total} nodes on the NPU -- the EP engaged"
        if self.status == "zero":
            return f"0/{self.total} nodes -- the EP registered and claimed nothing (this ran on CPU)"
        if self.status == "missing":
            return "no EP report -- the EP never ran, or XLNX_ONNX_EP_REPORT_FILE was unset"
        return f"EP report unreadable: {self.detail}"


def ep_verdict(cache_key: str) -> Verdict:
    """Read <cacheKey>/vitisai_ep_report.json. This, not a `_npu` filename and not a
    provider that registered, is the evidence that the NPU did the work."""
    path = ROOT / cache_key / "vitisai_ep_report.json"
    if not path.is_file():
        return Verdict("missing")
    try:
        summary = ep_report.read_report(path).summary()
    except (OSError, ValueError) as exc:
        return Verdict("unreadable", detail=str(exc))
    npu, total = summary["npu"], summary["total"]
    return Verdict("ok" if npu else "zero", npu, total, summary["report_sha256"])


def all_caches() -> list:
    """Every compile cache at the repo root, with what it holds. Read-only."""
    rows = []
    for d in sorted(ROOT.iterdir()):
        if not d.is_dir() or not (d / "context.json").is_file():
            continue
        rows.append((d.name, cache_source_model(d.name), ep_verdict(d.name)))
    return rows

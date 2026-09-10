"""Record which torch device a run can reach, and what else is on the GPU while it runs.

A GPU AdaRound arm (`scripts/quant-adaround.sh --device cuda`) needs two facts in its log
that the ADAROUND header does not carry. First, which physical adapter torch's "cuda"
resolved to: a ROCm build answers to "cuda", and a desktop Ryzen carries an iGPU that is a
second AMD adapter the runtime may enumerate. Second, whether anything else was using that
GPU during the timing. tools/host_load.ps1 answers that for the CPU only; this is the GPU
counterpart, read from the Windows "GPU Engine" performance counters, which see every
process on every adapter. The same counters are also the positive evidence that the run
engaged the GPU at all: a python pid with Compute or 3D load on the discrete card's luid.

    python tools/torch_device_info.py                  # DEVICE_INFO block
    python tools/torch_device_info.py --gpu-load       # ... plus one GPU_LOAD snapshot
    python tools/torch_device_info.py --load-only      # the snapshot alone
    python tools/torch_device_info.py --watch 30       # a snapshot every 30 s until killed

Records, one per line, stable enough to grep:

    DEVICE_INFO <key>=<value>
    GPU_MEM t=<hh:mm:ss> luid=<adapter> dedicated_mb=<f>        tells the discrete card apart
    GPU_LOAD t=<hh:mm:ss> luid=<adapter> engtype=<type> pct=<f> summed over processes/engines
    GPU_LOAD_TOP t=<hh:mm:ss> pid=<pid> name=<proc> luid=<adapter> engtype=<type> pct=<f>

The GPU records are Windows-only (the counters are WDDM's); DEVICE_INFO works anywhere.
A failure to read them prints `GPU_LOAD unavailable` and exits 0, because a witness that
kills the run it is watching is worse than a missing one. Beyond what
torch.cuda.get_device_properties does, it opens nothing on the GPU.
"""
from __future__ import annotations

import argparse
import os
import platform
import re
import socket
import subprocess
import sys
import time
from collections import defaultdict

INSTANCE = re.compile(r"pid_(\d+)_luid_(0x[0-9a-f]+_0x[0-9a-f]+)_phys_\d+_eng_\d+_engtype_(.*)$", re.I)
ADAPTER = re.compile(r"luid_(0x[0-9a-f]+_0x[0-9a-f]+)_phys_\d+$", re.I)

# One PowerShell call per snapshot: engine utilisation over SAMPLE seconds, each adapter's
# dedicated memory, and the names of the pids that showed any load.
PS_SNAPSHOT = r"""
$ErrorActionPreference = 'Stop'
$e = (Get-Counter '\GPU Engine(*)\Utilization Percentage' -SampleInterval SAMPLE -MaxSamples 1).CounterSamples
$busy = $e | Where-Object { $_.CookedValue -gt 0.05 }
foreach ($c in $busy) { "E`t$($c.InstanceName)`t$($c.CookedValue)" }
foreach ($c in (Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage').CounterSamples) { "M`t$($c.InstanceName)`t$($c.CookedValue)" }
$ids = $busy | ForEach-Object { if ($_.InstanceName -match '^pid_(\d+)_') { [int]$Matches[1] } } | Sort-Object -Unique
if ($ids) { Get-Process -Id $ids -ErrorAction SilentlyContinue | ForEach-Object { "P`t$($_.Id)`t$($_.ProcessName)" } }
"""


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def module_version(name: str) -> str:
    try:
        return str(getattr(__import__(name), "__version__", "unknown"))
    except Exception as exc:  # a missing package is a finding, not a crash
        return f"unavailable({type(exc).__name__}: {exc})"


def device_info() -> list[tuple[str, object]]:
    rows: list[tuple[str, object]] = [
        ("hostname", socket.gethostname()), ("cpu", cpu_name()), ("logical_cpus", os.cpu_count()),
        ("conda_env", os.environ.get("CONDA_DEFAULT_ENV", "none")), ("python", platform.python_version()),
        ("omp_num_threads", os.environ.get("OMP_NUM_THREADS", "unset"))]
    rows += [(name, module_version(name)) for name in ("numpy", "onnx", "onnxruntime")]
    try:
        import torch
    except Exception as exc:
        return rows + [("torch", f"unavailable({type(exc).__name__}: {exc})")]
    rows += [("torch", torch.__version__), ("torch_hip", getattr(torch.version, "hip", None)),
             ("torch_cuda", torch.version.cuda), ("torch_threads", torch.get_num_threads()),
             ("cuda_available", torch.cuda.is_available())]
    if torch.cuda.is_available():
        rows += [("cuda_device_count", torch.cuda.device_count()),
                 ("cuda_current_device", torch.cuda.current_device())]
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            rows.append((f"cuda_device_{i}", f"{p.name} total_mb={p.total_memory // 2**20} "
                                             f"arch={getattr(p, 'gcnArchName', 'unknown')}"))
    return rows


def gpu_snapshot(sample_s: int) -> list[str]:
    stamp = time.strftime("%H:%M:%S")
    try:
        done = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                               PS_SNAPSHOT.replace("SAMPLE", str(sample_s))],
                              capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=60 + sample_s)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [f"GPU_LOAD unavailable t={stamp} reason={type(exc).__name__}"]
    if done.returncode != 0:
        reason = (done.stderr.strip().splitlines() or ["no stderr"])[-1]
        return [f"GPU_LOAD unavailable t={stamp} reason={reason}"]
    names: dict[str, str] = {}
    engines: list[tuple[str, str, str, float]] = []
    lines: list[str] = []
    for raw in done.stdout.splitlines():
        kind, _, rest = raw.partition("\t")
        key, _, value = rest.partition("\t")
        if kind == "P":
            names[key] = value
        elif kind == "M" and (m := ADAPTER.search(key)):
            lines.append(f"GPU_MEM t={stamp} luid={m.group(1)} dedicated_mb={float(value) / 2**20:.0f}")
        elif kind == "E" and (m := INSTANCE.search(key)):
            engines.append((m.group(1), m.group(2), m.group(3) or "none", float(value)))
    per_type: dict[tuple[str, str], float] = defaultdict(float)
    per_proc: dict[tuple[str, str, str], float] = defaultdict(float)
    for pid, luid, engtype, pct in engines:
        per_type[(luid, engtype)] += pct
        per_proc[(pid, luid, engtype)] += pct
    if not per_type:
        lines.append(f"GPU_LOAD t={stamp} luid=all engtype=all pct=0.00")
    for (luid, engtype), pct in sorted(per_type.items()):
        lines.append(f"GPU_LOAD t={stamp} luid={luid} engtype={engtype} pct={pct:.2f}")
    top = sorted(per_proc.items(), key=lambda item: -item[1])[:6]
    for (pid, luid, engtype), pct in top:
        if pct >= 0.5:
            lines.append(f"GPU_LOAD_TOP t={stamp} pid={pid} name={names.get(pid, '?')} luid={luid} "
                         f"engtype={engtype} pct={pct:.2f}")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gpu-load", action="store_true", help="append one GPU_LOAD snapshot")
    ap.add_argument("--load-only", action="store_true", help="the snapshot without the DEVICE_INFO block")
    ap.add_argument("--watch", type=int, metavar="SEC", default=0,
                    help="print a snapshot every SEC seconds until killed (implies --load-only)")
    ap.add_argument("--sample", type=int, default=2, help="counter sampling window per snapshot, seconds")
    args = ap.parse_args(argv)
    if not (args.load_only or args.watch):
        for key, value in device_info():
            print(f"DEVICE_INFO {key}={value}", flush=True)
    if args.watch:
        while True:
            start = time.monotonic()
            print("\n".join(gpu_snapshot(args.sample)), flush=True)
            time.sleep(max(0.0, args.watch - (time.monotonic() - start)))
    if args.gpu_load or args.load_only:
        print("\n".join(gpu_snapshot(args.sample)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

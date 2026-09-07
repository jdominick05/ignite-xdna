"""
Feed HWiNFO64 real XDNA1 NPU telemetry via its documented Custom Sensors
registry interface, in place of the values HWiNFO reports natively on this
hardware: voltage stuck at 0.001 V, clock stuck at a flat 800 MHz, and
D3D/Compute0 usage stuck at 0% -- all sourced from Windows' generic MCDM
GPU-Engine-style counters, which never see load placed on the NPU the way
this repo's pipelines place it (VitisAI EP -> XRT hardware-context
submission, not a DirectML/WinML queue). Measured live on this Phoenix box
2026-09-06/07: xrt-smi's own aie-partitions report showed GOPS=9 and 93 MB
in use from a held session while HWiNFO's D3D/Compute0 read 0% the entire
time.

What this does NOT attempt, and why:
  - Voltage/power: `xrt-smi examine -r platform` itself reports
    "Estimated Power: N/A" on this NPU (Phoenix) -- AMD's own docs say
    power reporting is unsupported on PHX/HPT and only shipped from STX
    onward. The JSON report shows why at the driver level: the "electrical"
    block's query fails outright ("Failed to escape (0xc0000023): The data
    area passed to a system call is too small"), i.e. the underlying IOCTL
    errors, not merely "not wired up in the CLI." There is no documented,
    legitimate source for this number on XDNA1; inventing one would mean
    guessing at undocumented SMU mailbox registers, which this repo's own
    rules treat as unverified and risky. HWiNFO's 0.001 V is a placeholder
    reading a rail table slot that doesn't correspond to anything real
    here -- not a bug this script can fix, so it isn't emitted at all
    rather than replaced with more noise.
  - Clock: not present in any xrt-smi report either. HWiNFO's flat 800 MHz
    is almost certainly a static nameplate value, not a live PLL readback.
    No documented API for it was found; left unreported for the same
    reason as voltage.

What it DOES report, all sourced from `xrt-smi examine`, AMD's own signed
tool, polled the same way tools/session_hold.py already validates:
  - GOPS / EGOPS   -- xrt-smi's own per-hardware-context compute-rate report
  - Completions/s  -- delta of xrt-smi's command_completions counter between
                       polls: real submitted-and-finished work, independent
                       of what HWiNFO's engine-counter path can see
  - Columns active -- how many of the array's columns are allocated to a
                       running context right now (a real, live number, not a
                       static core count)
  - NPU memory MB  -- xrt-smi's own total_memory_usage figure

Usage:
    python tools/hwinfo_npu_bridge.py                  # poll every 2s, HKCU
    python tools/hwinfo_npu_bridge.py --interval 5
    python tools/hwinfo_npu_bridge.py --once            # one sample, print, exit

Requires HWiNFO64 to be running with its Sensors window open (Settings ->
"Enable Custom Sensors" is on by default since v6.10) so it picks up the
registry keys this script writes. No conda env needed -- stdlib only.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
import winreg
from pathlib import Path

XRT_SMI = r"C:\Windows\System32\AMD\xrt-smi.exe"
REG_ROOT = r"Software\HWiNFO64\Sensors\Custom"


def run_report(report, out_path):
    subprocess.run(
        [XRT_SMI, "examine", "-r", report, "-f", "JSON", "-o", str(out_path), "--force"],
        capture_output=True, text=True, timeout=15, check=True,
    )
    return json.loads(out_path.read_text())


def parse_mb(s):
    m = re.match(r"(\d+)\s*MB", s or "")
    return int(m.group(1)) if m else 0


def device_info(tmp_dir):
    """One-time platform lookup: device display name and total AIE columns."""
    data = run_report("platform", tmp_dir / "platform.json")
    static_region = data["devices"][0]["platforms"][0]["static_region"]
    name = (static_region.get("name") or "XDNA NPU").strip()
    total_cols = int(static_region.get("total_columns", 0) or 0)
    return name, total_cols


def sample_aie(tmp_dir):
    """Sum GOPS/EGOPS/completions/columns/memory across every active
    hardware context. Returns zeros when the device is idle."""
    data = run_report("aie-partitions", tmp_dir / "aie.json")
    dev = data["devices"][0]["aie_partitions"]
    mem_mb = parse_mb(dev.get("total_memory_usage", ""))
    partitions = dev.get("partitions") or []
    gops = egops = completions = cols_active = 0
    for part in partitions:
        cols_active += int(part.get("num_cols", 0) or 0)
        for ctx in part.get("hw_contexts") or []:
            if ctx.get("status") != "Active":
                continue
            gops += int(ctx.get("gops", 0) or 0)
            egops += int(ctx.get("egops", 0) or 0)
            completions += int(ctx.get("command_completions", 0) or 0)
    return dict(gops=gops, egops=egops, completions=completions,
                cols_active=cols_active, mem_mb=mem_mb)


def write_sensor(group, index, name, unit, value):
    key_path = f"{REG_ROOT}\\{group}\\Other{index}"
    key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path)
    winreg.SetValueEx(key, "Name", 0, winreg.REG_SZ, name)
    winreg.SetValueEx(key, "Unit", 0, winreg.REG_SZ, unit)
    winreg.SetValueEx(key, "Value", 0, winreg.REG_SZ, f"{value:.2f}")
    winreg.CloseKey(key)


def publish(group, total_cols, sample, completions_per_sec):
    util_pct = 100.0 * sample["cols_active"] / total_cols if total_cols else 0.0
    rows = [
        ("NPU GOPS", "GOPS", sample["gops"]),
        ("NPU EGOPS", "GOPS", sample["egops"]),
        ("NPU Completions", "/s", completions_per_sec),
        ("NPU Columns Active", "cols", sample["cols_active"]),
        ("NPU Array Utilization", "%", util_pct),
        ("NPU Memory", "MB", sample["mem_mb"]),
    ]
    for i, (name, unit, value) in enumerate(rows):
        write_sensor(group, i, name, unit, value)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval", type=float, default=2.0, help="seconds between polls (default 2)")
    ap.add_argument("--group", default=None, help="override the HWiNFO sensor group name")
    ap.add_argument("--once", action="store_true", help="sample once, print, exit")
    args = ap.parse_args()

    if not Path(XRT_SMI).exists():
        sys.exit(f"xrt-smi not found at {XRT_SMI} -- this machine has no XDNA1 driver installed")

    with tempfile.TemporaryDirectory(prefix="hwinfo_npu_bridge_") as tmp:
        tmp_dir = Path(tmp)
        name, total_cols = device_info(tmp_dir)
        group = args.group or name
        print(f"device: {name}  total_columns: {total_cols}  registry group: {group}")

        last_completions = None
        last_t = None
        while True:
            sample = sample_aie(tmp_dir)
            now = time.perf_counter()
            if last_completions is None or sample["completions"] < last_completions:
                completions_per_sec = 0.0  # first sample, or counter reset by a new context
            else:
                completions_per_sec = (sample["completions"] - last_completions) / (now - last_t)
            last_completions, last_t = sample["completions"], now

            rows = publish(group, total_cols, sample, completions_per_sec)
            print("  " + "  ".join(f"{n}={v:.1f}{u}" for n, u, v in rows))

            if args.once:
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()

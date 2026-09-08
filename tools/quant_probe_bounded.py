"""Run one experimental probe with limits, containing native compiler failures."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--rss-gib", type=float, default=8)
    parser.add_argument("probe_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command_args = args.probe_args
    if command_args[:1] == ["--"]:
        command_args = command_args[1:]
    if args.seconds <= 0 or args.rss_gib <= 0 or "--out" not in command_args:
        parser.error("positive limits and probe --out are required")
    sidecar = Path(command_args[command_args.index("--out") + 1] + ".probe.json")
    if sidecar.exists():
        parser.error("probe result exists; choose a new output")
    print("PROBE_LIMITS", json.dumps({"seconds": args.seconds, "rss_gib": args.rss_gib}), flush=True)
    # Called from an activated conda env; child inherits the same PATH and DLL setup.
    child = subprocess.Popen(["python", "tools/quant_probe.py", *command_args])
    process = psutil.Process(child.pid)
    start, peak = time.monotonic(), 0
    stop = None
    try:
        while child.poll() is None:
            try:
                peak = max(peak, process.memory_info().rss)
            except psutil.NoSuchProcess:
                break
            elapsed = time.monotonic() - start
            if elapsed > args.seconds or peak > args.rss_gib * 1024**3:
                stop = {"elapsed_seconds": elapsed, "peak_rss_bytes": peak,
                        "reason": "rss_limit" if peak > args.rss_gib * 1024**3 else "time_limit"}
                child.kill()
                break
            time.sleep(0.5)
        rc = child.wait()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    if stop:
        if sidecar.exists():
            report = json.loads(sidecar.read_text(encoding="utf-8"))
            report["status_before_stop"] = report["status"]
            report["status"] = "resource_stop"
            report["resource_stop"] = stop
            sidecar.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("PROBE_RESOURCE_STOP", json.dumps(stop), flush=True)
        raise SystemExit(4)
    print("PROBE_PROCESS_EXIT", json.dumps({"returncode": rc, "peak_rss_bytes": peak,
                                            "elapsed_seconds": time.monotonic() - start}), flush=True)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()

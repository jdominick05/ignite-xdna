#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ask long-lived NPU users on this machine to give the device up for a measurement, and wait until they have.

A run that needs the NPU to itself writes ``%LOCALAPPDATA%\\ignite-xdna\\npu-quiet.<pid>.request``, JSON
``{"pid": <Windows pid of the requester>, "since": <epoch s>, "reason": "<what is measuring>"}``, and removes it on
exit. One file per requester, so nested or parallel runs never clobber each other's. A cooperating long-lived user
(ignite-scribe's always-on OCR) polls for any ``npu-quiet*.request`` whose pid is alive, closes its hardware context
within about 2 s and opens nothing until none remains; a file whose pid is dead is stale and ignored. The requester
then waits for ``xrt-smi`` to report no hardware context, up to a deadline, instead of failing at its first look.

    python tools/npu_quiet.py hold --reason <name> --pid <pid>     # write the request, print its path
    python tools/npu_quiet.py release --pid <pid>                  # remove it if it is that pid's
    python tools/npu_quiet.py wait --seconds 10                    # exit 0 once xrt-smi shows no context, else 1
    python tools/npu_quiet.py list                                 # every request file and whether its pid is alive
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SMI = Path("C:/Windows/System32/AMD/xrt-smi.exe")
IDLE = "No hardware contexts running"
# A cooperating holder releases within ~2 s of seeing a request it polls for once a second; waiting 5 s covers it.
MIN_WAIT_S = 5.0


def folder() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "ignite-xdna"


def path_for(pid: int) -> Path:
    return folder() / f"npu-quiet.{pid}.request"


def hold(reason: str, pid: int | None = None) -> Path:
    """Write this requester's file atomically and return its path."""
    pid = os.getpid() if pid is None else pid
    d = folder()
    d.mkdir(parents=True, exist_ok=True)
    p = path_for(pid)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"pid": pid, "since": int(time.time()), "reason": reason}), encoding="utf-8")
    tmp.replace(p)
    return p


def release(pid: int | None = None) -> bool:
    """Remove the file only if it names this pid. True if one was removed."""
    pid = os.getpid() if pid is None else pid
    p = path_for(pid)
    try:
        if json.loads(p.read_text(encoding="utf-8")).get("pid") != pid:
            return False
        p.unlink()
        return True
    except (FileNotFoundError, ValueError):
        return False


def requests():
    """(path, record, pid alive) for every request file."""
    import psutil
    out = []
    for p in sorted(folder().glob("npu-quiet*.request")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            rec = {}
        try:
            alive = bool(rec.get("pid")) and psutil.pid_exists(int(rec["pid"]))
        except (OverflowError, ValueError, TypeError):
            alive = False                      # no such pid can exist, so the file is stale
        out.append((p, rec, alive))
    return out


def report() -> str:
    r = subprocess.run([str(SMI), "examine", "-r", "aie-partitions"], capture_output=True, text=True, timeout=30)
    if r.returncode:
        raise RuntimeError("xrt-smi failed: " + r.stderr)
    return r.stdout


def wait_idle(seconds: float, report_fn=report, poll_s: float = 0.5) -> str:
    """Poll the device until it reports no hardware context or ``max(seconds, MIN_WAIT_S)`` passes; return the last
    report (the caller decides what a busy one means)."""
    deadline = time.monotonic() + max(seconds, MIN_WAIT_S)
    while True:
        text = report_fn()
        if IDLE in text or time.monotonic() >= deadline:
            return text
        time.sleep(poll_s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("hold", "release", "wait", "list"))
    ap.add_argument("--reason", default="measurement")
    ap.add_argument("--pid", type=int)
    ap.add_argument("--seconds", type=float, default=MIN_WAIT_S, help="wait: the deadline (at least MIN_WAIT_S)")
    a = ap.parse_args()
    if a.mode == "hold":
        print(hold(a.reason, a.pid))
    elif a.mode == "release":
        print("released" if release(a.pid) else "nothing to release")
    elif a.mode == "wait":
        idle = IDLE in wait_idle(a.seconds)
        print("idle" if idle else "busy")
        return 0 if idle else 1
    else:
        for p, rec, alive in requests():
            print(p.name, json.dumps(rec), "alive" if alive else "stale")
    return 0


if __name__ == "__main__":
    sys.exit(main())

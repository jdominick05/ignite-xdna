"""The NPU's own device-wide power mode, lowered only where it pays and always put back.

``xrt-smi configure --pmode`` sets the clock of every AIE core on the NPU: 1.80 GHz in ``default``, 1.03 in
``balanced`` and 0.80 in ``powersaver``. Measured on Desktop 2 (Ryzen 7 8700G, Phoenix) on 2026-09-16 with YOLOv8n and
``efficiency`` host threads, ``results/aie/energy_npu_pmode_yolov8n_phoenix_20260916T1931Z.log``:

- at 30 fps, ``powersaver`` spent 131.8 mJ per frame against 153.1 in ``default`` (14 % less), with G2G 18.12 against
  9.52 ms: the dispatch took 2.19 times as long;
- flat out it saved nothing (110.5 against 110.0 mJ) and the frame rate halved;
- the NPU's ``balanced`` mode was not separated from ``default`` on energy.

So the mode pays only while frames are paced below what the NPU could do, and only if the slower dispatch still fits
the frame period. It is device-wide: every other NPU application on the machine slows with it. ``NpuPowerGovernor``
lowers it under all of these conditions and no others:

1. the host power mode is ``efficiency`` and ``IGNITE_XDNA_NPU_POWER`` is not ``off``;
2. frames are paced (a frame period is known);
3. the frame's measured CPU time plus 2.19 times its dispatch fits in 80 % of the period;
4. the device reads ``default`` (a mode someone else chose is left alone);
5. no other process holds a hardware context on the NPU.

After lowering it keeps checking: if G2G's 95th percentile over the next 100 frames exceeds 90 % of the period, or
another process opens a hardware context, it puts ``default`` back for the rest of the session. It puts it back on
every exit path it can see (``restore``, ``atexit``), and before switching it writes a lease file naming this process;
``recover_stale_lease`` restores the mode a dead process left lowered, and runs whenever a governor is created.

Reading and setting go through ``xrt-smi``: pyxrt reports the mode but cannot set it, and does not list other
processes' hardware contexts.
"""
from __future__ import annotations

import atexit
import ctypes
import json
import os
import platform
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

XRT_SMI = os.environ.get("IGNITE_XDNA_XRT_SMI", r"C:\Windows\System32\AMD\xrt-smi.exe")
ENV_NPU_POWER = "IGNITE_XDNA_NPU_POWER"          # "auto" (default) or "off"
ENV_WATCH_S = "IGNITE_XDNA_NPU_POWER_WATCH_S"     # seconds between checks for other NPU applications
ENV_LEASE = "IGNITE_XDNA_NPU_POWER_LEASE"         # lease file path (tests)
LOWER_MODE = "powersaver"
DISPATCH_SLOWDOWN = 2.19   # powersaver dispatch / default dispatch, YOLOv8n, 2026-09-16
BUDGET_FRACTION = 0.8      # predicted G2G must fit in this share of the frame period
REVERT_FRACTION = 0.9      # a measured P95 above this share of the period puts the mode back
CHECK_FRAMES = 100
DEFAULT_WATCH_S = 5.0

Runner = Callable[[Sequence[str]], Tuple[int, str]]
_MODE_RE = re.compile(r"Power Mode\s*:\s*(\w+)", re.I)
# A hardware-context row of `xrt-smi examine -r aie-partitions`: |PID |Ctx ID |Submissions |...
_CONTEXT_ROW_RE = re.compile(r"^\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|", re.M)


def run_xrt_smi(args: Sequence[str], timeout_s: float = 15.0) -> Tuple[int, str]:
    """(return code, stdout + stderr) of one xrt-smi call; 127 if it cannot be run."""
    try:
        p = subprocess.run([XRT_SMI, *args], capture_output=True, text=True, timeout=timeout_s)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, f"{type(exc).__name__}: {exc}"


def read_mode(runner: Runner = run_xrt_smi) -> Optional[str]:
    """The device's power mode in lower case ("default", "powersaver", ...), or None if xrt-smi cannot say."""
    rc, out = runner(["examine", "-r", "platform"])
    m = _MODE_RE.search(out) if rc == 0 else None
    return m.group(1).lower() if m else None


def set_mode(mode: str, runner: Runner = run_xrt_smi) -> bool:
    """Set the device's power mode and read it back; True only if the read-back matches."""
    runner(["configure", "--pmode", mode])  # `turbo` reports an error and applies anyway, so trust the read-back
    return read_mode(runner) == mode


def context_pids(runner: Runner = run_xrt_smi) -> Optional[List[int]]:
    """PIDs owning hardware contexts on the NPU (one entry per context), or None if xrt-smi cannot say."""
    rc, out = runner(["examine", "-r", "aie-partitions"])
    if rc != 0 or "AIE Partitions" not in out:
        return None
    if re.search(r"No hardware contexts running", out, re.I):
        return []
    return [int(m.group(1)) for m in _CONTEXT_ROW_RE.finditer(out)]


# ---- lease: which process lowered the mode, so a crash does not leave the NPU slowed ----

def lease_path() -> Path:
    if os.environ.get(ENV_LEASE):
        return Path(os.environ[ENV_LEASE])
    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    return Path(base) / "ignite-xdna" / "npu_power_lease.json"


def process_start_time(pid: int) -> Optional[int]:
    """A running process's creation time (Windows FILETIME ticks), or None if it is not running."""
    if platform.system() != "Windows":
        try:
            os.kill(pid, 0)
            return 0
        except OSError:
            return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    PROCESS_QUERY_LIMITED_INFORMATION, STILL_ACTIVE = 0x1000, 259
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        return None
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != STILL_ACTIVE:
            return None
        creation, exit_, kernel, user = (ctypes.c_ulonglong() for _ in range(4))
        if not kernel32.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_), ctypes.byref(kernel),
                                        ctypes.byref(user)):
            return 0
        return int(creation.value)
    finally:
        kernel32.CloseHandle(handle)


def _write_lease(previous: str, lowered: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": os.getpid(), "start_time": process_start_time(os.getpid()),
                                "previous": previous, "set": lowered,
                                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}), encoding="utf-8")


def _read_lease(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def recover_stale_lease(runner: Runner = run_xrt_smi, path: Optional[Path] = None) -> Optional[str]:
    """Put back a power mode that a process which is no longer running left lowered. Returns what it did, or None.

    A lease whose process still runs is left alone. The mode is restored only if the device still reads the mode
    the lease set, so a mode someone chose since is not overwritten.
    """
    path = path or lease_path()
    lease = _read_lease(path)
    if lease is None:
        if path.exists():
            path.unlink(missing_ok=True)
            return "removed an unreadable NPU power lease"
        return None
    pid, started = int(lease.get("pid", -1)), lease.get("start_time")
    if pid != os.getpid():
        now = process_start_time(pid)
        if now is not None and (started in (None, 0) or now == started):
            return None
    current = read_mode(runner)
    lowered, previous = lease.get("set"), lease.get("previous", "default")
    path.unlink(missing_ok=True)
    if current is not None and current == lowered:
        ok = set_mode(previous, runner)
        return (f"restored the NPU power mode to {previous} after process {pid} left it in {lowered}"
                if ok else f"could not restore the NPU power mode to {previous} (process {pid} left it in {lowered})")
    return f"dropped process {pid}'s NPU power lease; the device reads {current}, not {lowered}"


class NpuPowerGovernor:
    """Lowers the NPU's power mode to ``powersaver`` when the conditions in this module's docstring hold."""

    def __init__(self, host_mode: str, period_s: float, enabled: Optional[bool] = None, runner: Runner = run_xrt_smi,
                 lease: Optional[Path] = None, watch_interval_s: Optional[float] = None,
                 log: Callable[[str], None] = print):
        self.host_mode = host_mode
        self.period_s = float(period_s or 0.0)
        self.enabled = (os.environ.get(ENV_NPU_POWER, "auto").strip().lower() != "off") if enabled is None else enabled
        self.runner = runner
        self.lease = lease or lease_path()
        self.watch_interval_s = float(watch_interval_s if watch_interval_s is not None
                                      else os.environ.get(ENV_WATCH_S, DEFAULT_WATCH_S))
        self.log = log
        self.state = "not evaluated"   # "kept", "lowered", "restored"
        self.reason = ""
        self.predicted_ms: Optional[float] = None
        self.previous_mode: Optional[str] = None
        self.lowered_at_frame: Optional[int] = None
        self.check_p95_ms: Optional[float] = None
        self._g2g: List[float] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._watcher: Optional[threading.Thread] = None
        recovered = recover_stale_lease(runner, self.lease)
        self.recovered = recovered
        if recovered:
            log(f"[npu power] {recovered}")

    # -- decision --
    def consider(self, cpu_ms: float, dispatch_ms: float, frame: int = 0) -> bool:
        """Decide once, from a frame's measured CPU time and NPU dispatch time. True if the mode was lowered."""
        if self.state != "not evaluated":
            return self.state == "lowered"
        period_ms = self.period_s * 1000.0
        self.predicted_ms = cpu_ms + dispatch_ms * DISPATCH_SLOWDOWN
        if not self.enabled:
            return self._keep(f"{ENV_NPU_POWER}=off")
        if self.host_mode != "efficiency":
            return self._keep(f"the {self.host_mode} power mode leaves the NPU's mode alone")
        if period_ms <= 0:
            return self._keep("frames are not paced, and lowering the NPU saves nothing flat out")
        if self.predicted_ms > BUDGET_FRACTION * period_ms:
            return self._keep(f"predicted G2G {self.predicted_ms:.1f} ms in {LOWER_MODE} does not fit "
                              f"{BUDGET_FRACTION:.0%} of the {period_ms:.1f} ms frame period")
        mode = read_mode(self.runner)
        if mode is None:
            return self._keep("xrt-smi did not report the power mode")
        if mode != "default":
            return self._keep(f"the device is already in {mode}, which is left as it is")
        others = self._other_pids()
        if others is None:
            return self._keep("xrt-smi did not report the NPU's hardware contexts")
        if others:
            return self._keep(f"another process holds the NPU (PID {', '.join(map(str, sorted(set(others))))})")
        _write_lease(mode, LOWER_MODE, self.lease)
        if not set_mode(LOWER_MODE, self.runner):
            self.lease.unlink(missing_ok=True)
            set_mode(mode, self.runner)
            return self._keep(f"xrt-smi could not set {LOWER_MODE}")
        self.previous_mode, self.state, self.lowered_at_frame = mode, "lowered", frame
        self.reason = (f"predicted G2G {self.predicted_ms:.1f} ms fits {BUDGET_FRACTION:.0%} of the "
                       f"{period_ms:.1f} ms frame period")
        atexit.register(self.restore, "process exit")
        if self.watch_interval_s > 0:
            self._watcher = threading.Thread(target=self._watch, name="npu-power-watch", daemon=True)
            self._watcher.start()
        return True

    def _keep(self, reason: str) -> bool:
        self.state, self.reason = "kept", reason
        return False

    def _other_pids(self) -> Optional[List[int]]:
        pids = context_pids(self.runner)
        return None if pids is None else [p for p in pids if p != os.getpid()]

    # -- after lowering --
    def observe(self, g2g_ms: float) -> None:
        """Feed each frame's G2G; after CHECK_FRAMES frames, a P95 above REVERT_FRACTION of the period restores."""
        if self.state != "lowered" or self.check_p95_ms is not None:
            return
        self._g2g.append(float(g2g_ms))
        if len(self._g2g) >= CHECK_FRAMES:
            ordered = sorted(self._g2g)
            self.check_p95_ms = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
            limit = REVERT_FRACTION * self.period_s * 1000.0
            if self.check_p95_ms > limit:
                self.restore(f"G2G P95 {self.check_p95_ms:.1f} ms over {CHECK_FRAMES} frames exceeded "
                             f"{REVERT_FRACTION:.0%} of the frame period ({limit:.1f} ms)")

    def _watch(self) -> None:
        while not self._stop.wait(self.watch_interval_s):
            others = self._other_pids()
            if others:
                self.restore(f"another process opened the NPU (PID {', '.join(map(str, sorted(set(others))))})")
                return

    def restore(self, reason: str = "session end") -> None:
        """Put the mode back if this governor lowered it and the device still reads the lowered mode. Idempotent."""
        with self._lock:
            if self.state != "lowered":
                return
            self._stop.set()
            current = read_mode(self.runner)
            if current == LOWER_MODE:
                ok = set_mode(self.previous_mode or "default", self.runner)
                self.reason = f"{reason}; restored {self.previous_mode}" + ("" if ok else " FAILED")
            else:
                self.reason = f"{reason}; the device read {current}, so it was left as it is"
            self.lease.unlink(missing_ok=True)
            self.state = "restored"
        self.log(f"[npu power] {self.reason}")

    def describe(self) -> str:
        if self.state == "lowered":
            return f"{LOWER_MODE} from frame {self.lowered_at_frame}: {self.reason}"
        if self.state == "restored":
            return f"lowered to {LOWER_MODE} at frame {self.lowered_at_frame}, then {self.reason}"
        return f"default kept: {self.reason or 'not evaluated'}"

    def status(self) -> Dict:
        return {"state": self.state, "reason": self.reason, "lower_mode": LOWER_MODE, "previous_mode": self.previous_mode,
                "period_ms": self.period_s * 1000.0, "predicted_ms": self.predicted_ms,
                "lowered_at_frame": self.lowered_at_frame, "check_p95_ms": self.check_p95_ms,
                "watch_interval_s": self.watch_interval_s, "recovered_lease": self.recovered}

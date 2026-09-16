"""Energy per frame of whole application loops, as a delta against an idle baseline taken in the same sitting.

The NPU has no power domain of its own (see ``tools/power_probe.py``), so the only honest energy figure for an NPU
application is what the whole package spends on it: package power while the application's frame loop runs, minus
package power with nothing running, divided by how many frames the loop completed per second. That counts the NPU,
the host work between dispatches and the runtime around them together, which is the energy a user actually pays
per frame - and it is the same quantity for any two stacks, whatever each one puts on which device.

For each arm, in the order given:

1. **Idle baseline.** Package power sampled for ``--idle-s`` seconds with nothing launched.
2. **Run.** ``typeperf`` samples package power once a second for the arm's whole life; the arm is launched and every
   ``[run] frame N`` line it prints is stamped with the host clock as it arrives (Ignition's headless loop,
   ``tools/amd_vitisai_yolo.py``, ``tools/amd_vitisai_sesr.py`` and ``pipelines/yolov8n-pose/4_pose.py`` on an image
   each print one every 100 frames).
3. **Window.** From the ``--skip-lines``-th progress line to the last one. Everything outside it - interpreter start,
   model or session load, hardware-context creation, warm-up, shutdown - is excluded. Frames per second is
   ``(N_last - N_first) / (t_last - t_first)`` in wall time, so any work the loop does between frames is counted.
4. **Energy.** Mean package power over the samples stamped inside the window, minus the idle baseline, divided by
   frames per second: joules per frame. A bare wattage is never reported, only the delta and the baseline it is
   against.

Arms come from a JSON file, a list in run order::

    [{"label": "amd yolov8n #1", "cwd": "...", "env": {"VAR": "value"}, "cmd": ["python.exe", "script.py", "..."]}]

An arm may also carry ``"setup": [[argv], ...]``, commands run before its idle baseline with their output logged (a
device setting such as ``xrt-smi configure --pmode``, then a report that evidences it); a failing setup command stops
the sitting. The file may instead be ``{"arms": [...], "finally": [[argv], ...]}``, whose ``finally`` commands run
when the sitting ends, however it ends, to put such a setting back.

Interleave the stacks in that list (AMD, Ignition, AMD, Ignition) so drift lands on both. An idle baseline more than
``--idle-max-shift-w`` from the median of the sitting's earlier ones is flagged: measured, a load that showed no CPU
held idle about 5.5 W high for an hour and then vanished, which a disturbed-baseline check cannot see and which
makes the median-idle column wrong for every arm on the other side of the step (each arm's own idle stays valid). The log and JSON replace the
user profile directory with ``<home>``, and any ``--redact PATH=PLACEHOLDER`` path (a scratch or checkout directory)
with its placeholder, in every string including the arms' recorded environment.

    python tools/energy_sitting.py --arms arms.json --json out.json --log out.log --redact "D:/scratch=<scratch>"
"""
import argparse
import csv
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from power_probe import PKG  # noqa: E402

PROGRESS = re.compile(r"\[run\] frame (\d+)\b")
HOME = str(Path.home())
# Sampled beside package power so a watt delta can be read against how busy the CPU was while it was spent.
CPU = r"\Processor(_Total)\% Processor Time"
COUNTERS = [PKG, CPU]


# (path, placeholder) pairs from --redact, longest path first so a checkout inside another is replaced whole.
REDACTIONS: list = []


def _variants(path: str):
    return {path, path.replace("\\", "/"), path.replace("/", "\\"), path.replace("\\", "\\\\"),
            path.replace("/", "\\\\")}


def scrub(text: str) -> str:
    """Replace every --redact path, then the user profile, with placeholders, in any slash spelling."""
    for path, placeholder in REDACTIONS:
        for v in _variants(path):
            text = text.replace(v, placeholder)
    for home in _variants(HOME):
        text = text.replace(home, "<home>")
    return text


def scrub_tree(value):
    """``scrub`` every string inside a JSON-shaped value."""
    if isinstance(value, str):
        return scrub(value)
    if isinstance(value, list):
        return [scrub_tree(v) for v in value]
    if isinstance(value, dict):
        return {k: scrub_tree(v) for k, v in value.items()}
    return value


class Log:
    def __init__(self, path):
        self.path = Path(path) if path else None
        if self.path:
            self.path.write_text("", encoding="utf-8")

    def __call__(self, line: str = ""):
        line = scrub(line)
        print(line, flush=True)
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


def start_typeperf(csv_path: Path) -> subprocess.Popen:
    # No sample count: it runs until killed. -y answers the overwrite prompt that otherwise blocks silently.
    return subprocess.Popen(["typeperf", *COUNTERS, "-si", "1", "-o", str(csv_path), "-y"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop(proc: subprocess.Popen) -> None:
    # taskkill /T, not terminate(): a child that outlives its parent bleeds into the next window (power_probe.py).
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass


def read_samples(csv_path: Path):
    """(epoch seconds, package milliwatts, CPU percent) per typeperf row; a row cut short by the kill is skipped."""
    out = []
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in list(csv.reader(f))[1:]:
            if len(row) < 1 + len(COUNTERS):
                continue
            try:
                stamp = datetime.strptime(row[0], "%m/%d/%Y %H:%M:%S.%f").timestamp()
                out.append((stamp, float(row[1]), float(row[2])))
            except ValueError:
                continue
    return out


def idle_baseline_checked(seconds: int, tmp: Path, log: Log, max_cpu: float, max_stdev_w: float, tries: int = 3):
    """An idle baseline, retaken when something else was running during it.

    Measured: one baseline in an otherwise clean sitting read 46.5 W, stdev 4.7 W, CPU 19.4 % against 34.3-35.5 W,
    stdev <= 1.6 W and CPU 7.8-9.0 % for every other one, which turned a +21 W arm into +9.8 W. Retaking an idle
    reading touches no hardware, so it is safe to repeat; the arm is flagged if every try is disturbed.
    """
    for attempt in range(1, tries + 1):
        mw, sd, n, cpu = idle_baseline(seconds, tmp, log)
        if cpu <= max_cpu and sd / 1000.0 <= max_stdev_w:
            return mw, sd, n, cpu, False
        log(f"  idle      disturbed (CPU {cpu:.1f} % > {max_cpu} or stdev {sd / 1000:.3f} W > {max_stdev_w}); "
            f"{'retaking' if attempt < tries else 'giving up, arm flagged'}")
    return mw, sd, n, cpu, True


def idle_baseline(seconds: int, tmp: Path, log: Log):
    csv_path = tmp / f"idle_{int(time.time() * 1000)}.csv"
    proc = subprocess.run(["typeperf", *COUNTERS, "-si", "1", "-sc", str(seconds), "-o", str(csv_path), "-y"],
                          capture_output=True, text=True)
    samples = read_samples(csv_path) if csv_path.exists() else []
    csv_path.unlink(missing_ok=True)
    if not samples:
        raise SystemExit(f"idle typeperf produced nothing (rc={proc.returncode}): {proc.stdout} {proc.stderr}")
    mw = [s[1] for s in samples]
    cpu = statistics.fmean(s[2] for s in samples)
    log(f"  idle      {len(mw)} samples, mean {statistics.fmean(mw) / 1000:.3f} W, "
        f"stdev {statistics.pstdev(mw) / 1000:.3f} W, CPU {cpu:.1f} %")
    return statistics.fmean(mw), statistics.pstdev(mw), len(mw), cpu


def run_commands(cmds, log: Log, what: str, cwd=None, env=None, stop_on_error: bool = True):
    """Run each argv in ``cmds`` in order, logging the command and its non-blank output lines."""
    for cmd in cmds:
        log(f"  {what:<9} $ {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8",
                              errors="replace")
        for line in (proc.stdout + proc.stderr).splitlines():
            if line.strip() and set(line.strip()) != {"-"}:
                log(f"    {line.rstrip()}")
        if proc.returncode != 0:
            msg = f"{what} command exited {proc.returncode}: {' '.join(cmd)}"
            if stop_on_error:
                raise SystemExit(msg + "; stopping the sitting")
            log(f"  {msg}")


def run_arm(arm: dict, skip_lines: int, tmp: Path, log: Log):
    env = dict(os.environ)
    env.update(arm.get("env", {}))
    csv_path = tmp / f"arm_{int(time.time() * 1000)}.csv"
    sampler = start_typeperf(csv_path)
    time.sleep(1.5)  # the first typeperf row lands about a second after start
    log(f"  $ {' '.join(arm['cmd'])}")
    t_launch = time.time()
    child = subprocess.Popen(arm["cmd"], cwd=arm.get("cwd"), env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
    marks, tail = [], []
    for line in child.stdout:
        now = time.time()
        line = line.rstrip()
        m = PROGRESS.search(line)
        if m:
            marks.append((now, int(m.group(1))))
        elif (line.startswith(("[summary]", "[amd]", "[verify]", "[Ignition] power mode", "g2g ", "pre ", "infer ",
                               "Traceback", "RuntimeError", "Error")) or "Error" in line):
            tail.append(line)
    rc = child.wait()
    t_exit = time.time()
    stop(sampler)
    samples = read_samples(csv_path) if csv_path.exists() else []
    csv_path.unlink(missing_ok=True)
    for line in tail:
        log(f"    {line}")
    if rc != 0:
        raise SystemExit(f"arm '{arm['label']}' exited {rc}; stopping the sitting rather than retrying")
    if len(marks) < skip_lines + 2:
        raise SystemExit(f"arm '{arm['label']}' printed {len(marks)} progress lines; need at least {skip_lines + 2}")
    (t1, n1), (t2, n2) = marks[skip_lines], marks[-1]
    fps = (n2 - n1) / (t2 - t1)
    # A typeperf row stamped t holds the second that ENDS at t, so a row belongs to the window only if that whole
    # second does.
    inside = [(mw, cpu) for stamp, mw, cpu in samples if t1 + 1.0 <= stamp <= t2]
    if len(inside) < 10:
        raise SystemExit(f"arm '{arm['label']}': only {len(inside)} power samples inside a {t2 - t1:.1f} s window")
    mw = [s[0] for s in inside]
    return {"label": arm["label"], "returncode": rc, "launch_to_exit_s": t_exit - t_launch,
            "window_s": t2 - t1, "window_frames": n2 - n1, "window_fps": fps, "power_samples": len(inside),
            "package_mw_mean": statistics.fmean(mw), "package_mw_stdev": statistics.pstdev(mw),
            "cpu_percent_mean": statistics.fmean(s[1] for s in inside),
            "summary_lines": [scrub(x) for x in tail]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arms", required=True, help="JSON list of arms in run order")
    ap.add_argument("--idle-s", type=int, default=30, help="idle baseline before each arm, seconds")
    ap.add_argument("--skip-lines", type=int, default=2, help="progress lines to skip before the window opens")
    ap.add_argument("--settle-s", type=float, default=5.0, help="pause after each arm before the next idle baseline")
    ap.add_argument("--idle-max-cpu", type=float, default=12.0, help="retake an idle baseline above this CPU %%")
    ap.add_argument("--idle-max-stdev-w", type=float, default=2.5, help="retake an idle baseline noisier than this")
    ap.add_argument("--idle-max-shift-w", type=float, default=2.0,
                    help="flag an idle baseline this far from the median of the sitting's earlier ones")
    ap.add_argument("--json", required=True)
    ap.add_argument("--log", default=None)
    ap.add_argument("--redact", action="append", default=[], metavar="PATH=PLACEHOLDER",
                    help="replace a machine path with a placeholder in the log and JSON (repeatable); the user "
                         "profile is always replaced with <home>")
    args = ap.parse_args()
    for pair in args.redact:
        path, _, placeholder = pair.partition("=")
        REDACTIONS.append((path, placeholder))
    REDACTIONS.sort(key=lambda p: len(p[0]), reverse=True)

    spec = json.loads(Path(args.arms).read_text(encoding="utf-8"))
    arms, final_cmds = (spec["arms"], spec.get("finally", [])) if isinstance(spec, dict) else (spec, [])
    log = Log(args.log)
    tmp = Path(tempfile.mkdtemp(prefix="energy_sitting_"))
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    log(f"Energy sitting on {socket.gethostname()}, started {started}")
    log(f"Package power: typeperf '{PKG}', 1 s samples, milliwatts. Idle {args.idle_s} s before each arm; window "
        f"from progress line {args.skip_lines} to the last.")
    results = []
    try:
        for arm in arms:
            log(f"== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [{arm['label']}]")
            if arm.get("setup"):
                run_commands(arm["setup"], log, "setup")
            idle_mw, idle_sd, idle_n, idle_cpu, suspect = idle_baseline_checked(
                args.idle_s, tmp, log, args.idle_max_cpu, args.idle_max_stdev_w)
            earlier = [r["idle_mw_mean"] for r in results if not r["idle_suspect"]]
            shifted = len(earlier) >= 3 and abs(idle_mw - statistics.median(earlier)) > args.idle_max_shift_w * 1000
            if shifted:
                log(f"  idle      shifted {(idle_mw - statistics.median(earlier)) / 1000:+.3f} W from the median of "
                    f"the {len(earlier)} earlier baselines; read this arm against its own idle")
            r = run_arm(arm, args.skip_lines, tmp, log)
            r.update(idle_mw_mean=idle_mw, idle_mw_stdev=idle_sd, idle_samples=idle_n, idle_cpu_percent=idle_cpu,
                     idle_suspect=suspect, idle_shifted=shifted, env=arm.get("env", {}), setup=arm.get("setup", []))
            r["delta_w"] = (r["package_mw_mean"] - idle_mw) / 1000.0
            r["mj_per_frame"] = 1000.0 * r["delta_w"] / r["window_fps"]
            log(f"  window    {r['window_frames']} frames in {r['window_s']:.1f} s = {r['window_fps']:.2f} fps, "
                f"{r['power_samples']} power samples, CPU {r['cpu_percent_mean']:.1f} %")
            log(f"  energy    package {r['package_mw_mean'] / 1000:.3f} W - idle {idle_mw / 1000:.3f} W = "
                f"+{r['delta_w']:.3f} W -> {r['mj_per_frame']:.2f} mJ per frame")
            results.append(r)
            time.sleep(args.settle_s)
    finally:
        if final_cmds:
            run_commands(final_cmds, log, "finally", stop_on_error=False)

    # The sitting's median idle, from the undisturbed baselines: a second column that one bad baseline cannot move.
    clean = [r["idle_mw_mean"] for r in results if not r["idle_suspect"]] or [r["idle_mw_mean"] for r in results]
    median_idle = statistics.median(clean)
    if max(clean) - min(clean) > args.idle_max_shift_w * 1000:
        log(f"Idle baselines span {min(clean) / 1000:.3f}-{max(clean) / 1000:.3f} W, more than "
            f"{args.idle_max_shift_w} W: the median-idle column does not hold across that; read arms against their own "
            f"idle")
    for r in results:
        r["delta_w_vs_median_idle"] = (r["package_mw_mean"] - median_idle) / 1000.0
        r["mj_per_frame_vs_median_idle"] = 1000.0 * r["delta_w_vs_median_idle"] / r["window_fps"]
    log("")
    log(f"Median idle over {len(clean)} undisturbed baselines: {median_idle / 1000:.3f} W")
    log(f"{'arm':<40} {'fps':>8} {'CPU %':>6} {'+W own idle':>12} {'mJ/frame':>9} {'mJ vs median idle':>18}")
    for r in results:
        flag = ("  idle disturbed" if r["idle_suspect"] else "") + ("  idle shifted" if r["idle_shifted"] else "")
        log(f"{r['label']:<40} {r['window_fps']:>8.2f} {r['cpu_percent_mean']:>6.1f} {r['delta_w']:>12.3f} "
            f"{r['mj_per_frame']:>9.2f} {r['mj_per_frame_vs_median_idle']:>18.2f}{flag}")
    Path(args.json).write_text(json.dumps(scrub_tree({
        "host": socket.gethostname(), "started_utc": started,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "counter": PKG,
        "idle_s": args.idle_s, "skip_lines": args.skip_lines, "median_idle_mw": median_idle, "finally": final_cmds,
        "arms": [dict(r, command=" ".join(a["cmd"])) for r, a in zip(results, arms)],
    }), indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

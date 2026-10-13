#!/usr/bin/env python3
"""Interleaved same-sitting A/B of an activation layout against AMD's stack.

    python tools/bench_layout_ab.py --note "plane-packed layout"

Three arms per round - AMD's Ryzen AI 1.7.1 Vitis AI EP, Ignition on the control container, and
Ignition on the container under test - run in that order, twice, so drift on this shared
machine shows up as a spread within a round rather than as a result. 50 warm-up and 500 timed
frames per run on bus.jpg. xrt-smi must report no hardware contexts before every run.

The Ignition arms both run through PYTHONPATH pointed at this worktree's src, so the runtime is
the one under test and the only difference between them is the container.

Glass-to-glass, not dispatch: this measures a whole frame, ingress through decode.

Anchor form (2026-10-13): --amd-config-file adds a second AMD arm built from a VitisAI EP config_file (e.g.
enable_cost_model_tiling, from tools/amd_xcompiler_config.py), --no-test runs the control container alone, and
--alternate reverses the arm order on every second round so each arm runs both early and late:

    python tools/bench_layout_ab.py --label yolov8s --control <c.ignite> --no-test \\
        --amd-config-file <cmt.json> --alternate --rounds 4 --prefix engine_anchor
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HOME = Path.home()
MAIN_REPO = HOME / "PycharmProjects" / "ignite-xdna"
IGNITION_REPO = HOME / "PycharmProjects" / "Ignition"
PY_IRON = HOME / "miniforge3" / "envs" / "mlir-aie-iron" / "python.exe"
PY_RESNET17 = HOME / "miniforge3" / "envs" / "resnet_env17" / "python.exe"
XRT_SMI = Path(r"C:\Windows\System32\AMD\xrt-smi.exe")
XCLBIN_4X4 = MAIN_REPO / "yolocutcachekey" / "4x4.xclbin"
BUS_JPG = IGNITION_REPO / "examples" / "assets" / "bus.jpg"
SCRATCH = Path(os.environ.get("TEMP", str(HOME / "AppData/Local/Temp"))) / "layout_ab"
SCRATCH.mkdir(parents=True, exist_ok=True)
(SCRATCH / "cache").mkdir(exist_ok=True)

_PH = sorted([(REPO_ROOT, "<ignite-xdna worktree>"), (MAIN_REPO, "<ignite-xdna main>"),
              (IGNITION_REPO, "<Ignition main>"), (HOME / "miniforge3" / "envs", "<conda envs>"),
              (SCRATCH, "<scratch>"), (HOME, "<home>")], key=lambda i: -len(str(i[0])))


def scrub(text):
    for root, label in _PH:
        for form in (str(root), str(root).replace("\\", "/")):
            text = text.replace(form, label)
    return text


def smi():
    r = subprocess.run([str(XRT_SMI), "examine", "-r", "aie-partitions"],
                       capture_output=True, text=True, timeout=20)
    if "No hardware contexts running" not in r.stdout:
        raise RuntimeError("NPU not idle: " + " | ".join(r.stdout.split()))
    return "No hardware contexts running on device"


def cpu(seconds=3.0):
    try:
        import psutil
        return f"CPU {psutil.cpu_percent(interval=seconds):.1f} % over {int(seconds)} s"
    except Exception:
        return "CPU load unavailable"


def run(name, cmd, env, f, keep):
    head = scrub(f"\n== {datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} [{name}]\n")
    print(head, end="", flush=True)
    f.write(head)
    p = subprocess.Popen([str(c) for c in cmd], env=env,
                         cwd=str(IGNITION_REPO if "live_ignition" in str(cmd[1]) else REPO_ROOT),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", bufsize=1)
    tail = []
    for line in iter(p.stdout.readline, ""):
        tail = (tail + [line])[-25:]
        if any(k in line for k in keep):
            line = scrub(line)
            print(line, end="", flush=True)
            f.write(line)
            f.flush()
    p.stdout.close()
    p.wait()
    f.write(f"== [{name}] exit {p.returncode}\n")
    if p.returncode != 0:
        # a child that fails before printing anything the filter keeps would otherwise vanish
        for line in tail:
            f.write(scrub("   | " + line))
        f.flush()
        raise RuntimeError(f"{name} exited {p.returncode}; its last lines are in the log")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", default=str(MAIN_REPO / "models" / "yolov8s_cut_xint8.onnx"))
    ap.add_argument("--control", default=str(REPO_ROOT / "build" / "ab_yolov8s_ctl.ignite"))
    ap.add_argument("--test", default=str(REPO_ROOT / "build" / "ab_yolov8s_band.ignite"))
    ap.add_argument("--label", default="yolov8s")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--frames", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--note", default=None)
    ap.add_argument("--prefix", default="layout_ab")
    ap.add_argument("--amd-config-file", default=None,
                    help="add a second AMD arm built from this VitisAI EP config_file (tools/amd_xcompiler_config.py),"
                         " cached under its own key")
    ap.add_argument("--no-test", action="store_true", help="run the control container only (an anchor sitting)")
    ap.add_argument("--alternate", action="store_true",
                    help="reverse the arm order on every second round, so each arm runs early and late")
    args = ap.parse_args()
    # The Ignition arms run with cwd set to the Ignition repo, so a relative container path
    # resolves there and the run exits 2 before printing anything the filter keeps.
    args.onnx = str(Path(args.onnx).resolve())
    args.control = str(Path(args.control).resolve())
    # --test takes one container or several, comma-separated; each becomes its own arm (test, test2, ...).
    args.tests = [] if args.no_test else [str(Path(t).resolve()) for t in args.test.split(",")]
    for path in [args.control] + args.tests:
        if not Path(path).exists():
            raise SystemExit(f"container not found: {path}")
    if args.amd_config_file:
        args.amd_config_file = str(Path(args.amd_config_file).resolve())
        if not Path(args.amd_config_file).exists():
            raise SystemExit(f"AMD config file not found: {args.amd_config_file}")

    iron = os.environ.copy()
    iron["PYTHONPATH"] = str(REPO_ROOT / "src")
    iron.pop("OMP_NUM_THREADS", None)
    amd = os.environ.copy()
    amd["RYZEN_AI_INSTALLATION_PATH"] = r"C:\Program Files\RyzenAI\1.7.1"

    keep_amd = ("mean", "Mean", "p50", "P50", "objects", "Error", "error")
    keep_ign = ("[summary]", "Error", "error", "Traceback")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    out = REPO_ROOT / "results" / "aie" / f"{args.prefix}_{args.label}_phoenix_{ts}.log"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("# Interleaved same-sitting A/B: an activation layout, its control, and AMD's stack\n")
        f.write("# Host DESKTOP-CBL5NUA (Desktop 2, Ryzen 7 8700G, XDNA1 Phoenix), balanced default mode\n")
        f.write(f"# {args.warmup} warm-up + {args.frames} timed frames per run, {args.rounds} rounds\n")
        if args.note:
            f.write(f"# Note: {args.note}\n")
        f.write(scrub(f"# control  {args.control}\n"))
        for i, path in enumerate(args.tests):
            f.write(scrub(f"# {'test' if i == 0 else f'test{i + 1}':8s} {path}\n"))
        if args.amd_config_file:
            f.write(scrub(f"# amd_cfg  {args.amd_config_file}\n"))
        f.write(f"# arm order {'reversed on even rounds' if args.alternate else 'fixed'}\n")
        f.write(f"# Start {datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}\n")
        f.write(f"[xrt-smi before start] {smi()}\n[cpu] {cpu(3.0)}\n")

        def amd_arm(r, tag, extra):
            f.write(f"\n[xrt-smi before {tag} #{r}] {smi()}\n")
            run(f"{tag} {args.label} #{r}",
                [PY_RESNET17, REPO_ROOT / "tools" / "amd_vitisai_yolo.py", "--model", args.onnx,
                 "--image", BUS_JPG, "--ignition-src", IGNITION_REPO / "src",
                 "--cache-dir", SCRATCH / "cache", "--cache-key", f"{args.label}_{tag}_ab",
                 "--xclbin", XCLBIN_4X4, "--warmup", args.warmup, "--iterations", args.frames,
                 "--json", SCRATCH / f"{tag}_{r}.json"] + extra, amd, f, keep_amd)

        def ign_arm(r, tag, container):
            f.write(f"[xrt-smi before ignition {tag} #{r}] {smi()}\n")
            run(f"ignition {tag} {args.label} #{r}",
                [PY_IRON, IGNITION_REPO / "live_ignition.py", "--model", container,
                 "--source", BUS_JPG, "--headless", "--warmup", args.warmup,
                 "--frames", args.frames, "--json", SCRATCH / f"ign_{tag}_{r}.json"],
                iron, f, keep_ign)

        arms = [lambda r: amd_arm(r, "amd", [])]
        if args.amd_config_file:
            arms.append(lambda r: amd_arm(r, "amd_cfg", ["--config-file", args.amd_config_file]))
        arms.append(lambda r: ign_arm(r, "control", args.control))
        for i, path in enumerate(args.tests):
            arms.append(lambda r, tag="test" if i == 0 else f"test{i + 1}", path=path: ign_arm(r, tag, path))
        for r in range(1, args.rounds + 1):
            for arm in (arms[::-1] if args.alternate and r % 2 == 0 else arms):
                arm(r)
                time.sleep(2)
        f.write(f"\n[xrt-smi after] {smi()}\n[cpu] {cpu(3.0)}\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
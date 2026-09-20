#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
tools/bench_same_sitting_opt.py

Orchestrates an interleaved same-sitting benchmark comparing Ignition (graph-engine bare-metal
containers on Phoenix XDNA1 NPU with dispatch and readback optimizations) against AMD's
Ryzen AI 1.7.1 stack (Vitis AI EP on ONNX Runtime) in balanced default power mode.

Runs 50 warm-up + 500 timed frames per run, interleaved:
  - SESR M7 (AMD #1, Ignition #1, AMD #2, Ignition #2)
  - YOLOv8s (AMD #1, Ignition #1, AMD #2, Ignition #2)
  - YOLOv8n (AMD #1, Ignition #1, AMD #2, Ignition #2)
  - YOLOv8n-pose (AMD #1, Ignition #1, AMD #2, Ignition #2)

Before every run, checks xrt-smi to ensure no hardware contexts are running on the NPU.
Outputs log directly to results/aie/latency_balanced_dispatch_opt_phoenix_<timestamp>.log.
"""

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
WORKTREE_SRC = REPO_ROOT / "src"
USER_HOME = Path.home()
MAIN_REPO = USER_HOME / "PycharmProjects" / "ignite-xdna"
IGNITION_REPO = USER_HOME / "PycharmProjects" / "Ignition"
PYTHON_IRON = USER_HOME / "miniforge3" / "envs" / "mlir-aie-iron" / "python.exe"
PYTHON_RESNET17 = USER_HOME / "miniforge3" / "envs" / "resnet_env17" / "python.exe"
XRT_SMI = Path(r"C:\Windows\System32\AMD\xrt-smi.exe")
XCLBIN_4X4 = MAIN_REPO / "yolocutcachekey" / "4x4.xclbin"
BUS_JPG = IGNITION_REPO / "examples" / "assets" / "bus.jpg"

SCRATCH_DIR = Path(os.environ.get("TEMP", str(USER_HOME / "AppData" / "Local" / "Temp"))) / "sitting_dispatch_opt"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = SCRATCH_DIR / "vitisai_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def check_xrt_smi() -> str:
    """Returns status string and ensures no hardware contexts are active."""
    try:
        res = subprocess.run([str(XRT_SMI), "examine", "-r", "aie-partitions"],
                             capture_output=True, text=True, timeout=10)
        out = res.stdout.strip()
        lines = [line.strip() for line in out.splitlines() if line.strip()]
        summary = " | ".join(lines)
        if "No hardware contexts running" not in out:
            raise RuntimeError(f"NPU not idle: {summary}")
        return summary
    except Exception as e:
        raise RuntimeError(f"Failed to check xrt-smi: {e}")


def get_cpu_load(seconds: float = 2.0) -> str:
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=seconds)
        return f"[cpu] CPU {cpu:.1f} % over {int(seconds)} s"
    except Exception:
        return "[cpu] CPU load check unavailable"


_PLACEHOLDERS = sorted(
    [(REPO_ROOT, "<ignite-xdna worktree>" if REPO_ROOT.resolve() != MAIN_REPO.resolve() else "<ignite-xdna main>"),
     (MAIN_REPO, "<ignite-xdna main>"), (IGNITION_REPO, "<Ignition main>"),
     (USER_HOME / "miniforge3" / "envs", "<conda envs>"), (SCRATCH_DIR, "<scratch>"),
     (Path(os.environ.get("TEMP", str(USER_HOME / "AppData" / "Local" / "Temp"))), "<temp>"), (USER_HOME, "<home>")],
    key=lambda item: -len(str(item[0])))


def scrub(text: str) -> str:
    """Machine paths in a logged line become placeholders, so a log can be committed as it is written."""
    for root, label in _PLACEHOLDERS:
        for form in (str(root), str(root).replace("\\", "/")):
            text = text.replace(form, label)
    return text


def run_command(name: str, cmd: list, env: dict, log_f) -> None:
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cmd_str = " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)
    header = scrub(f"\n== {now_iso} [{name}] $ {cmd_str}\n")
    print(header, end="", flush=True)
    log_f.write(header)
    log_f.flush()

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        env=env,
        cwd=str(IGNITION_REPO if "live_ignition.py" in str(cmd[0]) or "live_ignition.py" in str(cmd[1]) else REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",     # a child that writes its own codepage otherwise lands NULs in a UTF-8 log
        bufsize=1,
    )
    for line in iter(proc.stdout.readline, ""):
        line = scrub(line)
        print(line, end="", flush=True)
        log_f.write(line)
        log_f.flush()
    proc.stdout.close()
    proc.wait()
    dt = time.perf_counter() - t0

    exit_line = f"== {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} [{name}] exit {proc.returncode} after {dt:.1f} s\n"
    print(exit_line, flush=True)
    log_f.write(exit_line)
    log_f.flush()

    if proc.returncode != 0:
        raise RuntimeError(f"Step {name} failed with exit code {proc.returncode}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Interleaved same-sitting benchmark")
    parser.add_argument("--prefix", default="latency_balanced_ingress_opt", help="Log file prefix")
    parser.add_argument("--note", default=None,
                        help="one line written into the log header, saying what this sitting is testing")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    log_dir = REPO_ROOT / "results" / "aie"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{args.prefix}_phoenix_{timestamp}.log"

    print(f"Starting same-sitting benchmark -> {log_path}")

    # Environment templates
    iron_env = os.environ.copy()
    iron_env["PYTHONPATH"] = str(WORKTREE_SRC)
    iron_env.pop("OMP_NUM_THREADS", None)

    amd_env = os.environ.copy()
    amd_env["RYZEN_AI_INSTALLATION_PATH"] = r"C:\Program Files\RyzenAI\1.7.1"

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# Interleaved same-sitting benchmark: Ignition bare-metal vs AMD Ryzen AI 1.7.1 (Vitis AI EP)\n")
        f.write(f"# Host: DESKTOP-CBL5NUA (Desktop 2, Ryzen 7 8700G, XDNA1 Phoenix), balanced default mode\n")
        f.write(f"# Optimizations: direct C8 channel-blocked native decode, consolidated head sync spans, AVX2 vectorized ingress staging & Q11 spatial interpolation\n")
        if args.note:
            f.write(f"# Note: {args.note}\n")
        f.write(f"# Start: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n\n")

        smi = check_xrt_smi()
        f.write(f"[xrt-smi before start] {smi}\n")
        f.write(f"{get_cpu_load(3.0)}\n\n")

        # -------------------------------------------------------------
        # 1. SESR M7
        # -------------------------------------------------------------
        f.write("## Timed: SESR M7 (Interleaved AMD vs Ignition)\n")
        print("\n=== SESR M7 Interleaved Suite ===")

        # AMD SESR #1
        f.write(f"[xrt-smi before amd sesr_m7 #1] {check_xrt_smi()}\n")
        run_command(
            "amd sesr_m7 #1",
            [
                str(PYTHON_RESNET17), str(REPO_ROOT / "tools" / "amd_vitisai_sesr.py"),
                "--model", str(MAIN_REPO / "models" / "sesr_m7_xint8.onnx"),
                "--image", str(BUS_JPG),
                "--ignition-src", str(IGNITION_REPO / "src"),
                "--cache-dir", str(CACHE_DIR),
                "--cache-key", "sesr_m7_sitting",
                "--xclbin", str(XCLBIN_4X4),
                "--warmup", "50", "--iterations", "500",
                "--json", str(SCRATCH_DIR / "amd_sesr_m7_1.json"),
            ],
            amd_env, f
        )
        time.sleep(2)

        # Ignition SESR #1
        f.write(f"[xrt-smi before ignition sesr_m7 #1] {check_xrt_smi()}\n")
        run_command(
            "ignition sesr_m7 #1",
            [
                str(PYTHON_IRON), str(IGNITION_REPO / "live_ignition.py"),
                "--model", str(MAIN_REPO / "build" / "sesr_m7.ignite"),
                "--source", str(BUS_JPG),
                "--headless", "--warmup", "50", "--frames", "500",
                "--json", str(SCRATCH_DIR / "ign_sesr_m7_1.json"),
            ],
            iron_env, f
        )
        time.sleep(2)

        # AMD SESR #2
        f.write(f"[xrt-smi before amd sesr_m7 #2] {check_xrt_smi()}\n")
        run_command(
            "amd sesr_m7 #2",
            [
                str(PYTHON_RESNET17), str(REPO_ROOT / "tools" / "amd_vitisai_sesr.py"),
                "--model", str(MAIN_REPO / "models" / "sesr_m7_xint8.onnx"),
                "--image", str(BUS_JPG),
                "--ignition-src", str(IGNITION_REPO / "src"),
                "--cache-dir", str(CACHE_DIR),
                "--cache-key", "sesr_m7_sitting",
                "--xclbin", str(XCLBIN_4X4),
                "--warmup", "50", "--iterations", "500",
                "--json", str(SCRATCH_DIR / "amd_sesr_m7_2.json"),
            ],
            amd_env, f
        )
        time.sleep(2)

        # Ignition SESR #2
        f.write(f"[xrt-smi before ignition sesr_m7 #2] {check_xrt_smi()}\n")
        run_command(
            "ignition sesr_m7 #2",
            [
                str(PYTHON_IRON), str(IGNITION_REPO / "live_ignition.py"),
                "--model", str(MAIN_REPO / "build" / "sesr_m7.ignite"),
                "--source", str(BUS_JPG),
                "--headless", "--warmup", "50", "--frames", "500",
                "--json", str(SCRATCH_DIR / "ign_sesr_m7_2.json"),
            ],
            iron_env, f
        )
        time.sleep(2)

        # -------------------------------------------------------------
        # 2. YOLOv8s
        # -------------------------------------------------------------
        f.write("\n## Timed: YOLOv8s (Interleaved AMD vs Ignition)\n")
        print("\n=== YOLOv8s Interleaved Suite ===")

        # AMD YOLOv8s #1
        f.write(f"[xrt-smi before amd yolov8s #1] {check_xrt_smi()}\n")
        run_command(
            "amd yolov8s #1",
            [
                str(PYTHON_RESNET17), str(REPO_ROOT / "tools" / "amd_vitisai_yolo.py"),
                "--model", str(MAIN_REPO / "models" / "yolov8s_cut_xint8.onnx"),
                "--image", str(BUS_JPG),
                "--ignition-src", str(IGNITION_REPO / "src"),
                "--cache-dir", str(CACHE_DIR),
                "--cache-key", "yolov8s_sitting",
                "--xclbin", str(XCLBIN_4X4),
                "--warmup", "50", "--iterations", "500",
                "--json", str(SCRATCH_DIR / "amd_yolov8s_1.json"),
            ],
            amd_env, f
        )
        time.sleep(2)

        # Ignition YOLOv8s #1
        f.write(f"[xrt-smi before ignition yolov8s #1] {check_xrt_smi()}\n")
        run_command(
            "ignition yolov8s #1",
            [
                str(PYTHON_IRON), str(IGNITION_REPO / "live_ignition.py"),
                "--model", str(MAIN_REPO / "build" / "yolov8s.ignite"),
                "--source", str(BUS_JPG),
                "--headless", "--warmup", "50", "--frames", "500",
                "--json", str(SCRATCH_DIR / "ign_yolov8s_1.json"),
            ],
            iron_env, f
        )
        time.sleep(2)

        # AMD YOLOv8s #2
        f.write(f"[xrt-smi before amd yolov8s #2] {check_xrt_smi()}\n")
        run_command(
            "amd yolov8s #2",
            [
                str(PYTHON_RESNET17), str(REPO_ROOT / "tools" / "amd_vitisai_yolo.py"),
                "--model", str(MAIN_REPO / "models" / "yolov8s_cut_xint8.onnx"),
                "--image", str(BUS_JPG),
                "--ignition-src", str(IGNITION_REPO / "src"),
                "--cache-dir", str(CACHE_DIR),
                "--cache-key", "yolov8s_sitting",
                "--xclbin", str(XCLBIN_4X4),
                "--warmup", "50", "--iterations", "500",
                "--json", str(SCRATCH_DIR / "amd_yolov8s_2.json"),
            ],
            amd_env, f
        )
        time.sleep(2)

        # Ignition YOLOv8s #2
        f.write(f"[xrt-smi before ignition yolov8s #2] {check_xrt_smi()}\n")
        run_command(
            "ignition yolov8s #2",
            [
                str(PYTHON_IRON), str(IGNITION_REPO / "live_ignition.py"),
                "--model", str(MAIN_REPO / "build" / "yolov8s.ignite"),
                "--source", str(BUS_JPG),
                "--headless", "--warmup", "50", "--frames", "500",
                "--json", str(SCRATCH_DIR / "ign_yolov8s_2.json"),
            ],
            iron_env, f
        )
        time.sleep(2)

        # -------------------------------------------------------------
        # 3. YOLOv8n
        # -------------------------------------------------------------
        f.write("\n## Timed: YOLOv8n (Interleaved AMD vs Ignition)\n")
        print("\n=== YOLOv8n Interleaved Suite ===")

        # AMD YOLOv8n #1
        f.write(f"[xrt-smi before amd yolov8n #1] {check_xrt_smi()}\n")
        run_command(
            "amd yolov8n #1",
            [
                str(PYTHON_RESNET17), str(REPO_ROOT / "tools" / "amd_vitisai_yolo.py"),
                "--model", str(MAIN_REPO / "models" / "yolov8n_cut_xint8.onnx"),
                "--image", str(BUS_JPG),
                "--ignition-src", str(IGNITION_REPO / "src"),
                "--cache-dir", str(CACHE_DIR),
                "--cache-key", "yolov8n_energy",
                "--xclbin", str(XCLBIN_4X4),
                "--warmup", "50", "--iterations", "500",
                "--json", str(SCRATCH_DIR / "amd_yolov8n_1.json"),
            ],
            amd_env, f
        )
        time.sleep(2)

        # Ignition YOLOv8n #1
        f.write(f"[xrt-smi before ignition yolov8n #1] {check_xrt_smi()}\n")
        run_command(
            "ignition yolov8n #1",
            [
                str(PYTHON_IRON), str(IGNITION_REPO / "live_ignition.py"),
                "--model", str(MAIN_REPO / "build" / "yolov8n_full.ignite"),
                "--source", str(BUS_JPG),
                "--headless", "--warmup", "50", "--frames", "500",
                "--json", str(SCRATCH_DIR / "ign_yolov8n_1.json"),
            ],
            iron_env, f
        )
        time.sleep(2)

        # AMD YOLOv8n #2
        f.write(f"[xrt-smi before amd yolov8n #2] {check_xrt_smi()}\n")
        run_command(
            "amd yolov8n #2",
            [
                str(PYTHON_RESNET17), str(REPO_ROOT / "tools" / "amd_vitisai_yolo.py"),
                "--model", str(MAIN_REPO / "models" / "yolov8n_cut_xint8.onnx"),
                "--image", str(BUS_JPG),
                "--ignition-src", str(IGNITION_REPO / "src"),
                "--cache-dir", str(CACHE_DIR),
                "--cache-key", "yolov8n_energy",
                "--xclbin", str(XCLBIN_4X4),
                "--warmup", "50", "--iterations", "500",
                "--json", str(SCRATCH_DIR / "amd_yolov8n_2.json"),
            ],
            amd_env, f
        )
        time.sleep(2)

        # Ignition YOLOv8n #2
        f.write(f"[xrt-smi before ignition yolov8n #2] {check_xrt_smi()}\n")
        run_command(
            "ignition yolov8n #2",
            [
                str(PYTHON_IRON), str(IGNITION_REPO / "live_ignition.py"),
                "--model", str(MAIN_REPO / "build" / "yolov8n_full.ignite"),
                "--source", str(BUS_JPG),
                "--headless", "--warmup", "50", "--frames", "500",
                "--json", str(SCRATCH_DIR / "ign_yolov8n_2.json"),
            ],
            iron_env, f
        )
        time.sleep(2)

        # -------------------------------------------------------------
        # 4. YOLOv8n-pose
        # -------------------------------------------------------------
        f.write("\n## Timed: YOLOv8n-pose (Interleaved AMD vs Ignition)\n")
        print("\n=== YOLOv8n-pose Interleaved Suite ===")

        # AMD YOLOv8n-pose #1
        f.write(f"[xrt-smi before amd pose #1] {check_xrt_smi()}\n")
        run_command(
            "amd pose #1",
            [
                str(PYTHON_RESNET17), str(MAIN_REPO / "pipelines" / "yolov8n-pose" / "4_pose.py"),
                "--model", str(MAIN_REPO / "models" / "yolov8n-pose_cut_xint8.onnx"),
                "--ep", "npu", "--source", str(MAIN_REPO / "assets" / "bus.jpg"),
                "--warmup", "50", "--runs", "500", "--log", "2",
                "--out", str(SCRATCH_DIR / "amd_pose_1.jpg"),
            ],
            amd_env, f
        )
        time.sleep(2)

        # Ignition YOLOv8n-pose #1
        f.write(f"[xrt-smi before ignition pose #1] {check_xrt_smi()}\n")
        run_command(
            "ignition pose #1",
            [
                str(PYTHON_IRON), str(IGNITION_REPO / "live_ignition.py"),
                "--model", str(MAIN_REPO / "build" / "yolov8n_pose.ignite"),
                "--source", str(MAIN_REPO / "assets" / "bus.jpg"),
                "--headless", "--warmup", "50", "--frames", "500",
                "--json", str(SCRATCH_DIR / "ign_pose_1.json"),
            ],
            iron_env, f
        )
        time.sleep(2)

        # AMD YOLOv8n-pose #2
        f.write(f"[xrt-smi before amd pose #2] {check_xrt_smi()}\n")
        run_command(
            "amd pose #2",
            [
                str(PYTHON_RESNET17), str(MAIN_REPO / "pipelines" / "yolov8n-pose" / "4_pose.py"),
                "--model", str(MAIN_REPO / "models" / "yolov8n-pose_cut_xint8.onnx"),
                "--ep", "npu", "--source", str(MAIN_REPO / "assets" / "bus.jpg"),
                "--warmup", "50", "--runs", "500", "--log", "2",
                "--out", str(SCRATCH_DIR / "amd_pose_2.jpg"),
            ],
            amd_env, f
        )
        time.sleep(2)

        # Ignition YOLOv8n-pose #2
        f.write(f"[xrt-smi before ignition pose #2] {check_xrt_smi()}\n")
        run_command(
            "ignition pose #2",
            [
                str(PYTHON_IRON), str(IGNITION_REPO / "live_ignition.py"),
                "--model", str(MAIN_REPO / "build" / "yolov8n_pose.ignite"),
                "--source", str(MAIN_REPO / "assets" / "bus.jpg"),
                "--headless", "--warmup", "50", "--frames", "500",
                "--json", str(SCRATCH_DIR / "ign_pose_2.json"),
            ],
            iron_env, f
        )

        f.write(f"\n[xrt-smi after complete suite] {check_xrt_smi()}\n")
        f.write(f"# Finished at: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n")

    print(f"\nBenchmark sitting completed successfully! Full log: {log_path}")


if __name__ == "__main__":
    main()

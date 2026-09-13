"""Standalone DFL decode qualification for Phoenix XDNA1.

``--offline`` checks the NumPy float32 oracle and the fixed-point approximation.
``--compile`` builds the sixteen Peano core ELFs and the four-column transport.
``--hardware`` runs synthetic Q4 INT8 head tensors on Device 0, checks every
box and score, and applies the requested 300 us dispatch latency gate.

The latency measurement covers XRT start and device completion, including the
two output streams.  Host BO writes, readback, and reference calculation are
outside the bracket.  A passing compile is never reported as a silicon pass.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STAGE_PATH = ROOT / "kernels" / "aie2" / "dfl" / "dfl_stage.py"
STAGE_SPEC = importlib.util.spec_from_file_location("ignite_xdna_dfl_stage", STAGE_PATH)
if STAGE_SPEC is None or STAGE_SPEC.loader is None:
    raise ImportError(f"cannot load DFL stage module from {STAGE_PATH}")
STAGE = importlib.util.module_from_spec(STAGE_SPEC)
STAGE_SPEC.loader.exec_module(STAGE)

INPUT_BYTES = STAGE.INPUT_BYTES
BOXES_BYTES = STAGE.BOXES_BYTES
SCORES_BYTES = STAGE.SCORES_BYTES
ANCHORS = STAGE.ANCHORS
CORES = STAGE.CORES
ANCHORS_PER_CORE = STAGE.ANCHORS_PER_CORE
CHUNK = 15
# The polynomial is evaluated from Q4 INT8 logits.  One worst-case pixel is a
# conservative bound for the resulting DFL expectation after stride scaling.
BOX_TOLERANCE = 1.0
SCORE_TOLERANCE = 0.02
LATENCY_LIMIT_US = 300.0


def emit(event, **data):
    print(json.dumps(dict(event=event, **data), sort_keys=True), flush=True)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    return {
        name: digest(ROOT / "kernels/aie2/dfl" / name)
        for name in ("dfl_decode.cc", "dfl_stage.py")
    }


def artifact_directory(base):
    identity = json.dumps(sources(), sort_keys=True).encode()
    return Path(base).resolve() / hashlib.sha256(identity).hexdigest()[:16]


def fixture(seed, mode):
    rng = np.random.default_rng(seed)
    if mode == "random":
        return rng.integers(-32, 33, (ANCHORS, 144), dtype=np.int16).astype(np.int8)
    if mode == "extreme":
        return rng.integers(-128, 128, (ANCHORS, 144), dtype=np.int16).astype(np.int8)
    if mode == "uniform":
        out = np.zeros((ANCHORS, 144), dtype=np.int8)
        out[:, 64:] = -8
        return out
    if mode == "ramp":
        anchor, field = np.indices((ANCHORS, 144))
        return ((anchor * 7 + field * 13 + seed) % 65 - 32).astype(np.int8)
    raise ValueError(mode)


def _exp_poly_q15(residual):
    # Exact integer counterpart of dfl_decode.cc's Q15 degree-four polynomial.
    x = int(residual) * 128
    x = max(-32768, min(32767, x))

    def mul(a, b):
        product = int(a) * int(b)
        # AIE conv_even is round-to-nearest-even.  The values used here are
        # non-negative except for the residual terms; implement that rule.
        q, rem = divmod(abs(product), 1 << 15)
        if rem > (1 << 14) or (rem == (1 << 14) and (q & 1)):
            q += 1
        return (-q if product < 0 else q)

    r2 = mul(x, x)
    r3 = mul(r2, x)
    r4 = mul(r2, r2)
    term2 = mul(r2, 16384)
    term3 = mul(r3, 5461)
    term4 = mul(r4, 1365)
    return max(-32768, min(32767, 32767 + x + term2 + term3 + term4))


def _reciprocal_fixed(denominator, numerator_bits):
    numerator = 1 << numerator_bits
    return numerator // int(denominator)


def _exp_negative_q15(logits, max_logit):
    exp = np.zeros(16, dtype=np.int32)
    for lane, value in enumerate(logits):
        z = max(0, int(max_logit) - int(value))
        z_q8 = z * 16
        q = (z * 23 + 128) >> 8
        while q * 177 > z_q8:
            q -= 1
        while (q + 1) * 177 <= z_q8:
            q += 1
        residual = q * 177 - z_q8
        poly = _exp_poly_q15(residual)
        exp[lane] = 0 if q >= 15 else (poly >> q)
    return exp


def _fixed_expectation(logits):
    logits = np.asarray(logits, dtype=np.int8)
    maximum = int(logits.max())
    exp = _exp_negative_q15(logits, maximum)
    total = int(exp.sum())
    reciprocal = _reciprocal_fixed(max(1, total >> 4), 26)
    probabilities = (exp * reciprocal + (1 << 14)) >> 15
    return float(np.dot(probabilities.astype(np.float32), np.arange(16, dtype=np.float32)) / 32768.0)


def _fixed_sigmoid(logit):
    magnitude = abs(int(logit))
    negative_abs = np.array([-magnitude], dtype=np.int8)
    exp = int(_exp_negative_q15(negative_abs, 0)[0])
    inverse = _reciprocal_fixed(32768 + exp, 30)
    if int(logit) >= 0:
        return inverse / 32768.0
    return ((exp * inverse + (1 << 14)) >> 15) / 32768.0


def reference_float(head):
    dfl = head[:, :64].astype(np.float32).reshape(ANCHORS, 4, 16) / 16.0
    classes = head[:, 64:].astype(np.float32) / 16.0
    dfl = dfl - dfl.max(axis=2, keepdims=True)
    probabilities = np.exp(dfl, dtype=np.float32)
    probabilities /= probabilities.sum(axis=2, keepdims=True)
    distances = np.sum(probabilities * np.arange(16, dtype=np.float32), axis=2)
    scores = 1.0 / (1.0 + np.exp(-classes, dtype=np.float32))

    boxes = np.empty((ANCHORS, 4), dtype=np.float32)
    for start, end, grid, stride in ((0, 6400, 80, 8), (6400, 8000, 40, 16), (8000, 8400, 20, 32)):
        cells = np.arange(end - start, dtype=np.int32)
        x = cells % grid
        y = cells // grid
        cx = (x.astype(np.float32) + 0.5) * stride
        cy = (y.astype(np.float32) + 0.5) * stride
        boxes[start:end, 0] = cx - distances[start:end, 0] * stride
        boxes[start:end, 1] = cy - distances[start:end, 1] * stride
        boxes[start:end, 2] = cx + distances[start:end, 2] * stride
        boxes[start:end, 3] = cy + distances[start:end, 3] * stride
    return boxes, scores


def reference_fixed(head):
    boxes = np.empty((ANCHORS, 4), dtype=np.float32)
    scores = np.empty((ANCHORS, 80), dtype=np.float32)
    for anchor in range(ANCHORS):
        distances = [_fixed_expectation(head[anchor, dim * 16:(dim + 1) * 16]) for dim in range(4)]
        if anchor < 6400:
            cell, grid, stride = anchor, 80, 8
        elif anchor < 8000:
            cell, grid, stride = anchor - 6400, 40, 16
        else:
            cell, grid, stride = anchor - 8000, 20, 32
        cx = (cell % grid + 0.5) * stride
        cy = (cell // grid + 0.5) * stride
        boxes[anchor] = (cx - distances[0] * stride, cy - distances[1] * stride,
                         cx + distances[2] * stride, cy + distances[3] * stride)
        scores[anchor] = [_fixed_sigmoid(value) for value in head[anchor, 64:]]
    return boxes, scores


def offline():
    reports = []
    for seed, mode in ((17, "random"), (91, "extreme"), (123, "uniform"), (7, "ramp")):
        head = fixture(seed, mode)
        expected_boxes, expected_scores = reference_float(head)
        fixed_boxes, fixed_scores = reference_fixed(head)
        box_error = float(np.max(np.abs(fixed_boxes - expected_boxes)))
        score_error = float(np.max(np.abs(fixed_scores - expected_scores)))
        if box_error > BOX_TOLERANCE or score_error > SCORE_TOLERANCE:
            raise AssertionError(
                f"fixed-point tolerance failed for {mode}: boxes={box_error}, scores={score_error}"
            )
        reports.append({"mode": mode, "box_max_abs": box_error, "score_max_abs": score_error})
    emit("offline_pass", anchors=ANCHORS, cases=reports,
         input_layout="[8400,144] int8 Q4", output_layout="boxes[8400,4] + scores[8400,80] float32")


def preflight():
    smi = subprocess.run(
        [r"C:\Windows\System32\AMD\xrt-smi.exe", "examine", "-r", "aie-partitions"],
        capture_output=True, text=True, check=True, timeout=25,
    )
    print(smi.stdout, flush=True)
    if "[003d:00:01.1] : NPU Phoenix" not in smi.stdout:
        raise RuntimeError("Expected physical Phoenix Device 0 [003d:00:01.1]")
    if "No hardware contexts running" not in smi.stdout:
        raise RuntimeError("Device 0 is occupied; refusing a contended measurement")
    load = subprocess.run(
        ["powershell", "-NoProfile", "-File", str(ROOT / "tools/host_load.ps1")],
        capture_output=True, text=True, check=True, timeout=25,
    )
    print(load.stdout, flush=True)
    if "HOST_LOAD_VERDICT CLEAR" not in load.stdout:
        raise RuntimeError("Host-load preflight failed")


def compile_bundles(directory):
    from aie.utils import config

    STAGE.compile_design(directory)
    elfs = list((directory / "design.prj").rglob("*.elf"))
    if len(elfs) != CORES:
        raise AssertionError(f"Expected {CORES} core ELFs, got {len(elfs)}")
    peano = Path(config.peano_install_dir()) / "bin"
    obj = directory / "design.prj" / "dfl_decode.o"
    disassembly = subprocess.check_output(
        [str(peano / "llvm-objdump.exe"), "-d", str(obj)], text=True,
    )
    (directory / "design.prj" / "dfl_decode.disassembly.txt").write_text(
        disassembly, encoding="utf-8",
    )
    vector_instruction_count = len(re.findall(
        r"\b(?:vmul|vadd|vbcst|vsrs)\b", disassembly,
    ))
    if not re.search(r"\bvmul\b", disassembly) or not re.search(
        r"\bvadd(?:\.16)?\b", disassembly,
    ):
        raise AssertionError("Peano object contains no DFL vector multiply/add")
    manifest = {
        "sources": sources(),
        "compiler": subprocess.check_output(
            [str(Path(config.peano_install_dir()) / "bin" / "clang++.exe"), "--version"],
            text=True,
        ),
        "artifacts": {
            p.relative_to(directory).as_posix(): digest(p)
            for p in directory.rglob("*")
            if p.suffix in (".elf", ".o", ".bin", ".xclbin")
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    emit("compile_pass", cores=len(elfs), vector_instruction_count=vector_instruction_count,
         manifest=manifest)


def hardware(args, directory):
    from ignite_xdna.runtime.driver import XrtSiliconHarness

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["sources"] != sources():
        raise RuntimeError("Sources changed; recompile")
    for name, expected in manifest["artifacts"].items():
        if digest(directory / name) != expected:
            raise RuntimeError(f"Artifact mismatch: {name}")

    harness = XrtSiliconHarness(device_idx=0)
    harness.load_xclbin(str(directory / "design.xclbin"))
    instr, ninstr = harness.create_instruction_bo(str(directory / "insts.bin"))
    inp = harness.create_host_bo(INPUT_BYTES, 3)
    boxes_bo = harness.create_host_bo(BOXES_BYTES, 4)
    scores_bo = harness.create_host_bo(SCORES_BYTES, 5)
    run = harness.pyxrt.run(harness.kernel)
    for index, value in enumerate((3, instr, ninstr, inp, boxes_bo, scores_bo)):
        run.set_arg(index, value)

    to_device = harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    from_device = harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
    cases = [(17, "random"), (91, "extreme"), (123, "uniform"), (7, "ramp")]
    samples = []
    for index, (seed, mode) in enumerate(cases + [(1000 + i, "random") for i in range(args.warmup + args.iters)]):
        head = fixture(seed, mode)
        expected_boxes, expected_scores = reference_float(head)
        inp.write(head.tobytes(), 0)
        inp.sync(to_device)
        boxes_bo.write(np.full((ANCHORS, 4), np.nan, dtype=np.float32).tobytes(), 0)
        scores_bo.write(np.full((ANCHORS, 80), np.nan, dtype=np.float32).tobytes(), 0)
        boxes_bo.sync(to_device)
        scores_bo.sync(to_device)
        start = time.perf_counter_ns()
        run.start()
        state = run.wait(5000)
        elapsed = (time.perf_counter_ns() - start) / 1000.0
        if str(state) != "ert_cmd_state.ERT_CMD_STATE_COMPLETED":
            raise RuntimeError(f"hardware completion failed: {state}")
        boxes_bo.sync(from_device)
        scores_bo.sync(from_device)
        actual_boxes = np.frombuffer(boxes_bo.read(BOXES_BYTES, 0), dtype=np.float32).copy().reshape(ANCHORS, 4)
        actual_scores = np.frombuffer(scores_bo.read(SCORES_BYTES, 0), dtype=np.float32).copy().reshape(ANCHORS, 80)
        box_error = float(np.max(np.abs(actual_boxes - expected_boxes)))
        score_error = float(np.max(np.abs(actual_scores - expected_scores)))
        emit("comparison", index=index, seed=seed, mode=mode, measured=index >= len(cases),
             boxes_max_abs=box_error, scores_max_abs=score_error,
             boxes_bytes=BOXES_BYTES, scores_bytes=SCORES_BYTES,
             dispatch_wait_us=elapsed)
        if box_error > BOX_TOLERANCE or score_error > SCORE_TOLERANCE:
            raise AssertionError("DFL output exceeded INT8 quantization tolerance")
        if index >= len(cases):
            samples.append(elapsed)

    result = {
        "iterations": len(samples),
        "warmup": args.warmup,
        "mean_us": statistics.mean(samples),
        "median_us": statistics.median(samples),
        "p95_us": float(np.percentile(samples, 95)),
        "max_us": max(samples),
        "latency_limit_us": LATENCY_LIMIT_US,
        "latency_verdict": "PASS" if max(samples) < LATENCY_LIMIT_US else "FAIL",
    }
    emit("hardware_result", **result)
    if result["latency_verdict"] != "PASS":
        raise AssertionError(f"DFL dispatch latency exceeded {LATENCY_LIMIT_US} us: {result}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/dfl_decode")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if not (args.offline or args.compile or args.hardware):
        parser.error("select --offline, --compile, or --hardware")
    offline()
    directory = artifact_directory(args.build_dir)
    if args.hardware:
        preflight()
    if args.compile:
        compile_bundles(directory)
    if args.hardware:
        if not (directory / "manifest.json").exists():
            raise RuntimeError("hardware mode requires a prior --compile")
        hardware(args, directory)


if __name__ == "__main__":
    main()

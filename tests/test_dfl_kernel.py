"""Standalone DFL decode qualification for Phoenix XDNA1.

``--offline`` checks the NumPy float32 oracle and the fixed-point approximation.
``--compile`` builds the Peano core ELFs and the MemTile transport.
``--hardware`` runs synthetic Q4 INT8 head tensors on Device 0, checks every
decoded box and score of the instantiated cores against the float32 oracle and
the fixed-point reference, and applies the 300 us dispatch latency gate.

``--cols`` and ``--cores-per-col`` build a reduced design with the full design's
anchor mapping and host BO layout (kernels/aie2/dfl/dfl_stage.py). Anchors of
cores that are not instantiated must stay untouched in the host BOs. The latency
gate applies only to the full 4 x 4 shape; a reduced shape reports its timing
with verdict NOT_APPLICABLE.

Every dispatch runs in a fresh hardware context by default, because the design
is one-shot: each core and MemTile DMA program ends after one pass. Reusing a
context (``--shared-context``) timed out on the second dispatch, and afterwards
every new context failed with 0xc01e0009 until the NPU device was restarted.

The latency bracket is run.start() to run.wait() returning, so it includes XRT
submission, the input and both output DMA streams, and all core compute. Host
BO writes and syncs, readback and reference calculation are outside it. A
passing compile is never reported as a silicon pass.

Parity tolerance is one INT8 Q4 LSB: a box coordinate may differ from the
float32 oracle by at most stride/16 pixels (one Q4 step of DFL distance), and a
class score by at most 0.25/16 (one Q4 logit step through the steepest sigmoid
slope). Non-finite outputs and writes outside the instantiated anchors fail.

``--log PATH`` appends every printed line to a UTF-8 log, with the local profile
path replaced by C:\\Users\\<user>.
"""
from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys
import time
import traceback

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
STRIDES = np.repeat(np.array([8, 16, 32], dtype=np.float32), [6400, 1600, 400])
BOX_TOLERANCE_Q4_LSB = 1.0
SCORE_TOLERANCE = 0.25 / 16
LATENCY_LIMIT_US = 300.0
MEASURED_FIXTURES = 8
COMPLETED = "ert_cmd_state.ERT_CMD_STATE_COMPLETED"
XRT_SMI = r"C:\Windows\System32\AMD\xrt-smi.exe"

_LOG = None
_PROFILE = os.environ.get("USERPROFILE", "")


def out(text):
    text = str(text)
    # The checkout root first (a worktree path would otherwise leak local
    # tooling directory names), then the profile path in its escaped forms.
    root = str(ROOT)
    text = (text.replace(root.replace("\\", "\\\\"), "<worktree>")
                .replace(root, "<worktree>")
                .replace(root.replace("\\", "/"), "<worktree>"))
    if _PROFILE:
        text = (text.replace(_PROFILE.replace("\\", "\\\\"), "C:\\\\Users\\\\<user>")
                    .replace(_PROFILE, "C:\\Users\\<user>")
                    .replace(_PROFILE.replace("\\", "/"), "C:/Users/<user>"))
    print(text, flush=True)
    if _LOG is not None:
        _LOG.write(text + "\n")
        _LOG.flush()


def emit(event, **data):
    out(json.dumps(dict(event=event, **data), sort_keys=True))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    return {
        name: digest(ROOT / "kernels/aie2/dfl" / name)
        for name in ("dfl_decode.cc", "dfl_stage.py")
    }


def shape(cols, cores_per_col):
    return {"cols": cols, "cores_per_col": cores_per_col}


def artifact_directory(base, cols, cores_per_col):
    identity = json.dumps(dict(sources=sources(), shape=shape(cols, cores_per_col)),
                          sort_keys=True).encode()
    return Path(base).resolve() / hashlib.sha256(identity).hexdigest()[:16]


def active_mask(cols, cores_per_col):
    mask = np.zeros(ANCHORS, dtype=bool)
    for core in STAGE.active_cores(cols, cores_per_col):
        mask[core * ANCHORS_PER_CORE:(core + 1) * ANCHORS_PER_CORE] = True
    return mask


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


def reference_fixed(head, anchors=None):
    boxes = np.full((ANCHORS, 4), np.nan, dtype=np.float32)
    scores = np.full((ANCHORS, 80), np.nan, dtype=np.float32)
    for anchor in (range(ANCHORS) if anchors is None else anchors):
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


def box_q4_lsb(error_px, mask=None):
    strides = STRIDES if mask is None else STRIDES[mask]
    return error_px / strides[:, None] * 16.0


def offline():
    reports = []
    for seed, mode in ((17, "random"), (91, "extreme"), (123, "uniform"), (7, "ramp")):
        head = fixture(seed, mode)
        expected_boxes, expected_scores = reference_float(head)
        fixed_boxes, fixed_scores = reference_fixed(head)
        box_px = np.abs(fixed_boxes - expected_boxes)
        box_lsb = float(np.max(box_q4_lsb(box_px)))
        score_error = float(np.max(np.abs(fixed_scores - expected_scores)))
        if box_lsb > BOX_TOLERANCE_Q4_LSB or score_error > SCORE_TOLERANCE:
            raise AssertionError(
                f"fixed-point tolerance failed for {mode}: boxes={box_lsb} Q4 LSB, scores={score_error}"
            )
        reports.append({"mode": mode, "box_max_abs": float(box_px.max()),
                        "box_max_q4_lsb": box_lsb, "score_max_abs": score_error})
    emit("offline_pass", anchors=ANCHORS, cases=reports,
         box_tolerance_q4_lsb=BOX_TOLERANCE_Q4_LSB, score_tolerance=SCORE_TOLERANCE,
         input_layout="[8400,144] int8 Q4", output_layout="boxes[8400,4] + scores[8400,80] float32")


def smi_partitions():
    return subprocess.run([XRT_SMI, "examine", "-r", "aie-partitions"],
                          capture_output=True, text=True, check=True, timeout=25).stdout


def preflight():
    smi = smi_partitions()
    out(smi)
    if "[003d:00:01.1] : NPU Phoenix" not in smi:
        raise RuntimeError("Expected physical Phoenix Device 0 [003d:00:01.1]")
    if "No hardware contexts running" not in smi:
        raise RuntimeError("Device 0 is occupied; refusing a contended measurement")
    load = subprocess.run(
        ["powershell", "-NoProfile", "-File", str(ROOT / "tools/host_load.ps1")],
        capture_output=True, text=True, check=True, timeout=25,
    )
    out(load.stdout)
    if "HOST_LOAD_VERDICT CLEAR" not in load.stdout:
        raise RuntimeError("Host-load preflight failed")


def compile_bundles(directory, cols, cores_per_col):
    from aie.utils import config

    cores = len(STAGE.active_cores(cols, cores_per_col))
    STAGE.compile_design(directory, cols, cores_per_col)
    elfs = list((directory / "design.prj").rglob("*.elf"))
    if len(elfs) != cores:
        raise AssertionError(f"Expected {cores} core ELFs, got {len(elfs)}")
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
        "shape": shape(cols, cores_per_col),
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


def compare(actual_boxes, actual_scores, boxes, scores, fixed, mask):
    """Errors over the instantiated anchors, plus untouched-poison accounting."""
    active_boxes = actual_boxes[mask]
    active_scores = actual_scores[mask]
    nonfinite = int(np.count_nonzero(~np.isfinite(active_boxes))
                    + np.count_nonzero(~np.isfinite(active_scores)))
    stray = int(np.count_nonzero(~np.isnan(actual_boxes[~mask]))
                + np.count_nonzero(~np.isnan(actual_scores[~mask])))
    with np.errstate(invalid="ignore"):
        box_px = np.abs(active_boxes - boxes[mask])
        score = np.abs(active_scores - scores[mask])
    report = {
        "nonfinite_outputs": nonfinite,
        "stray_writes": stray,
        "boxes_max_abs_px": float(np.nanmax(box_px)) if box_px.size else 0.0,
        "boxes_max_q4_lsb": float(np.nanmax(box_q4_lsb(box_px, mask))) if box_px.size else 0.0,
        "scores_max_abs": float(np.nanmax(score)) if score.size else 0.0,
    }
    if fixed is not None:
        fixed_boxes, fixed_scores = fixed
        report["fixed_boxes_max_abs"] = float(np.nanmax(np.abs(active_boxes - fixed_boxes[mask])))
        report["fixed_scores_max_abs"] = float(np.nanmax(np.abs(active_scores - fixed_scores[mask])))
        report["fixed_mismatched_values"] = int(
            np.count_nonzero(active_boxes != fixed_boxes[mask])
            + np.count_nonzero(active_scores != fixed_scores[mask]))
    report["parity"] = ("PASS" if nonfinite == 0 and stray == 0
                        and report["boxes_max_q4_lsb"] <= BOX_TOLERANCE_Q4_LSB
                        and report["scores_max_abs"] <= SCORE_TOLERANCE else "FAIL")
    return report


def open_session(directory):
    """Register the xclbin, open a hardware context and prepare one ERT run."""
    from ignite_xdna.runtime.driver import XrtSiliconHarness

    harness = XrtSiliconHarness(device_idx=0)
    session = {"harness": harness}
    try:
        harness.load_xclbin(str(directory / "design.xclbin"))
        session["instr"], ninstr = harness.create_instruction_bo(str(directory / "insts.bin"))
        session["inp"] = harness.create_host_bo(INPUT_BYTES, 3)
        session["boxes"] = harness.create_host_bo(BOXES_BYTES, 4)
        session["scores"] = harness.create_host_bo(SCORES_BYTES, 5)
        session["run"] = harness.pyxrt.run(harness.kernel)
        for index, value in enumerate((3, session["instr"], ninstr, session["inp"],
                                       session["boxes"], session["scores"])):
            session["run"].set_arg(index, value)
        session["group_ids"] = {
            str(arg): f"0x{harness.kernel.group_id(arg):08x}" for arg in (1, 3, 4, 5)}
        session["to_device"] = harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        session["from_device"] = harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
    except BaseException:
        close_session(session)
        raise
    return session


def close_session(session):
    """Release the run, BOs, kernel and context in dependency order.

    MemTile and core lock values are not host state: the xclbin's lock
    initialisation is applied again when the next context is created.
    """
    harness = session.pop("harness")
    for key in ("run", "scores", "boxes", "inp", "instr"):
        session.pop(key, None)
    session.clear()
    harness.kernel = None
    harness.context = None
    harness.xclbin = None
    del harness
    gc.collect()


def hardware(args, directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["sources"] != sources():
        raise RuntimeError("Sources changed; recompile")
    if manifest.get("shape") != shape(args.cols, args.cores_per_col):
        raise RuntimeError("Build shape differs from --cols/--cores-per-col; recompile")
    for name, expected in manifest["artifacts"].items():
        if digest(directory / name) != expected:
            raise RuntimeError(f"Artifact mismatch: {name}")

    full = (args.cols, args.cores_per_col) == (STAGE.COLS, STAGE.CORES_PER_COL)
    mask = active_mask(args.cols, args.cores_per_col)
    emit("build_identity", shape=manifest["shape"], full_design=full,
         active_cores=len(STAGE.active_cores(args.cols, args.cores_per_col)),
         active_anchors=int(mask.sum()), sources=manifest["sources"],
         xclbin_sha256=digest(directory / "design.xclbin"),
         insts_sha256=digest(directory / "insts.bin"), test_sha256=digest(__file__))

    # All references are computed before the hardware context exists, so the
    # device is held only for dispatches and the timed loop runs on a quiet host.
    anchors = np.flatnonzero(mask)
    plan = [(seed, mode, "correctness") for seed, mode in
            ((17, "random"), (91, "extreme"), (123, "uniform"), (7, "ramp"))]
    plan += [(1000 + i % MEASURED_FIXTURES, "random", "warmup") for i in range(args.warmup)]
    plan += [(1000 + i % MEASURED_FIXTURES, "random", "measured") for i in range(args.iters)]
    references = {}
    for seed, mode, role in plan:
        if (seed, mode) not in references:
            head = fixture(seed, mode)
            boxes, scores = reference_float(head)
            fixed = reference_fixed(head, anchors) if role == "correctness" else None
            references[(seed, mode)] = (head, boxes, scores, fixed)
    emit("references_ready", dispatches=len(plan), distinct_fixtures=len(references),
         measured_fixture_pool=MEASURED_FIXTURES)

    preflight()
    context_mode = "shared" if args.shared_context else "fresh_per_dispatch"
    session = None
    samples = []
    worst = {"boxes_max_q4_lsb": 0.0, "boxes_max_abs_px": 0.0, "scores_max_abs": 0.0}
    fixed_worst = {"fixed_boxes_max_abs": 0.0, "fixed_scores_max_abs": 0.0,
                   "fixed_mismatched_values": 0}
    poison_boxes = np.full((ANCHORS, 4), np.nan, dtype=np.float32).tobytes()
    poison_scores = np.full((ANCHORS, 80), np.nan, dtype=np.float32).tobytes()
    try:
        for index, (seed, mode, role) in enumerate(plan):
            if session is None:
                session = open_session(directory)
                if index == 0:
                    emit("context_ready", context_mode=context_mode,
                         group_ids=session["group_ids"])
            head, boxes, scores, fixed = references[(seed, mode)]
            session["inp"].write(head.tobytes(), 0)
            session["inp"].sync(session["to_device"])
            session["boxes"].write(poison_boxes, 0)
            session["scores"].write(poison_scores, 0)
            session["boxes"].sync(session["to_device"])
            session["scores"].sync(session["to_device"])
            start = time.perf_counter_ns()
            session["run"].start()
            state = session["run"].wait(args.timeout_ms)
            elapsed = (time.perf_counter_ns() - start) / 1000.0
            if str(state) != COMPLETED:
                emit("dispatch_failure", index=index, seed=seed, mode=mode, role=role,
                     context_mode=context_mode, state=str(state), wait_us=elapsed)
                raise RuntimeError(f"hardware completion failed: {state}")
            session["boxes"].sync(session["from_device"])
            session["scores"].sync(session["from_device"])
            actual_boxes = np.frombuffer(session["boxes"].read(BOXES_BYTES, 0),
                                         dtype=np.float32).copy().reshape(ANCHORS, 4)
            actual_scores = np.frombuffer(session["scores"].read(SCORES_BYTES, 0),
                                          dtype=np.float32).copy().reshape(ANCHORS, 80)
            if not args.shared_context:
                close_session(session)
                session = None
            report = compare(actual_boxes, actual_scores, boxes, scores, fixed, mask)
            emit("comparison", index=index, seed=seed, mode=mode, role=role,
                 context_mode=context_mode, dispatch_wait_us=elapsed, **report)
            if report["parity"] != "PASS":
                witness = directory / f"parity_witness_{time.time_ns()}_{index}.npz"
                empty = np.empty(0, dtype=np.float32)
                np.savez_compressed(witness, head=head, mask=mask,
                                    actual_boxes=actual_boxes, actual_scores=actual_scores,
                                    boxes=boxes, scores=scores,
                                    fixed_boxes=fixed[0] if fixed is not None else empty,
                                    fixed_scores=fixed[1] if fixed is not None else empty)
                emit("parity_witness", index=index, file=str(witness), sha256=digest(witness))
                raise AssertionError(f"DFL output failed parity at dispatch {index}: {report}")
            for key in worst:
                worst[key] = max(worst[key], report[key])
            if fixed is not None:
                for key in fixed_worst:
                    fixed_worst[key] = max(fixed_worst[key], report[key])
            if role == "measured":
                samples.append(elapsed)
    finally:
        if session is not None:
            close_session(session)
            session = None
        post = smi_partitions()
        emit("postflight", no_hardware_contexts="No hardware contexts running" in post)

    result = {
        "shape": shape(args.cols, args.cores_per_col),
        "full_design": full,
        "context_mode": context_mode,
        "active_anchors": int(mask.sum()),
        "parity_verdict": "PASS",
        "dispatches_checked": len(plan),
        **worst,
        **fixed_worst,
        "box_tolerance_q4_lsb": BOX_TOLERANCE_Q4_LSB,
        "score_tolerance": SCORE_TOLERANCE,
        "iterations": len(samples),
        "warmup": args.warmup,
        "mean_us": statistics.mean(samples),
        "median_us": statistics.median(samples),
        "p95_us": float(np.percentile(samples, 95)),
        "min_us": min(samples),
        "max_us": max(samples),
        "stdev_us": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "latency_limit_us": LATENCY_LIMIT_US,
        "latency_gate": "max of measured dispatch waits below the limit, full 4x4 design only",
        "latency_verdict": ("PASS" if max(samples) < LATENCY_LIMIT_US else "FAIL")
                           if full else "NOT_APPLICABLE",
    }
    emit("hardware_result", **result)
    if result["latency_verdict"] == "FAIL":
        raise AssertionError(f"DFL dispatch latency exceeded {LATENCY_LIMIT_US} us: max {max(samples):.1f} us")


def header(args):
    out("Phoenix DFL decode silicon qualification")
    out(f"date_utc={datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}")
    out(f"host={socket.gethostname()}")
    out(f"command={subprocess.list2cmdline([sys.executable] + sys.argv)}")
    out(f"shape={args.cols}x{args.cores_per_col} "
        f"({len(STAGE.active_cores(args.cols, args.cores_per_col))} cores)")
    out(f"test_sha256={digest(__file__)}")
    for name, value in sources().items():
        out(f"source_{name}_sha256={value}")


def main():
    global _LOG
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--skip-offline", action="store_true",
                        help="skip the offline oracle check that otherwise runs first")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/dfl_decode")
    parser.add_argument("--cols", type=int, default=STAGE.COLS)
    parser.add_argument("--cores-per-col", type=int, default=STAGE.CORES_PER_COL)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timeout-ms", type=int, default=10000)
    parser.add_argument("--shared-context", action="store_true",
                        help="reuse one hardware context for every dispatch. The design is "
                             "one-shot (cores and MemTile programs end after one pass): on "
                             "Phoenix the second dispatch in one context timed out, and new "
                             "contexts then failed with 0xc01e0009 until the NPU device was "
                             "restarted")
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    if not (args.offline or args.compile or args.hardware):
        parser.error("select --offline, --compile, or --hardware")
    STAGE.check_shape(args.cols, args.cores_per_col)
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        _LOG = open(args.log, "a", encoding="utf-8", newline="\n")
    try:
        header(args)
        if args.offline or not args.skip_offline:
            offline()
        directory = artifact_directory(args.build_dir, args.cols, args.cores_per_col)
        if args.compile:
            compile_bundles(directory, args.cols, args.cores_per_col)
        if args.hardware:
            if not (directory / "manifest.json").exists():
                raise RuntimeError("hardware mode requires a prior --compile")
            hardware(args, directory)
    except BaseException:
        out(traceback.format_exc())
        raise
    finally:
        if _LOG is not None:
            _LOG.close()
            _LOG = None


if __name__ == "__main__":
    main()

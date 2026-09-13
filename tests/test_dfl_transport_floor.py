"""Transport floor of the Phoenix DFL stage with the decode math removed.

Builds the transport from kernels/aie2/dfl/dfl_stage.py unchanged, but links
kernels/aie2/dfl/dfl_transport_probe.cc in place of dfl_decode.cc. The probe core
writes anchor_index * 256 + input byte into every output float, so each box and
score is checked exactly (tolerance zero) while the on-array work is reduced to
moving the same bytes through the same DMA, lock and MemTile path.

The timed bracket and context handling are tests/test_dfl_kernel.py's, imported
from it: run.start() to run.wait() returning, and a fresh hardware context per
dispatch. The dispatch wait is therefore the floor any DFL kernel pays with this
I/O contract and this transport on this driver stack. It is compared with the
same 300 us limit as an informational verdict; it is not a decode result.

Run from the mlir-aie ironenv; --compile also needs the XRT SDK directory on PATH
for xclbinutil (see kernels/README.md):

    python tests/test_dfl_transport_floor.py --compile --cols 4 --cores-per-col 4
    python tests/test_dfl_transport_floor.py --hardware --cols 4 --cores-per-col 4 \\
        --log results/aie/dfl_decode_phoenix_transport_floor_4x4_<utc>.log
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DFL = ROOT / "kernels" / "aie2" / "dfl"
PROBE_SOURCE = DFL / "dfl_transport_probe.cc"
HARNESS = ROOT / "tests" / "test_dfl_kernel.py"
_SPEC = importlib.util.spec_from_file_location("ignite_xdna_test_dfl_kernel", HARNESS)
T = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(T)
STAGE = T.STAGE
MODE = "transport_probe"


def sources():
    return {"dfl_transport_probe.cc": T.digest(PROBE_SOURCE),
            "dfl_stage.py": T.digest(DFL / "dfl_stage.py")}


def artifact_directory(base, cols, cores_per_col):
    identity = json.dumps(dict(sources=sources(), shape=T.shape(cols, cores_per_col), mode=MODE),
                          sort_keys=True).encode()
    return Path(base).resolve() / hashlib.sha256(identity).hexdigest()[:16]


def compile_probe(directory, cols, cores_per_col):
    from aie.utils.compile.utils import compile_cxx_core_function, compile_mlir_module

    directory.mkdir(parents=True, exist_ok=True)
    work = directory / "design.prj"
    work.mkdir(exist_ok=True)
    # The transport MLIR links "dfl_decode.o"; the probe object takes that name.
    compile_cxx_core_function(str(PROBE_SOURCE), "aie2", str(work / "dfl_decode.o"),
                              compile_args=["-O3"])
    ir = STAGE.module(cols, cores_per_col)
    (directory / "dfl_stage.mlir").write_text(ir, encoding="utf-8")
    compile_mlir_module(ir, insts_path=directory / "insts.bin",
                        xclbin_path=directory / "design.xclbin", work_dir=work)
    cores = len(STAGE.active_cores(cols, cores_per_col))
    elfs = list(work.rglob("*.elf"))
    if len(elfs) != cores:
        raise AssertionError(f"Expected {cores} core ELFs, got {len(elfs)}")
    manifest = {
        "sources": sources(),
        "shape": T.shape(cols, cores_per_col),
        "mode": MODE,
        "artifacts": {
            p.relative_to(directory).as_posix(): T.digest(p)
            for p in directory.rglob("*")
            if p.suffix in (".elf", ".o", ".bin", ".xclbin")
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    T.emit("compile_pass", mode=MODE, cores=len(elfs), manifest=manifest)


def expected_outputs(head):
    base = (np.arange(T.ANCHORS, dtype=np.float32) * np.float32(256.0))[:, None]
    wire = base + head[:, :84].astype(np.float32)
    return wire[:, :4].copy(), wire[:, 4:84].copy()


def hardware(args, directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["sources"] != sources() or manifest.get("mode") != MODE
            or manifest.get("shape") != T.shape(args.cols, args.cores_per_col)):
        raise RuntimeError("Probe sources, shape or mode changed; recompile")
    for name, expected in manifest["artifacts"].items():
        if T.digest(directory / name) != expected:
            raise RuntimeError(f"Artifact mismatch: {name}")

    mask = T.active_mask(args.cols, args.cores_per_col)
    full = (args.cols, args.cores_per_col) == (STAGE.COLS, STAGE.CORES_PER_COL)
    T.emit("build_identity", mode=MODE, shape=manifest["shape"], full_design=full,
           active_anchors=int(mask.sum()), sources=manifest["sources"],
           xclbin_sha256=T.digest(directory / "design.xclbin"),
           insts_sha256=T.digest(directory / "insts.bin"),
           test_sha256=T.digest(__file__), harness_sha256=T.digest(HARNESS))

    plan = [(seed, mode, "correctness") for seed, mode in
            ((17, "random"), (91, "extreme"), (123, "uniform"), (7, "ramp"))]
    plan += [(1000 + i % T.MEASURED_FIXTURES, "random", "warmup") for i in range(args.warmup)]
    plan += [(1000 + i % T.MEASURED_FIXTURES, "random", "measured") for i in range(args.iters)]
    references = {}
    for seed, mode, _ in plan:
        if (seed, mode) not in references:
            head = T.fixture(seed, mode)
            references[(seed, mode)] = (head, *expected_outputs(head))

    T.preflight()
    poison_boxes = np.full((T.ANCHORS, 4), np.nan, dtype=np.float32).tobytes()
    poison_scores = np.full((T.ANCHORS, 80), np.nan, dtype=np.float32).tobytes()
    samples = []
    session = None
    try:
        for index, (seed, mode, role) in enumerate(plan):
            session = T.open_session(directory)
            head, boxes, scores = references[(seed, mode)]
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
            if str(state) != T.COMPLETED:
                T.emit("dispatch_failure", index=index, seed=seed, mode=mode, role=role,
                       state=str(state), wait_us=elapsed)
                raise RuntimeError(f"hardware completion failed: {state}")
            session["boxes"].sync(session["from_device"])
            session["scores"].sync(session["from_device"])
            actual_boxes = np.frombuffer(session["boxes"].read(T.BOXES_BYTES, 0),
                                         dtype=np.float32).copy().reshape(T.ANCHORS, 4)
            actual_scores = np.frombuffer(session["scores"].read(T.SCORES_BYTES, 0),
                                          dtype=np.float32).copy().reshape(T.ANCHORS, 80)
            T.close_session(session)
            session = None
            box_mismatches = int(np.count_nonzero(actual_boxes[mask] != boxes[mask]))
            score_mismatches = int(np.count_nonzero(actual_scores[mask] != scores[mask]))
            stray = int(np.count_nonzero(~np.isnan(actual_boxes[~mask]))
                        + np.count_nonzero(~np.isnan(actual_scores[~mask])))
            parity = "PASS" if box_mismatches == score_mismatches == stray == 0 else "FAIL"
            T.emit("comparison", index=index, seed=seed, mode=mode, role=role,
                   context_mode="fresh_per_dispatch", dispatch_wait_us=elapsed,
                   box_mismatches=box_mismatches, score_mismatches=score_mismatches,
                   stray_writes=stray, parity=parity)
            if parity != "PASS":
                raise AssertionError(f"transport probe output mismatch at dispatch {index}")
            if role == "measured":
                samples.append(elapsed)
    finally:
        if session is not None:
            T.close_session(session)
        post = T.smi_partitions()
        T.emit("postflight", no_hardware_contexts="No hardware contexts running" in post)

    median = statistics.median(samples)
    T.emit("transport_floor_result", mode=MODE, shape=T.shape(args.cols, args.cores_per_col),
           full_design=full, active_anchors=int(mask.sum()), parity_verdict="PASS",
           dispatches_checked=len(plan), context_mode="fresh_per_dispatch",
           iterations=len(samples), warmup=args.warmup,
           mean_us=statistics.mean(samples), median_us=median,
           p95_us=float(np.percentile(samples, 95)), min_us=min(samples), max_us=max(samples),
           stdev_us=statistics.stdev(samples) if len(samples) > 1 else 0.0,
           latency_limit_us=T.LATENCY_LIMIT_US,
           floor_verdict="BELOW_LIMIT" if median < T.LATENCY_LIMIT_US else "AT_OR_ABOVE_LIMIT",
           scope="no decode math on the cores; dispatch wait of the unchanged transport")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/dfl_transport_probe")
    parser.add_argument("--cols", type=int, default=STAGE.COLS)
    parser.add_argument("--cores-per-col", type=int, default=STAGE.CORES_PER_COL)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timeout-ms", type=int, default=10000)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    if not (args.compile or args.hardware):
        parser.error("select --compile and/or --hardware")
    STAGE.check_shape(args.cols, args.cores_per_col)
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        T._LOG = open(args.log, "a", encoding="utf-8", newline="\n")
    try:
        T.out("Phoenix DFL transport floor probe (decode math removed)")
        T.out(f"date_utc={datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}")
        T.out(f"host={socket.gethostname()}")
        T.out(f"command={subprocess.list2cmdline([sys.executable] + sys.argv)}")
        T.out(f"shape={args.cols}x{args.cores_per_col} "
              f"({len(STAGE.active_cores(args.cols, args.cores_per_col))} cores)")
        T.out(f"test_sha256={T.digest(__file__)}")
        T.out(f"harness_sha256={T.digest(HARNESS)}")
        for name, value in sources().items():
            T.out(f"source_{name}_sha256={value}")
        directory = artifact_directory(args.build_dir, args.cols, args.cores_per_col)
        if args.compile:
            compile_probe(directory, args.cols, args.cores_per_col)
        if args.hardware:
            if not (directory / "manifest.json").exists():
                raise RuntimeError("hardware mode requires a prior --compile")
            hardware(args, directory)
    except BaseException:
        T.out(traceback.format_exc())
        raise
    finally:
        if T._LOG is not None:
            T._LOG.close()
            T._LOG = None


if __name__ == "__main__":
    main()

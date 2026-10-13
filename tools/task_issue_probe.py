"""What does one shim task cost the instruction stream, measured directly?

Every Block 5 sizing priced an extra shim task at 0.36 us: 4 instruction ops at the 0.09 us per op that the
engine's per-layer fit (engine_layer_timing) attributes to ops. That is a fit coefficient, not a measurement.
This moves the same bytes through the engine's activation path - shim MM2S -> MemTile split -> the four cores of
each column, 6,400 B per core object, no compute - on all four columns, issued as tasks of k quads each (a quad
is one object for each of the column's four cores). The tasks are retired the way the engine's emitter retires
them: a completion token on every second task and on the channel's last, at most 4 tasks pending per channel,
the oldest token group awaited and freed before a fifth is configured. So the instruction stream per task is
the engine's (configure + patch + push, and an await every second task).

    python tools/task_issue_probe.py build                (in the IRON env: bash scripts/research-iron.sh ...)
    python tools/task_issue_probe.py sit --rounds 3 --iters 30 --json <out>

Grid: k in 1, 2, 4, 16, 64 quads per task x Q in 512, 1024 quads per column. Fit over all points:
    dispatch ms = a + b * MB + c * tasks           (tasks = 4 columns x Q / k)
c is the per-task cost; b the per-byte cost at this traffic. Each measurement runs in its own child process,
the grid rotates across rounds. Written on Desktop 2 (Phoenix), 2026-10-13; what has been run is in its logs.
"""
import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

A_BYTES = 6400
QUAD = 4 * A_BYTES
COLS = 4
KS = (1, 2, 4, 16, 64)
QS = (512, 1024)
QUEUE_DEPTH = 4
RETIRE_BATCH = 2
BUILD = ROOT / "build" / "task_issue_probe"


# How one task of k quads addresses DDR. Each entry gives (bytes of DDR one quad spans, sizes, strides) for a task of
# k quads; the outermost size is the queue-push repeat count (shim_dma_single_bd_task), the other three are the BD's
# addressing dimensions. Every shape moves exactly k x 25,600 B per task.
#   contig  the original grid: one contiguous run (the BD folds it), k quads by repeat.
#   t3      one quad as a strided 3-D pattern, 16 planes x 8 rows x 200 B at a 2,048 / 256 B pitch; k quads by
#           repeat. k=1 has no repeat at all.
#   e4      the engine's own shape (e.g. YOLOv8s's (4, 4, 8, 200) fills): 4 per-core slices on the repeat, each a
#           3-D 4 x 8 x 200 B pattern. A second quad would need a fifth dimension, so only k=1 exists.
FOOT = 32768
SHAPES = {
    "contig": lambda k: (QUAD, [k, 1, 64, 400], [QUAD, 0, 400, 1]),
    "t3": lambda k: (FOOT, [k, 16, 8, 200], [FOOT, 2048, 256, 1]),
    "e4": lambda k: (FOOT, [4, 4, 8, 200], [8192, 2048, 256, 1]),
}


def name_of(k, q, shape="contig"):
    return f"k{k}_q{q}" if shape == "contig" else f"{shape}_k{k}_q{q}"


def build_one(k: int, q: int, shape: str = "contig") -> Path:
    import aie.iron as iron
    from aie.dialects.aiex import dma_await_task, dma_free_task, dma_start_task, shim_dma_single_bd_task
    from aie.iron import ObjectFifo, Program, Runtime, Worker
    from aie.iron.device import NPU1, Tile
    from aie.utils.compile.utils import compile_mlir_module

    assert q % k == 0 and k <= 64
    assert shape != "e4" or k == 1, "the engine's 4-D quad leaves no dimension for a second quad"
    foot, sizes, strides = SHAPES[shape](k)
    out_dir = BUILD / name_of(k, q, shape)
    work = out_dir / "design.prj"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    iron.set_current_device(NPU1())
    obj_ty = np.ndarray[(A_BYTES,), np.dtype[np.uint8]]
    quad_ty = np.ndarray[(QUAD,), np.dtype[np.uint8]]

    def core_fn(i):
        i.acquire(1)
        i.release(1)

    workers, shim_ends, names = [], [], []
    for c in range(COLS):
        name = f"a{c}"
        f = ObjectFifo(quad_ty, name=name, depth=2)
        parts = f.cons().split([j * A_BYTES for j in range(4)], tile=Tile(c, 1), depths=[2] * 4,
                               obj_types=[obj_ty] * 4, names=[f"{name}_{j}" for j in range(4)])
        for j, part in enumerate(parts):
            workers.append(Worker(core_fn, [part.cons()], tile=Tile(c, 2 + j)))
        shim_ends.append(f.prod(tile=Tile(c, 0)))
        names.append(name)
    col_bytes = q * foot             # DDR one column's quads span; each moves QUAD of it
    total = COLS * col_bytes
    moved = COLS * q * QUAD
    ntasks = q // k

    def sequence(inp, outp, *handles):
        inp = getattr(inp, "op", inp)
        queues = {c: [] for c in range(COLS)}   # [task, token]

        def retire(c):
            qu = queues[c]
            i = next(i for i, (_, tok) in enumerate(qu) if tok)
            dma_await_task(qu[i][0])
            for t, _ in qu[:i + 1]:
                dma_free_task(t)
            del qu[:i + 1]

        # Columns interleaved task by task, as the engine's emitter interleaves items.
        for n in range(ntasks):
            for c in range(COLS):
                if len(queues[c]) >= QUEUE_DEPTH:
                    retire(c)
                token = (n + 1) % RETIRE_BATCH == 0 or n == ntasks - 1
                t = shim_dma_single_bd_task(names[c], inp, offset=c * col_bytes + n * k * foot,
                                            sizes=list(sizes), strides=list(strides), issue_token=token)
                dma_start_task(t)
                queues[c].append([t, token])
        for c in range(COLS):
            while queues[c]:
                retire(c)

    in_t = np.ndarray[(total,), np.dtype[np.uint8]]
    out_t = np.ndarray[(64,), np.dtype[np.uint8]]
    rt = Runtime(sequence, [in_t, out_t] + shim_ends)
    module = Program(iron.get_current_device(), rt, workers=workers).resolve_program()
    (out_dir / "design.mlir").write_text(str(module), encoding="utf-8")
    compile_mlir_module(module, insts_path=out_dir / "insts.bin", xclbin_path=out_dir / "probe.xclbin",
                        work_dir=work, device=iron.get_current_device())
    insts = (out_dir / "insts.bin").stat().st_size
    (out_dir / "meta.json").write_text(json.dumps({"k": k, "q": q, "shape": shape, "sizes": list(sizes),
                                                   "strides": list(strides), "tasks": COLS * ntasks,
                                                   "total_bytes": total, "moved_bytes": moved,
                                                   "insts_bytes": insts}), encoding="utf-8")
    return out_dir


def time_one(k: int, q: int, iters: int, warmup: int, timeout_ms: int, shape: str = "contig") -> dict:
    from ignite_xdna.runtime.driver import XrtSiliconHarness
    d = BUILD / name_of(k, q, shape)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    total = meta["total_bytes"]
    moved = meta.get("moved_bytes", total)   # builds before the shape axis moved their whole buffer
    with XrtSiliconHarness(0) as h:
        pyxrt = h.pyxrt
        h.load_xclbin(str(d / "probe.xclbin"), "MLIR_AIE")
        bo_instr, n_instr = h.create_instruction_bo_from_bytes((d / "insts.bin").read_bytes())
        bo_in = h.create_host_bo(total, 3)
        bo_out = h.create_host_bo(64, 4)
        bo_in.write(np.arange(total, dtype=np.uint32).astype(np.uint8), 0)
        bo_in.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        run = pyxrt.run(h.kernel)
        for i, arg in enumerate((3, bo_instr, n_instr, bo_in, bo_out)):
            run.set_arg(i, arg)
        done = pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED
        times = []
        for i in range(warmup + iters):
            t0 = time.perf_counter()
            run.start()
            state = run.wait(timeout_ms)
            dt = (time.perf_counter() - t0) * 1e3
            if state != done:
                raise RuntimeError(f"k{k} q{q}: dispatch {i} ended in state {state}")
            if i >= warmup:
                times.append(dt)
        del run, bo_in, bo_out, bo_instr
    return {"k": k, "q": q, "shape": shape, "tasks": meta["tasks"], "mb": moved / 1e6,
            "insts_bytes": meta["insts_bytes"],
            "mean_ms": statistics.fmean(times), "median_ms": statistics.median(times), "min_ms": min(times),
            "times_ms": times}


def fit(rows):
    """Least squares: mean_ms = a + b * mb + c * tasks."""
    X = np.array([[1.0, r["mb"], r["tasks"]] for r in rows])
    y = np.array([r["mean_ms"] for r in rows])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return coef, r2, y - pred


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    points_help = ("comma list of shape:k:q points (shapes: " + ", ".join(SHAPES) + "); replaces the --ks x --qs "
                   "grid of contiguous tasks, and skips the plane fit")
    b = sub.add_parser("build")
    b.add_argument("--ks", default=",".join(map(str, KS)))
    b.add_argument("--qs", default=",".join(map(str, QS)))
    b.add_argument("--points", default=None, help=points_help)
    t = sub.add_parser("time", help="one measurement; used by 'sit' as a child process")
    t.add_argument("k", type=int)
    t.add_argument("q", type=int)
    t.add_argument("--shape", default="contig", choices=sorted(SHAPES))
    t.add_argument("--iters", type=int, default=30)
    t.add_argument("--warmup", type=int, default=3)
    t.add_argument("--timeout-ms", type=int, default=5000)
    s = sub.add_parser("sit")
    s.add_argument("--ks", default=",".join(map(str, KS)))
    s.add_argument("--qs", default=",".join(map(str, QS)))
    s.add_argument("--points", default=None, help=points_help)
    s.add_argument("--rounds", type=int, default=3)
    s.add_argument("--iters", type=int, default=30)
    s.add_argument("--json", required=True)
    args = ap.parse_args()

    if args.cmd == "time":
        print(json.dumps(time_one(args.k, args.q, args.iters, args.warmup, args.timeout_ms, args.shape)))
        return
    if args.points:
        combos = []
        for p in args.points.split(","):
            shape, k, q = p.split(":")
            if shape not in SHAPES:
                raise SystemExit(f"unknown shape {shape!r}")
            combos.append((shape, int(k), int(q)))
    else:
        combos = [("contig", k, q) for q in map(int, args.qs.split(",")) for k in map(int, args.ks.split(","))]
    if args.cmd == "build":
        for shape, k, q in combos:
            t0 = time.perf_counter()
            d = build_one(k, q, shape)
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            print(f"[build] {d.name}: {meta['tasks']} tasks, insts {meta['insts_bytes']:,} B "
                  f"in {time.perf_counter() - t0:.1f} s", flush=True)
        return

    results = []
    for rnd in range(args.rounds):
        order = combos[rnd % len(combos):] + combos[:rnd % len(combos)]
        if rnd % 2:
            order = order[::-1]
        for shape, k, q in order:
            proc = subprocess.run([sys.executable, __file__, "time", str(k), str(q), "--shape", shape,
                                   "--iters", str(args.iters)], capture_output=True, text=True, timeout=300)
            if proc.returncode != 0:
                print(proc.stdout[-2000:], proc.stderr[-2000:], sep="\n")
                raise SystemExit(f"[sit] {shape} k{k} q{q} failed in round {rnd}; stopping - do not retry in a loop")
            rec = json.loads(proc.stdout.strip().splitlines()[-1])
            rec["round"] = rnd
            results.append(rec)
            print(f"[sit] round {rnd} {shape:6s} k{k:<3} q{q:<5} {rec['tasks']:5d} tasks {rec['mb']:6.1f} MB: "
                  f"mean {rec['mean_ms']:.3f} ms, min {rec['min_ms']:.3f}", flush=True)
    if args.points:
        by = {}
        for r in results:
            by.setdefault((r["shape"], r["k"], r["q"]), []).append(r["mean_ms"])
        for (shape, k, q), v in by.items():
            print(f"[point] {shape:6s} k{k:<3} q{q:<5} mean of rounds {statistics.fmean(v):.3f} ms "
                  f"(round spread {(max(v) - min(v)) * 1e3:.0f} us)")
        Path(args.json).write_text(json.dumps({"results": results}, indent=1), encoding="utf-8")
        return
    coef, r2, resid = fit(results)
    a, b, c = coef
    print(f"\n[fit] dispatch ms = {a:.4f} + {b:.5f} * MB + {c * 1e3:.4f} us * tasks   (R2 {r2:.4f}, "
          f"{len(results)} points, max |residual| {np.abs(resid).max() * 1e3:.1f} us)")
    print(f"[fit] per task {c * 1e3:.3f} us; per-byte rate {1 / b:.2f} GB/s across the array")
    Path(args.json).write_text(json.dumps({"results": results, "fit": {"a_ms": a, "b_ms_per_mb": b,
                                                                       "c_ms_per_task": c, "r2": r2}},
                                          indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

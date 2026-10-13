"""Is the engine's activation transport capped per column (one shim-to-MemTile stream) or per core?

The per-layer fit of the graph engine's YOLOv8s puts its non-compute time at 0.235 us per 6,400 B activation
packet: about 27 GB/s across the array, 6.8 GB/s per column, close to one 32-bit stream at the 1.80 GHz core clock
(7.2 GB/s) and to the 7 GB/s the one-core MemTile hop probe measured. If one shim MM2S stream per column is the
cap, more activation streams per column is the lever; if the cap is per core, packet shape and bytes are. This
measures it with the engine's own object size and path, input only, no compute:

    c1     shim ch -> MemTile forward -> 1 core                      (column 0)
    c2     shim ch -> MemTile split   -> 2 cores                     (column 0)
    c4     shim ch -> MemTile split   -> 4 cores                     (column 0; the engine's arrangement)
    c4x2   2 shim ch -> 2 MemTile splits -> 2 + 2 cores              (column 0; the candidate lever)
    a16    c4 in each of the 4 columns
    a16x2  c4x2 in each of the 4 columns

Every core takes 6,400 B objects and releases them; each shim fifo moves its share in one
``shim_dma_single_bd_task`` with a completion token, awaited. Downstream buffering is two objects per stage,
so the token lands within a few objects of the last core release; the slope over several volumes removes that
and the fixed dispatch cost. Rate = 1 / slope.

    bash scripts/research-iron.sh tools/stream_ceiling_probe.py build
    python tools/stream_ceiling_probe.py sit --rounds 3 --iters 30 --json <out>   (env as research-iron.sh)

Each timed measurement runs in its own child process (a hardware context never outlives it); routes and
volumes rotate across rounds. Built and run on Desktop 2 (Phoenix), 2026-10-13.
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
ROUTES = ("c1", "c2", "c4", "c4x2", "a16", "a16x2")
# Objects per CORE; total bytes = cores * objects * 6,400. 256..2048 objects is 1.6-13 MB per core.
OBJECTS = (256, 512, 1024, 2048)
BUILD = ROOT / "build" / "stream_ceiling_probe"


def layout(route):
    """[(column, [core-count per shim fifo])] for a route."""
    one = {"c1": [1], "c2": [2], "c4": [4], "c4x2": [2, 2], "a16": [4], "a16x2": [2, 2]}[route]
    cols = 4 if route.startswith("a16") else 1
    return [(c, one) for c in range(cols)]


def cores_of(route):
    return sum(sum(f) for _, f in layout(route))


def build_one(route: str, n_objects: int) -> Path:
    import aie.iron as iron
    from aie.dialects.aiex import dma_await_task, dma_free_task, dma_start_task, shim_dma_single_bd_task
    from aie.iron import ObjectFifo, Program, Runtime, Worker
    from aie.iron.device import NPU1, Tile
    from aie.utils.compile.utils import compile_mlir_module

    out_dir = BUILD / f"{route}_{n_objects}"
    work = out_dir / "design.prj"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    iron.set_current_device(NPU1())
    obj_ty = np.ndarray[(A_BYTES,), np.dtype[np.uint8]]

    def core_fn(i):
        i.acquire(1)
        i.release(1)

    workers, shim_ends, fifos = [], [], []   # fifos: (name, bytes it moves)
    for col, groups in layout(route):
        row = 2
        for g, ncores in enumerate(groups):
            name = f"a{col}_{g}"
            col_ty = np.ndarray[(ncores * A_BYTES,), np.dtype[np.uint8]]
            f = ObjectFifo(col_ty if ncores > 1 else obj_ty, name=name, depth=2)
            if ncores == 1:
                ends = [f.cons().forward(tile=Tile(col, 1), obj_type=obj_ty, depth=2, name=f"{name}_fwd").cons()]
            else:
                parts = f.cons().split([k * A_BYTES for k in range(ncores)], tile=Tile(col, 1), depths=[2] * ncores,
                                       obj_types=[obj_ty] * ncores, names=[f"{name}_{k}" for k in range(ncores)])
                ends = [p.cons() for p in parts]
            for e in ends:
                workers.append(Worker(core_fn, [e], tile=Tile(col, row)))
                row += 1
            shim_ends.append(f.prod(tile=Tile(col, 0)))
            fifos.append((name, ncores * n_objects * A_BYTES))
    total = sum(b for _, b in fifos)

    def sequence(inp, outp, *handles):
        inp = getattr(inp, "op", inp)
        tasks, off = [], 0
        for name, nbytes in fifos:
            tasks.append(shim_dma_single_bd_task(name, inp, offset=off, sizes=[nbytes], issue_token=True))
            off += nbytes
        dma_start_task(*tasks)
        for t in tasks:
            dma_await_task(t)
        dma_free_task(*tasks)

    in_t = np.ndarray[(total,), np.dtype[np.uint8]]
    out_t = np.ndarray[(64,), np.dtype[np.uint8]]
    rt = Runtime(sequence, [in_t, out_t] + shim_ends)
    module = Program(iron.get_current_device(), rt, workers=workers).resolve_program()
    (out_dir / "design.mlir").write_text(str(module), encoding="utf-8")
    compile_mlir_module(module, insts_path=out_dir / "insts.bin", xclbin_path=out_dir / "probe.xclbin",
                        work_dir=work, device=iron.get_current_device())
    (out_dir / "meta.json").write_text(json.dumps({"route": route, "objects_per_core": n_objects,
                                                   "cores": cores_of(route), "total_bytes": total,
                                                   "fifos": fifos}), encoding="utf-8")
    return out_dir


def time_one(route: str, n_objects: int, iters: int, warmup: int, timeout_ms: int) -> dict:
    from ignite_xdna.runtime.driver import XrtSiliconHarness
    d = BUILD / f"{route}_{n_objects}"
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    total = meta["total_bytes"]
    with XrtSiliconHarness(0) as h:
        pyxrt = h.pyxrt
        h.load_xclbin(str(d / "probe.xclbin"), "MLIR_AIE")
        bo_instr, n_instr = h.create_instruction_bo_from_bytes((d / "insts.bin").read_bytes())
        bo_in = h.create_host_bo(total, 3)
        bo_out = h.create_host_bo(64, 4)
        bo_in.write(np.arange(total, dtype=np.uint32).astype(np.uint8), 0)
        bo_in.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        run = pyxrt.run(h.kernel)
        for k, arg in enumerate((3, bo_instr, n_instr, bo_in, bo_out)):
            run.set_arg(k, arg)
        done = pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED
        times = []
        for k in range(warmup + iters):
            t0 = time.perf_counter()
            run.start()
            state = run.wait(timeout_ms)
            dt = (time.perf_counter() - t0) * 1e3
            if state != done:
                raise RuntimeError(f"{route} x{n_objects}: dispatch {k} ended in state {state}")
            if k >= warmup:
                times.append(dt)
        del run, bo_in, bo_out, bo_instr
    return {"route": route, "objects_per_core": n_objects, "cores": meta["cores"], "mb": total / 1e6,
            "mean_ms": statistics.fmean(times), "median_ms": statistics.median(times), "min_ms": min(times),
            "times_ms": times}


def fit(points):
    xs, ys = zip(*points)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return slope, my - slope * mx


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--routes", default=",".join(ROUTES))
    b.add_argument("--objects", default=",".join(map(str, OBJECTS)))
    t = sub.add_parser("time", help="one measurement; used by 'sit' as a child process")
    t.add_argument("route")
    t.add_argument("objects", type=int)
    t.add_argument("--iters", type=int, default=30)
    t.add_argument("--warmup", type=int, default=3)
    t.add_argument("--timeout-ms", type=int, default=5000)
    s = sub.add_parser("sit")
    s.add_argument("--routes", default=",".join(ROUTES))
    s.add_argument("--objects", default=",".join(map(str, OBJECTS)))
    s.add_argument("--rounds", type=int, default=3)
    s.add_argument("--iters", type=int, default=30)
    s.add_argument("--json", required=True)
    args = ap.parse_args()

    if args.cmd == "build":
        for route in args.routes.split(","):
            for n in map(int, args.objects.split(",")):
                t0 = time.perf_counter()
                d = build_one(route, n)
                print(f"[build] {route} x{n}: {d.name} in {time.perf_counter() - t0:.1f} s", flush=True)
        return
    if args.cmd == "time":
        print(json.dumps(time_one(args.route, args.objects, args.iters, args.warmup, args.timeout_ms)))
        return

    routes = args.routes.split(",")
    sizes = list(map(int, args.objects.split(",")))
    combos = [(r, n) for r in routes for n in sizes]
    results = []
    for rnd in range(args.rounds):
        order = combos[rnd % len(combos):] + combos[:rnd % len(combos)]
        if rnd % 2:
            order = order[::-1]
        for route, n in order:
            proc = subprocess.run([sys.executable, __file__, "time", route, str(n), "--iters", str(args.iters)],
                                  capture_output=True, text=True, timeout=300)
            if proc.returncode != 0:
                print(proc.stdout[-2000:], proc.stderr[-2000:], sep="\n")
                raise SystemExit(f"[sit] {route} x{n} failed in round {rnd}; stopping - do not retry in a loop")
            rec = json.loads(proc.stdout.strip().splitlines()[-1])
            rec["round"] = rnd
            results.append(rec)
            print(f"[sit] round {rnd} {route:6s} x{n:>5} {rec['mb']:7.1f} MB: mean {rec['mean_ms']:.3f} ms, "
                  f"min {rec['min_ms']:.3f}", flush=True)
    summary = {}
    print("\n[fit] dispatch ms = intercept + slope * MB; rate = 1 / slope")
    for route in routes:
        slope, icpt = fit([(r["mb"], r["mean_ms"]) for r in results if r["route"] == route])
        gbs = 1.0 / slope   # MB per ms == GB/s
        summary[route] = {"ms_per_mb": slope, "intercept_ms": icpt, "gb_per_s": gbs, "cores": cores_of(route)}
        print(f"  {route:6s} cores {cores_of(route):2d}: {gbs:6.2f} GB/s (slope {slope:.5f} ms/MB, intercept {icpt:.3f} ms)")
    Path(args.json).write_text(json.dumps({"results": results, "fit": summary}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

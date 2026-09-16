"""What does a MemTile store-and-forward hop cost per byte, against the same bytes sent directly?

Two MemTile residency designs lost on silicon, the activation ring by 3.40 ms and the weight buffer by 3.63 ms
on yolov8s, and the weight buffer's loss tracked the weight bytes routed through the MemTile at 0.044 ms per
MB. The engine's DEFAULT path still sends every activation (237.7 MB per yolov8s frame) and output (27.8 MB)
through a MemTile split and join; only weights go shim-to-core directly. Whether that default hop pays a
comparable per-byte cost was never measured - the weight buffer's loss may equally come from its own
protocol, where a fill must land whole before its serve begins. This measures the hop in isolation.

One core, the engine's object sizes (6,400 B in, 3,200 B out), one output per input and no compute, so only
transport is timed. Four routes that differ in nothing but the path:

    dd  shim -> core,            core -> shim
    md  shim -> MemTile -> core, core -> shim
    dm  shim -> core,            core -> MemTile -> shim
    mm  shim -> MemTile -> core, core -> MemTile -> shim

Transfers are issued the way the engine's instruction stream issues them, ``shim_dma_single_bd_task`` by
fifo name. Each route is built at several volumes; the slope of dispatch time against megabytes is fitted
per route, and the difference between slopes is the hop's cost per MB, separately for each direction.

    bash scripts/research-iron.sh tools/memtile_hop_probe.py build
    bash scripts/research-iron.sh tools/memtile_hop_probe.py sit --rounds 3 --iters 30 --json <out>

Every timed measurement runs in its own child process, so a hardware context can never outlive its
measurement: the driver allows five per device and the sixth fails with 0xc01e0009, the same code as a
wedge. Routes and volumes are interleaved in a rotating order across rounds.
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

A_BYTES = 6400    # the engine's activation packet
O_BYTES = 3200    # and its output block
ROUTES = ("dd", "md", "dm", "mm")
OBJECTS = (512, 1024, 2048, 4096, 8192)
BUILD = ROOT / "build" / "hop_probe"


def build_one(route: str, n_objects: int) -> Path:
    import aie.iron as iron
    from aie.dialects.aiex import dma_await_task, dma_free_task, dma_start_task, shim_dma_single_bd_task
    from aie.iron import ObjectFifo, Program, Runtime, Worker
    from aie.iron.device import NPU1, Tile
    from aie.utils.compile.utils import compile_mlir_module

    out_dir = BUILD / f"{route}_{n_objects}"
    work = out_dir / "design.prj"
    if work.exists():
        shutil.rmtree(work)    # a stale design.prj can carry a stale core object into the build
    work.mkdir(parents=True, exist_ok=True)

    iron.set_current_device(NPU1())
    shim, mem, core = Tile(0, 0), Tile(0, 1), Tile(0, 2)
    in_ty = np.ndarray[(A_BYTES,), np.dtype[np.uint8]]
    out_ty = np.ndarray[(O_BYTES,), np.dtype[np.uint8]]

    # The shim-side fifos are named "a" and "o" in every route, so the sequence is identical across routes.
    a = ObjectFifo(in_ty, name="a", depth=2)
    core_in = (a.cons().forward(tile=mem, obj_type=in_ty, depth=2, name="a_fwd").cons()
               if route[0] == "m" else a.cons())
    if route[1] == "m":
        o_core = ObjectFifo(out_ty, name="o_core", depth=2)
        o = o_core.cons().forward(tile=mem, obj_type=out_ty, depth=2, name="o")
        core_out = o_core.prod()
    else:
        o = ObjectFifo(out_ty, name="o", depth=2)
        core_out = o.prod()

    def core_fn(i, q):
        # Transport only: take an input object, take an output slot, hand both back. No compute.
        i.acquire(1)
        q.acquire(1)
        i.release(1)
        q.release(1)

    worker = Worker(core_fn, [core_in, core_out], tile=core)

    def sequence(inp, outp, a_p, o_c):
        # IRON hands the sequence RuntimeData wrappers; the BD wants the MLIR value (the emitter's _mem idiom).
        inp, outp = getattr(inp, "op", inp), getattr(outp, "op", outp)
        t_in = shim_dma_single_bd_task("a", inp, offset=0, sizes=[n_objects * A_BYTES])
        t_out = shim_dma_single_bd_task("o", outp, offset=0, sizes=[n_objects * O_BYTES], issue_token=True)
        dma_start_task(t_in, t_out)
        dma_await_task(t_out)
        dma_free_task(t_in, t_out)

    in_t = np.ndarray[(n_objects * A_BYTES,), np.dtype[np.uint8]]
    out_t = np.ndarray[(n_objects * O_BYTES,), np.dtype[np.uint8]]
    rt = Runtime(sequence, [in_t, out_t, a.prod(tile=shim), o.cons(tile=shim)])
    module = Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()
    (out_dir / "design.mlir").write_text(str(module), encoding="utf-8")
    compile_mlir_module(module, insts_path=out_dir / "insts.bin", xclbin_path=out_dir / "probe.xclbin",
                        work_dir=work, device=iron.get_current_device())
    return out_dir


def time_one(route: str, n_objects: int, iters: int, warmup: int, timeout_ms: int) -> dict:
    """Load one build, time ``iters`` dispatches, release everything. Runs inside a child process."""
    from ignite_xdna.runtime.driver import XrtSiliconHarness
    d = BUILD / f"{route}_{n_objects}"
    with XrtSiliconHarness(0) as h:
        pyxrt = h.pyxrt
        h.load_xclbin(str(d / "probe.xclbin"), "MLIR_AIE")
        bo_instr, n_instr = h.create_instruction_bo_from_bytes((d / "insts.bin").read_bytes())
        bo_in = h.create_host_bo(n_objects * A_BYTES, 3)
        bo_out = h.create_host_bo(n_objects * O_BYTES, 4)
        bo_in.write(np.arange(n_objects * A_BYTES, dtype=np.uint32).astype(np.uint8), 0)
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
    return {"route": route, "objects": n_objects, "in_mb": n_objects * A_BYTES / 1e6,
            "out_mb": n_objects * O_BYTES / 1e6, "mean_ms": statistics.fmean(times),
            "median_ms": statistics.median(times), "min_ms": min(times), "times_ms": times}


def fit(points):
    """Least-squares slope and intercept of (megabytes in, milliseconds)."""
    xs, ys = zip(*points)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
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
        # Rotate the order each round so no route always runs first or last.
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
            print(f"[sit] round {rnd} {route} x{n:>5} {rec['in_mb']:6.1f} MB in: mean {rec['mean_ms']:.3f} ms, "
                  f"min {rec['min_ms']:.3f}", flush=True)

    summary = {}
    for route in routes:
        pts = [(r["in_mb"], r["mean_ms"]) for r in results if r["route"] == route]
        slope, icpt = fit(pts)
        summary[route] = {"ms_per_mb_in": slope, "intercept_ms": icpt}
    print("\n[fit] dispatch ms = intercept + slope * MB in (outputs are half the input bytes)")
    for route in routes:
        print(f"  {route}: slope {summary[route]['ms_per_mb_in']:.5f} ms/MB, "
              f"intercept {summary[route]['intercept_ms']:.3f} ms")
    if {"dd", "md", "dm", "mm"} <= set(routes):
        s = {r: summary[r]["ms_per_mb_in"] for r in routes}
        hop_in = s["md"] - s["dd"]
        hop_out = (s["dm"] - s["dd"]) * A_BYTES / O_BYTES   # per MB of OUTPUT bytes
        print(f"  hop on the input path:  {hop_in:+.5f} ms per MB in")
        print(f"  hop on the output path: {hop_out:+.5f} ms per MB out")
        print(f"  additivity check: mm - dd = {s['mm'] - s['dd']:+.5f}, "
              f"(md - dd) + (dm - dd) = {hop_in + s['dm'] - s['dd']:+.5f} ms per MB in")
        summary["hop_in_ms_per_mb"] = hop_in
        summary["hop_out_ms_per_mb"] = hop_out
    Path(args.json).write_text(json.dumps({"results": results, "fit": summary}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

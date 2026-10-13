"""Can a column's second shim stream carry weights AND half the activations, at the full stream rate?

stream_ceiling_probe.py found the engine's activations capped at one ~6.9 GB/s shim MM2S stream per column, and
the column's second stream carrying only weights. Splitting each activation fill by core across the two streams
doubles the shim task count (+5,621 tasks on YOLOv8s, ~2.0 ms, engine_stream_split_tasks_*.log). Alternating whole
quads between the streams does not: a quad (one 6,400 B object for each of the four cores) lands in one of two
MemTile buffers, and every core's MemTile MM2S chain reads the two buffers in the same fixed order, so each core's
packets stay in order. The second stream then has to carry two kinds of traffic, which needs packet switching:
weights fanned out to the four cores, and its quads routed to the MemTile's second input. This measures that
transport, with the engine's object sizes and no compute, against the layout the engine uses today:

    base   stream 0 (circuit): 4 quads per round -> MemTile buffer A -> each core's 6,400 B slice
           stream 1 (circuit broadcast): 4 weight objects per round -> each core's weight input
    alt    stream 0 (circuit): quads 0 and 2 of each round -> MemTile buffer A
           stream 1 (packet): 4 weight objects (one packet, fanned out to the 4 cores), then quads 1 and 3 (one
           packet each) -> MemTile buffer B; each core's MemTile MM2S chain reads A, B, A, B
    base4 / alt4   the same in each of the four columns

A round moves the same bytes on every route: 4 quads (102,400 B) and 4 weight objects (37,888 B) per column. Every
core consumes 4 weight objects and then 4 activation objects per round. The busier stream carries 102,400 B per
round on base and 89,088 B on alt, so at one stream's rate alt's round should take 0.870 of base's. Each shim
channel issues one task per 64 rounds (a descriptor chain repeated once per round, stepping to new addresses), so
the per-task issue cost the engine pays per quad is NOT measured here; that is sized offline. Packet headers are:
stream 1 starts a new packet for every weight run and every quad, as the engine would.

    bash scripts/research-iron.sh tools/stream_share_probe.py build
    python tools/stream_share_probe.py sit --rounds 3 --iters 30 --json <out>   (env as research-iron.sh)

Each timed measurement runs in its own child process; routes and volumes rotate across rounds. Rate = slope of
dispatch time against rounds. Written on Desktop 2 (Phoenix), 2026-10-13; what has been run is in its logs.
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

A_BYTES = 6400            # one core's activation object
QUAD = 4 * A_BYTES        # one object for each of the four cores
W_BYTES = 9472            # one weight object
W_PER_ROUND = 4
Q_PER_ROUND = 4
ROUTES = ("base", "alt", "base4", "alt4")
NROUNDS = (64, 128, 192, 256)
CHUNK = 64                     # rounds per task: the iteration field steps at most 64 times
BUILD = ROOT / "build" / "stream_share_probe"
PKT_W, PKT_A = 1, 2


def round_bytes():
    return Q_PER_ROUND * QUAD + W_PER_ROUND * W_BYTES


def stream_bytes(route):
    """(stream 0, stream 1) bytes per round per column."""
    if route.startswith("base"):
        return Q_PER_ROUND * QUAD, W_PER_ROUND * W_BYTES
    return (Q_PER_ROUND // 2) * QUAD, (Q_PER_ROUND // 2) * QUAD + W_PER_ROUND * W_BYTES


def build_one(route: str, nrounds: int) -> Path:
    import aie.iron as iron
    from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir, WireBundle
    from aie.dialects.aie import EndOp, next_bd
    from aie.dialects.aiex import (bds, dma_await_task, dma_configure_task, dma_free_task, dma_start_task,
                                   shim_dma_bd)
    from aie.iron import (Acquire, Bd, Buffer, DmaChannel, Flow, Lock, PacketFlow, Program, Release, Runtime,
                          TileDma, Worker)
    from aie.iron.dataflow.flow import PacketDest
    from aie.iron.device import NPU1, Tile
    from aie.utils.compile.utils import compile_mlir_module

    assert 1 <= nrounds <= 256
    alt = route.startswith("alt")
    cols = 4 if route.endswith("4") else 1
    out_dir = BUILD / f"{route}_{nrounds}"
    work = out_dir / "design.prj"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    iron.set_current_device(NPU1())
    u8 = np.dtype[np.uint8]

    flows, locks, dmas, workers = [], [], [], []

    def core_fn(w_full, w_space, a_full, a_space):
        for _ in range(W_PER_ROUND):
            w_full.acquire(1)
            w_space.release(1)
        for _ in range(Q_PER_ROUND):
            a_full.acquire(1)
            a_space.release(1)

    shims = []
    for c in range(cols):
        shim = Tile(c, 0, tile_type=AIETileType.ShimNOCTile)
        mem = Tile(c, 1, tile_type=AIETileType.MemTile)
        shims.append(shim)
        cores = [Tile(c, 2 + k, tile_type=AIETileType.CoreTile) for k in range(4)]
        nbuf = 2 if alt else 1
        mbuf = [Buffer(np.ndarray[(2 * QUAD,), u8], name=f"m{c}_{b}", tile=mem) for b in range(nbuf)]
        space = [[Lock(mem, init=2, name=f"m{c}_{b}_space{k}") for k in range(4)] for b in range(nbuf)]
        full = [[Lock(mem, init=0, name=f"m{c}_{b}_full{k}") for k in range(4)] for b in range(nbuf)]
        locks += [lk for grid in (space, full) for row in grid for lk in row]
        chans = []
        for b in range(nbuf):   # MemTile input b: slot s, core k
            chain = [Bd(mbuf[b], offset=s * QUAD + k * A_BYTES, length=A_BYTES, acquires=[Acquire(space[b][k])],
                        releases=[Release(full[b][k])]) for s in range(2) for k in range(4)]
            for i, bd in enumerate(chain):
                bd.next = (i + 1) % len(chain)
            chans.append(DmaChannel(DMAChannelDir.S2MM, b, chain))
        for k in range(4):      # MemTile output k -> core k: slot 0 of A (, B), slot 1 of A (, B)
            chain = [Bd(mbuf[b], offset=s * QUAD + k * A_BYTES, length=A_BYTES, acquires=[Acquire(full[b][k])],
                        releases=[Release(space[b][k])]) for s in range(2) for b in range(nbuf)]
            for i, bd in enumerate(chain):
                bd.next = (i + 1) % len(chain)
            chans.append(DmaChannel(DMAChannelDir.MM2S, k, chain))
            flows.append(Flow(mem, cores[k], src_channel=k, dst_channel=0))
        dmas.append(TileDma(mem, chans))
        for k, core in enumerate(cores):
            abuf = [Buffer(np.ndarray[(A_BYTES,), u8], name=f"a{c}{k}_{s}", tile=core) for s in range(2)]
            wbuf = [Buffer(np.ndarray[(W_BYTES,), u8], name=f"w{c}{k}_{s}", tile=core) for s in range(2)]
            a_space, a_full = Lock(core, init=2, name=f"a{c}{k}_space"), Lock(core, init=0, name=f"a{c}{k}_full")
            w_space, w_full = Lock(core, init=2, name=f"w{c}{k}_space"), Lock(core, init=0, name=f"w{c}{k}_full")
            locks += [a_space, a_full, w_space, w_full]
            achain = [Bd(abuf[s], acquires=[Acquire(a_space)], releases=[Release(a_full)], next=1 - s)
                      for s in range(2)]
            wchain = [Bd(wbuf[s], acquires=[Acquire(w_space)], releases=[Release(w_full)], next=1 - s)
                      for s in range(2)]
            dmas.append(TileDma(core, [DmaChannel(DMAChannelDir.S2MM, 0, achain),
                                       DmaChannel(DMAChannelDir.S2MM, 1, wchain)]))
            workers.append(Worker(core_fn, [w_full, w_space, a_full, a_space], tile=core))
        flows.append(Flow(shim, mem, src_channel=0, dst_channel=0))
        if alt:
            flows.append(PacketFlow(pkt_id=PKT_A + 8 * c, src=shim, dst=mem, src_port=WireBundle.DMA, src_channel=1,
                                    dst_port=WireBundle.DMA, dst_channel=1))
            flows.append(PacketFlow(pkt_id=PKT_W + 8 * c, src=shim, dst=cores[0], src_port=WireBundle.DMA,
                                    src_channel=1, dst_port=WireBundle.DMA, dst_channel=1,
                                    extra_dsts=[PacketDest(core, WireBundle.DMA, 1) for core in cores[1:]]))
        else:
            flows += [Flow(shim, core, src_channel=1, dst_channel=1) for core in cores]

    s0, s1 = stream_bytes(route)
    col_bytes = nrounds * round_bytes()
    total = cols * col_bytes

    rb = round_bytes()

    def chunk_tasks(mem_arg, first, r):
        """Every column's two tasks for rounds first..first+r-1. Round i of column c lives at
        c * col_bytes + i * rb as [stream 0's bytes | stream 1's bytes]; each descriptor's outer (iteration)
        dimension steps one round per repeat, so every round reads new addresses."""
        tasks = []
        for c, shim in enumerate(shims):
            base = c * col_bytes + first * rb
            t0 = dma_configure_task(shim.op, DMAChannelDir.MM2S, 0, repeat_count=r - 1, issue_token=True)
            with bds(t0) as bd:
                with bd[0]:
                    shim_dma_bd(mem_arg, offset=base, sizes=[r, 1, 1, s0], strides=[rb, 0, 0, 1])
                    EndOp()
            t1 = dma_configure_task(shim.op, DMAChannelDir.MM2S, 1, repeat_count=r - 1, issue_token=True)
            woff = base + s0
            if not alt:
                with bds(t1) as bd:
                    with bd[0]:
                        shim_dma_bd(mem_arg, offset=woff, sizes=[r, 1, 1, s1], strides=[rb, 0, 0, 1])
                        EndOp()
            else:
                n_q1 = Q_PER_ROUND // 2
                with bds(t1) as bd:
                    with bd[0]:
                        shim_dma_bd(mem_arg, offset=woff, sizes=[r, 1, 1, W_PER_ROUND * W_BYTES],
                                    strides=[rb, 0, 0, 1], packet=(0, PKT_W + 8 * c))
                        next_bd(bd[1])
                    for q in range(n_q1):
                        with bd[1 + q]:
                            shim_dma_bd(mem_arg, offset=woff + W_PER_ROUND * W_BYTES + q * QUAD,
                                        sizes=[r, 1, 1, QUAD], strides=[rb, 0, 0, 1], packet=(0, PKT_A + 8 * c))
                            if q + 1 < n_q1:
                                next_bd(bd[2 + q])
                            else:
                                EndOp()
            tasks += [t0, t1]
        return tasks

    def sequence(inp, outp):
        mem_arg = getattr(inp, "op", inp)
        # Rounds go out in chunks of at most 64 (the iteration field); two chunks in flight per channel, so the
        # next chunk is queued before the previous one is awaited and the streams never idle on the host.
        prev = None
        for first in range(0, nrounds, CHUNK):
            cur = chunk_tasks(mem_arg, first, min(CHUNK, nrounds - first))
            dma_start_task(*cur)
            if prev is not None:
                for t in prev:
                    dma_await_task(t)
                dma_free_task(*prev)
            prev = cur
        for t in prev:
            dma_await_task(t)
        dma_free_task(*prev)

    in_t = np.ndarray[(total,), u8]
    out_t = np.ndarray[(64,), u8]
    rt = Runtime(sequence, [in_t, out_t])
    for f in flows:
        rt.add_flow(f)
    for lk in locks:
        rt.add_lock(lk)
    for d in dmas:
        rt.add_tile_dma(d)
    module = Program(iron.get_current_device(), rt, workers=workers).resolve_program()
    (out_dir / "design.mlir").write_text(str(module), encoding="utf-8")
    compile_mlir_module(module, insts_path=out_dir / "insts.bin", xclbin_path=out_dir / "probe.xclbin",
                        work_dir=work, device=iron.get_current_device())
    (out_dir / "meta.json").write_text(json.dumps({"route": route, "rounds": nrounds, "cols": cols,
                                                   "total_bytes": total, "stream_bytes_per_round": [s0, s1]}),
                                       encoding="utf-8")
    return out_dir


def time_one(route: str, nrounds: int, iters: int, warmup: int, timeout_ms: int) -> dict:
    from ignite_xdna.runtime.driver import XrtSiliconHarness
    d = BUILD / f"{route}_{nrounds}"
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
                raise RuntimeError(f"{route} x{nrounds}: dispatch {k} ended in state {state}")
            if k >= warmup:
                times.append(dt)
        del run, bo_in, bo_out, bo_instr
    return {"route": route, "rounds": nrounds, "cols": meta["cols"], "mb": total / 1e6,
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
    b.add_argument("--rounds", default=",".join(map(str, NROUNDS)))
    t = sub.add_parser("time", help="one measurement; used by 'sit' as a child process")
    t.add_argument("route")
    t.add_argument("nrounds", type=int)
    t.add_argument("--iters", type=int, default=30)
    t.add_argument("--warmup", type=int, default=3)
    t.add_argument("--timeout-ms", type=int, default=5000)
    s = sub.add_parser("sit")
    s.add_argument("--routes", default=",".join(ROUTES))
    s.add_argument("--nrounds", default=",".join(map(str, NROUNDS)))
    s.add_argument("--rounds", type=int, default=3, help="sitting rounds (rotations), not transfer rounds")
    s.add_argument("--iters", type=int, default=30)
    s.add_argument("--json", required=True)
    args = ap.parse_args()

    if args.cmd == "build":
        for route in args.routes.split(","):
            for n in map(int, args.rounds.split(",")):
                t0 = time.perf_counter()
                d = build_one(route, n)
                print(f"[build] {route} x{n}: {d.name} in {time.perf_counter() - t0:.1f} s", flush=True)
        return
    if args.cmd == "time":
        print(json.dumps(time_one(args.route, args.nrounds, args.iters, args.warmup, args.timeout_ms)))
        return

    routes = args.routes.split(",")
    sizes = list(map(int, args.nrounds.split(",")))
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
            print(f"[sit] round {rnd} {route:6s} x{n:>4} {rec['mb']:7.1f} MB: mean {rec['mean_ms']:.3f} ms, "
                  f"min {rec['min_ms']:.3f}", flush=True)
    summary = {}
    print("\n[fit] dispatch ms = intercept + slope * rounds; busier-stream rate = its bytes per round / slope")
    for route in routes:
        slope, icpt = fit([(r["rounds"], r["mean_ms"]) for r in results if r["route"] == route])
        s0, s1 = stream_bytes(route)
        busy = max(s0, s1)
        summary[route] = {"ms_per_round": slope, "intercept_ms": icpt, "busier_stream_gb_s": busy / slope / 1e6,
                          "column_gb_s": round_bytes() / slope / 1e6}
        print(f"  {route:6s}: {slope * 1e3:7.3f} us/round, intercept {icpt:.3f} ms; busier stream "
              f"{busy / slope / 1e6:5.2f} GB/s; per column {round_bytes() / slope / 1e6:5.2f} GB/s")
    for a, b in (("alt", "base"), ("alt4", "base4")):
        if a in summary and b in summary:
            print(f"  {a}/{b} round-time ratio: {summary[a]['ms_per_round'] / summary[b]['ms_per_round']:.3f} "
                  f"(one-stream-rate prediction 0.870)")
    Path(args.json).write_text(json.dumps({"results": results, "fit": summary}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

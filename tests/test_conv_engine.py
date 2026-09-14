"""Phoenix convolution engine: build, emulate and verify on silicon.

  python tests/test_conv_engine.py --offline            emulator self-checks (any env)
  python tests/test_conv_engine.py --compile            build the engine xclbin + synthetic sequence (ironenv)
  python tests/test_conv_engine.py --hardware           run the synthetic sequence on Device 0 (ironenv)

The synthetic sequence drives every packet kind the core program implements
(3x3 stride 1 with HardSwish, chunked 1x1 with partial sums, 3x3 stride 2,
2x upsampling, 5x5 max pool and a residual packet) through all sixteen cores
and compares every output byte with the NumPy emulator.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ignite_xdna.compiler import engine_emulator as em  # noqa: E402

BUILD_DEFAULT = ROOT / "build" / "conv_engine"
COLS, ROWS = 4, 4


def _hs_params_identityish():
    """HardSwish constants for scale 0.25 in/out (checked exactly by the compiler later)."""
    return em.HardSwishParams(a1=5461, b1=524288, s1=13, qmax=128, k2=16386, s2=14, ysh=7)


def synthetic_scenarios(seed: int):
    """Per-column list of (W packet, [[A packets per round] per core]) and the expected outputs.

    Every column runs the same scenario list; data differ per column and core.
    """
    rng = np.random.default_rng(seed)
    scenarios = []

    def rand_w(taps, ncin):
        return rng.integers(-40, 41, size=(taps, ncin, 4, 8, 8), dtype=np.int8)

    def rand_bias(shift):
        return rng.integers(-2000, 2000, size=32, dtype=np.int32) + (128 << shift)

    def rand_a(n):
        return [rng.integers(0, 256, size=em.A_BYTES, dtype=np.uint8) for _ in range(n)]

    # 1. 3x3 stride 1, four input blocks, HardSwish, three tiles per core.
    hdr = em.PacketHeader(op=em.OP_CONV, k=3, stride=1, ncin=4, nco=4, flags=em.F_EMIT | em.F_HSWISH,
                          shift_out=9, hs=_hs_params_identityish(), count_out=3, rows_in=8, cols_in=25,
                          plane_bytes=1600)
    scenarios.append(("k3s1_hswish", em.pack_w_packet(hdr, rand_bias(9), rand_w(9, 4)), 3, 0))
    # 2. 1x1 in two chunks: accumulate, then emit with partial sums (linear output).
    hdr = em.PacketHeader(op=em.OP_CONV, k=1, stride=1, ncin=8, nco=4, flags=0, shift_out=0, count_out=0,
                          count_acc=1, rows_in=5, cols_in=20, plane_bytes=800)
    scenarios.append(("k1_chunk0", em.pack_w_packet(hdr, rand_bias(7), rand_w(1, 8)), 0, 1))
    hdr = em.PacketHeader(op=em.OP_CONV, k=1, stride=1, ncin=8, nco=4, flags=em.F_LOAD_PSUM | em.F_EMIT,
                          shift_out=7, count_out=1, rows_in=5, cols_in=20, plane_bytes=800)
    scenarios.append(("k1_chunk1", em.pack_w_packet(hdr, rand_bias(7), rand_w(1, 8)), 1, 0))
    # 3. 3x3 stride 2, one block of contiguous rows, HardSwish.
    hdr = em.PacketHeader(op=em.OP_CONV, k=3, stride=2, ncin=1, nco=4, flags=em.F_EMIT | em.F_HSWISH,
                          shift_out=7, hs=_hs_params_identityish(), count_out=2, rows_in=16, cols_in=50,
                          plane_bytes=6400)
    scenarios.append(("k3s2_hswish", em.pack_w_packet(hdr, rand_bias(7), rand_w(9, 1)), 2, 0))
    # 4. 2x upsampled 1x1 accumulate (phases differ per core row) followed by a held
    #    emit-less chunk and a residual packet.
    hdr = em.PacketHeader(op=em.OP_CONV, k=1, stride=1, ncin=8, nco=4, flags=em.F_UP2, shift_out=0,
                          count_out=0, count_acc=1, phases=(0, 1, 0, 1), rows_in=5, cols_in=20, plane_bytes=800)
    scenarios.append(("k1_up2_acc", em.pack_w_packet(hdr, rand_bias(7), rand_w(1, 8)), 0, 1))
    hdr = em.PacketHeader(op=em.OP_CONV, k=1, stride=1, ncin=8, nco=4,
                          flags=em.F_LOAD_PSUM | em.F_HOLD | em.F_HSWISH, shift_out=7,
                          hs=_hs_params_identityish(), count_out=0, count_acc=1, rows_in=5, cols_in=20,
                          plane_bytes=800)
    scenarios.append(("k1_hold", em.pack_w_packet(hdr, rand_bias(7), rand_w(1, 8)), 0, 1))
    hdr = em.PacketHeader(op=em.OP_RESIDUAL, ncin=4, nco=4, flags=em.F_EMIT, rsh=1, count_out=1)
    scenarios.append(("residual", em.pack_w_packet(hdr, np.zeros(32, np.int32), None), 1, 0))
    # 5. 5x5 max pool: two blocks held, then two more blocks emitted with the held pair.
    hdr = em.PacketHeader(op=em.OP_MAXPOOL, ncin=2, nco=4, flags=em.F_HOLD, count_out=0, count_acc=1,
                          rows_in=16, cols_in=25, plane_bytes=3200)
    scenarios.append(("maxpool_hold", em.pack_w_packet(hdr, np.zeros(32, np.int32), None), 0, 1))
    hdr = em.PacketHeader(op=em.OP_MAXPOOL, ncin=2, nco=4, flags=em.F_EMIT | em.F_LOAD_PSUM, count_out=1,
                          rows_in=16, cols_in=25, plane_bytes=3200)
    scenarios.append(("maxpool_emit", em.pack_w_packet(hdr, np.zeros(32, np.int32), None), 1, 0))

    plan = []  # per column: list of dicts
    for c in range(COLS):
        col = []
        for name, wpkt, n_out, n_acc in scenarios:
            rounds = n_out + n_acc
            a_rounds = [[rand_a(1)[0] for _ in range(ROWS)] for _ in range(rounds)]  # [round][core]
            col.append({"name": name, "w": wpkt, "n_out": n_out, "n_acc": n_acc, "a": a_rounds})
        plan.append(col)
    return plan


def emulate_plan(plan):
    """Expected per-column output objects in drain order: list per column of [round][core] arrays."""
    expected = []
    for c in range(COLS):
        states = [em.CoreState() for _ in range(ROWS)]
        col_out = []
        for sc in plan[c]:
            for rnd in range(sc["n_out"] + sc["n_acc"]):
                outs = []
                for r in range(ROWS):
                    o = em.run_packet(sc["w"], sc["a"][rnd][r], states[r], r)
                    outs.append(o)
                if all(o is not None for o in outs):
                    col_out.append(np.concatenate(outs))
                elif any(o is not None for o in outs):
                    raise AssertionError("cores disagree on output emission")
        expected.append(col_out)
    return expected


def layout_buffers(plan):
    """Pack the plan into the two DDR buffers and return (ws, wp, meta)."""
    wp_parts, ws_parts = [], []
    meta = {"cols": []}
    wp_off = 0
    ws_off = 0
    for c in range(COLS):
        col_meta = {"scenarios": []}
        for sc in plan[c]:
            w_off = wp_off
            wp_parts.append(sc["w"])
            wp_off += em.W_BYTES
            a_offs = []
            for rnd in range(sc["n_out"] + sc["n_acc"]):
                a_offs.append(ws_off)
                ws_parts.append(np.concatenate(sc["a"][rnd]))
                ws_off += ROWS * em.A_BYTES
            col_meta["scenarios"].append({"name": sc["name"], "w_off": w_off, "n_out": sc["n_out"],
                                          "n_acc": sc["n_acc"], "a_offs": a_offs})
        meta["cols"].append(col_meta)
    # Output region after all inputs, 64-byte aligned.
    out_base = (ws_off + 63) // 64 * 64
    ws_parts.append(np.zeros(out_base - ws_off, dtype=np.uint8))
    o_off = out_base
    for c in range(COLS):
        for sc in meta["cols"][c]["scenarios"]:
            sc["o_offs"] = []
            for _ in range(sc["n_out"]):
                sc["o_offs"].append(o_off)
                o_off += ROWS * em.O_BYTES
    ws = np.concatenate(ws_parts + [np.zeros(o_off - out_base, dtype=np.uint8)])
    wp = np.concatenate(wp_parts)
    meta["ws_bytes"] = int(ws.size)
    meta["wp_bytes"] = int(wp.size)
    meta["out_base"] = int(out_base)
    return ws, wp, meta


def column_programs(meta):
    """Translate the plan into per-column item lists (see engine_sequence)."""
    from ignite_xdna.compiler.engine_sequence import DmaPattern
    programs = []
    for c in range(COLS):
        items = []
        for sc in meta["cols"][c]["scenarios"]:
            items.append(("w", sc["w_off"], em.W_BYTES, sc["n_out"] + sc["n_acc"]))
            for i, a_off in enumerate(sc["a_offs"]):
                items.append(("a", [DmaPattern("ws", a_off + r * em.A_BYTES, (em.A_BYTES,), (1,))
                                    for r in range(ROWS)]))
                if i < sc["n_out"]:
                    items.append(("o", DmaPattern("ws", sc["o_offs"][i], (ROWS * em.O_BYTES,), (1,))))
        programs.append(items)
    return programs


def sequence_from_meta(meta):
    """Return a sequence_body(ws, wp) that replays the plan two rounds deep per column."""
    from ignite_xdna.compiler.engine_sequence import SequenceEmitter
    from kernels.aie2.conv_engine import design as eng

    programs = column_programs(meta)

    def body(ws, wp):
        emitter = SequenceEmitter(ws, wp, eng.WS_BYTES, eng.WP_BYTES, {c: eng.fifo_names(c) for c in range(COLS)})
        emitter.run_column_programs(programs, bd_budget=14)
    return body


def compile_engine(build_dir: Path, seed: int):
    import aie.iron as iron
    from aie.iron.device import NPU1
    from aie.utils.compile.utils import compile_mlir_module
    from kernels.aie2.conv_engine import design as eng

    iron.set_current_device(NPU1())
    plan = synthetic_scenarios(seed)
    ws, wp, meta = layout_buffers(plan)
    build_dir.mkdir(parents=True, exist_ok=True)
    np.save(build_dir / "ws.npy", ws)
    np.save(build_dir / "wp.npy", wp)
    expected = emulate_plan(plan)
    flat = [np.concatenate(col) for col in expected]
    meta["expected_lengths"] = [int(f.size) for f in flat]
    np.save(build_dir / "expected.npy", np.concatenate(flat))
    (build_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    t0 = time.perf_counter()
    program = eng.build_program(iron.get_current_device(), sequence_from_meta(meta))
    module = program.resolve_program()
    work = build_dir / "design.prj"
    # IRON reuses an existing engine.o; a stale one would silently ship an old kernel.
    if work.exists():
        import shutil
        shutil.rmtree(work)
    work.mkdir(exist_ok=True)
    (build_dir / "design.mlir").write_text(str(module), encoding="utf-8")
    compile_mlir_module(module, insts_path=build_dir / "insts.bin", xclbin_path=build_dir / "design.xclbin",
                        work_dir=work, device=iron.get_current_device())
    dt = time.perf_counter() - t0
    manifest = {"seed": seed, "compile_seconds": round(dt, 1),
                "kernel_sha256": hashlib.sha256(eng.KERNEL_SOURCE.read_bytes()).hexdigest(),
                "xclbin_sha256": hashlib.sha256((build_dir / "design.xclbin").read_bytes()).hexdigest(),
                "insts_bytes": (build_dir / "insts.bin").stat().st_size}
    (build_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"[compile] xclbin + insts ({manifest['insts_bytes']} B) in {dt:.1f} s -> {build_dir}")
    return manifest


def run_hardware(build_dir: Path, device_idx: int = 0, iters: int = 3):
    from ignite_xdna.runtime.driver import XrtSiliconHarness

    meta = json.loads((build_dir / "meta.json").read_text(encoding="utf-8"))
    ws = np.load(build_dir / "ws.npy")
    wp = np.load(build_dir / "wp.npy")
    flat = np.load(build_dir / "expected.npy")
    expected, pos = [], 0
    for n in meta["expected_lengths"]:
        expected.append(flat[pos:pos + n])
        pos += n
    harness = XrtSiliconHarness(device_idx=device_idx)
    try:
        harness.load_xclbin(str(build_dir / "design.xclbin"))
        bo_instr, n_instr = harness.create_instruction_bo(str(build_dir / "insts.bin"))
        bo_ws = harness.create_host_bo(int(ws.size), 3)
        bo_wp = harness.create_host_bo(int(wp.size), 4)
        bo_wp.write(wp.tobytes(), 0)
        bo_wp.sync(harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        results = []
        for it in range(iters):
            bo_ws.write(ws.tobytes(), 0)
            bo_ws.sync(harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            t0 = time.perf_counter()
            run, state = harness.dispatch_kernel(bo_instr, n_instr, bo_ws, bo_wp, timeout_ms=5000)
            dt = (time.perf_counter() - t0) * 1e3
            if str(state) != "ert_cmd_state.ERT_CMD_STATE_COMPLETED":
                raise RuntimeError(f"dispatch state {state}")
            bo_ws.sync(harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            got = np.frombuffer(bo_ws.read(int(ws.size), 0), dtype=np.uint8)
            mismatches = {}
            for c in range(COLS):
                exp = expected[c]
                offs = [o for sc in meta["cols"][c]["scenarios"] for o in sc["o_offs"]]
                chunks = [got[o:o + ROWS * em.O_BYTES] for o in offs]
                hw = np.concatenate(chunks) if chunks else np.zeros(0, np.uint8)
                if hw.size != exp.size or not np.array_equal(hw, exp):
                    diff = np.flatnonzero(hw != exp) if hw.size == exp.size else np.arange(exp.size)
                    mismatches[c] = int(diff.size)
                    # Report the first differing scenario for diagnosis.
                    pos = 0
                    for sc in meta["cols"][c]["scenarios"]:
                        n = sc["n_out"] * ROWS * em.O_BYTES
                        seg = np.flatnonzero(hw[pos:pos + n] != exp[pos:pos + n]) if hw.size == exp.size else []
                        if len(seg):
                            print(f"  col {c} scenario {sc['name']}: {len(seg)} differing bytes, first at "
                                  f"{seg[0]} (core {seg[0] // em.O_BYTES % ROWS}) hw={hw[pos + seg[0]]} "
                                  f"exp={exp[pos + seg[0]]}")
                        pos += n
            results.append({"iter": it, "ms": round(dt, 3), "mismatch_bytes": mismatches})
            print(f"[hardware] iter {it}: {dt:.3f} ms, mismatching bytes per column: {mismatches or 'none'}")
        bo_ws = bo_wp = bo_instr = None
    finally:
        harness.close()
    return results


class EngineEmulatorOffline(unittest.TestCase):
    def test_rne_shift_matches_numpy_banker_rounding(self):
        x = np.arange(-4096, 4096, dtype=np.int64)
        for s in (1, 2, 5, 9):
            ref = np.round(x / (1 << s)).astype(np.int64)  # numpy rounds half to even
            self.assertTrue(np.array_equal(em.rne_shift(x, s), ref), s)

    def test_packet_roundtrip(self):
        hdr = em.PacketHeader(op=em.OP_CONV, k=3, stride=2, ncin=1, nco=4, flags=em.F_EMIT | em.F_HSWISH,
                              shift_out=7, hs=_hs_params_identityish(), count_out=5, phases=(0, 1, 0, 1),
                              rows_in=16, cols_in=50, plane_bytes=6400)
        w = np.arange(9 * 1 * 4 * 64, dtype=np.int64).reshape(9, 1, 4, 8, 8) % 200 - 100
        pkt = em.pack_w_packet(hdr, np.arange(32, dtype=np.int32), w.astype(np.int8))
        h2, b2, w2 = em.unpack_w_packet(pkt)
        self.assertEqual(h2, hdr)
        self.assertTrue(np.array_equal(b2, np.arange(32, dtype=np.int32)))
        self.assertTrue(np.array_equal(w2, w.astype(np.int8)))

    def test_synthetic_plan_emulates(self):
        plan = synthetic_scenarios(1)
        expected = emulate_plan(plan)
        self.assertEqual(len(expected), COLS)
        # 3 + 1 + 2 + 1 (residual) + 1 (pool emit) output rounds per column
        self.assertEqual(len(expected[0]), 8)
        self.assertTrue(all(o.size == ROWS * em.O_BYTES for o in expected[0]))

    def test_up2_expand_duplicates_pixels(self):
        src = np.zeros((10, 4, 20, 8), dtype=np.uint8)
        src[:, :, :, 0] = np.arange(20, dtype=np.uint8)[None, None, :]
        src[:, :, :, 1] = np.arange(4, dtype=np.uint8)[None, :, None]
        raw = np.zeros(em.A_BYTES, dtype=np.uint8)
        raw[:src.size] = src.reshape(-1)
        tile = em.up2_expand(raw, 1)[:8 * 800].reshape(8, 5, 20, 8)
        self.assertEqual([int(v) for v in tile[0, 0, :, 0]], [x >> 1 for x in range(20)])
        self.assertEqual([int(v) for v in tile[0, :, 0, 1]], [(r + 1) >> 1 for r in range(5)])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--build-dir", default=str(BUILD_DEFAULT))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--device", type=int, default=0)
    args, rest = parser.parse_known_args()
    # aiecc and Peano resolve paths against the work dir, so it must be absolute.
    build_dir = Path(args.build_dir).resolve()
    if args.compile:
        compile_engine(build_dir, args.seed)
    if args.hardware:
        results = run_hardware(build_dir, args.device, args.iters)
        bad = [r for r in results if r["mismatch_bytes"]]
        if bad:
            print(f"[hardware] FAILED: {len(bad)} of {len(results)} iterations mismatch")
            sys.exit(1)
        print(f"[hardware] PASS: {len(results)} iterations bit-exact against the emulator")
    if args.offline or not (args.compile or args.hardware):
        sys.argv = [sys.argv[0]] + rest
        unittest.main(verbosity=2)


if __name__ == "__main__":
    main()

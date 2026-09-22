"""Sixteen cores of the bf16 engine, byte-checked against the emulator on silicon.

The single-core harness (``engine_bf16.py``) proved the kernel's arithmetic: thirteen cases,
1,600 of 1,600 bytes each. It proved nothing about the TRANSPORT, because it has none - one core,
one ObjectFifo per operand, no MemTile. This is the other half, and everything it exercises is
unproven before it runs:

* the activation split and the output join at the MemTile, whose offsets are counted in ELEMENTS
  rather than bytes (``design.py``) and have only ever been checked by reading the emitted MLIR,
* the ``count_out`` / ``count_acc`` loops, which the core drives from header words 5 and 6,
* sixteen cores advancing in lockstep on one instruction stream,
* and ``scratch`` surviving between packets while the accumulate-only loop aliases it as the
  output pointer.

It needs no schedule and no model. ``SequenceEmitter`` works in byte offsets over the two DDR
buffers and in fifo symbol names, and packing goes through the bf16 emulator, so the whole path
from packet bytes to silicon exists today - which is why this runs before the packer is
parameterised rather than after it.

Two unit conventions meet here and it is worth stating once: the DDR side is BYTES (``DmaPattern``
says so, and the design's workspace argument is uint8), while the MemTile split and join offsets
are ELEMENTS of the fifo's own type. int8 cannot tell the two apart anywhere.

    bash scripts/research-iron.sh kernels/bf16_conv/engine_bf16_16core.py --compile
    bash scripts/research-iron.sh kernels/bf16_conv/engine_bf16_16core.py --run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ignite_xdna.compiler import engine_bf16_emulator as em  # noqa: E402

COLS = 4
ROWS = 4

W_BYTES = 9472
A_BYTES = 12800
O_BYTES = 3200
A_ELEMS = A_BYTES // 2
O_ELEMS = O_BYTES // 2


def emits(header) -> bool:
    """Does this packet write an output object?

    The two opcodes choose their destination differently and the difference is not cosmetic:
    a convolution writes `out` only under F_EMIT, while a residual always retires somewhere and
    picks `scratch` only under F_HOLD. Getting this wrong here would put an emitting packet in
    the accumulate-only loop, where `out` is aliased onto `scratch` - see design._core_fn.
    """
    op, flags = int(header[em.H_OP]), int(header[em.H_FLAGS])
    if op == em.OP_RESIDUAL:
        return not (flags & em.F_HOLD)
    if op == em.OP_CONV:
        return bool(flags & em.F_EMIT)
    return False


def pack_w(header, bias_f, wts_f) -> np.ndarray:
    """One 9,472 B weight packet: 128 B header, 128 B replicated bias, then bf16 weights."""
    w = np.zeros(W_BYTES, np.uint8)
    w[:128] = np.asarray(header, np.int32).view(np.uint8)
    w[128:256] = np.asarray(bias_f, np.float32).astype(bfloat16).view(np.uint8)
    wb = np.asarray(wts_f, np.float32).astype(bfloat16).view(np.uint8)
    if wb.size > W_BYTES - 256:
        raise ValueError(f"weights {wb.size} B exceed the {W_BYTES - 256} B payload")
    w[256:256 + wb.size] = wb
    return w


def unpack_w(w: np.ndarray):
    """The three arrays run_packet wants, back out of a packed weight packet.

    The weight region is sliced to the shape the HEADER implies, not handed over whole: the
    packet is a fixed 9,472 B and the payload it actually carries is
    `k*k*ncin*NCO*8*4` bf16, so the emulator's reshape would fail on the 4,608 the buffer holds.
    The core reads exactly this many too - it walks ky, kx, input block and stops.
    """
    header = w[:128].view(np.int32)
    bias = w[128:256].view(bfloat16).astype(np.float32)
    k, ncin = int(header[em.H_K]), int(header[em.H_NCIN])
    n = k * k * ncin * em.NCO * 8 * 4
    wts = w[256:256 + n * 2].view(bfloat16).astype(np.float32)
    return header, bias, wts


def geometry(k, stride):
    rows_in = (em.TILE_ROWS - 1) * stride + k
    cols_in = (em.TILE_COLS - 1) * stride + k
    return rows_in, cols_in, rows_in * cols_in * 8


def scenarios(seed: int):
    """(name, weight packet, count_out, count_acc) - one W packet per scenario, as the core expects.

    A scenario's header serves every one of its rounds, so a scenario is either all-emitting or
    all-accumulating; a chain that accumulates and then emits is two scenarios sharing the core's
    psum, exactly as the int8 synthetic plan builds one.
    """
    rng = np.random.default_rng(seed)
    out = []

    def w_for(k, ncin):
        return rng.normal(scale=0.25, size=k * k * ncin * em.NCO * 32).astype(np.float32)

    def bias():
        # Replicated to mmul's 4x4 C shape on the host, element m*4+n being output channel n:
        # bf16 has no 4-element load, so the core cannot broadcast it itself.
        per_ch = rng.normal(scale=0.1, size=em.NCO * 4).astype(np.float32).reshape(em.NCO, 4)
        return np.repeat(per_ch[:, None, :], 4, axis=1).reshape(-1)

    # 1. 3x3 stride 1, one input block, ReLU, two tiles per core.
    r_in, c_in, plane = geometry(3, 1)
    h = em.make_header(op=em.OP_CONV, k=3, stride=1, ncin=1, flags=em.F_EMIT | em.F_RELU,
                       count_out=2, count_acc=0, rows_in=r_in, cols_in=c_in, plane_elems=plane)
    out.append(("k3s1_relu", pack_w(h, bias(), w_for(3, 1)), 2, 0))

    # 2. 1x1 over eight input blocks - the shape that fills the 12,800 B activation packet.
    r_in, c_in, plane = geometry(1, 1)
    h = em.make_header(op=em.OP_CONV, k=1, stride=1, ncin=8, flags=em.F_EMIT,
                       count_out=1, count_acc=0, rows_in=r_in, cols_in=c_in, plane_elems=plane)
    out.append(("k1s1_c8", pack_w(h, bias(), w_for(1, 8)), 1, 0))

    # 3. A chunked 1x1: accumulate into psum with NO output object, then emit loading it back.
    #    This is the only pair where count_acc is non-zero and the second loop of _core_fn runs.
    h = em.make_header(op=em.OP_CONV, k=1, stride=1, ncin=4, flags=0,
                       count_out=0, count_acc=1, rows_in=r_in, cols_in=c_in, plane_elems=plane)
    out.append(("k1_chunk0", pack_w(h, bias(), w_for(1, 4)), 0, 1))
    h = em.make_header(op=em.OP_CONV, k=1, stride=1, ncin=4, flags=em.F_LOAD_PSUM | em.F_EMIT,
                       count_out=1, count_acc=0, rows_in=r_in, cols_in=c_in, plane_elems=plane)
    out.append(("k1_chunk1", pack_w(h, bias(), w_for(1, 4)), 1, 0))

    # 4. Hold, then add. The held packet runs in the accumulate-only loop, where `out` IS `scratch`
    #    - it is F_HOLD, so the core writes scratch either way, and the residual then reads it back
    #    and retires to a real output object. If that aliasing were wrong this case reads garbage.
    h = em.make_header(op=em.OP_CONV, k=1, stride=1, ncin=1, flags=em.F_HOLD,
                       count_out=0, count_acc=1, rows_in=r_in, cols_in=c_in, plane_elems=plane)
    out.append(("k1_hold", pack_w(h, bias(), w_for(1, 1)), 0, 1))
    h = em.make_header(op=em.OP_RESIDUAL, k=1, stride=1, ncin=1, flags=em.F_EMIT | em.F_RELU6,
                       count_out=1, count_acc=0, rows_in=r_in, cols_in=c_in, plane_elems=plane)
    out.append(("residual_relu6", pack_w(h, np.zeros(em.NCO * 16, np.float32),
                                         np.zeros(em.NCO * 32, np.float32)), 1, 0))

    # 5. 3x3 stride 2, the downsample - its load path unzips eight pixels and keeps four.
    r_in, c_in, plane = geometry(3, 2)
    h = em.make_header(op=em.OP_CONV, k=3, stride=2, ncin=1, flags=em.F_EMIT,
                       count_out=1, count_acc=0, rows_in=r_in, cols_in=c_in, plane_elems=plane)
    out.append(("k3s2", pack_w(h, bias(), w_for(3, 1)), 1, 0))
    return out


def build_plan(seed: int):
    """Per column, the scenario list with its own activation bytes for every core and round."""
    rng = np.random.default_rng(seed + 1)
    scs = scenarios(seed)
    plan = []
    for c in range(COLS):
        col = []
        for name, w, n_out, n_acc in scs:
            rounds = n_out + n_acc
            a = [[np.asarray(rng.normal(scale=1.5, size=A_ELEMS), np.float32)
                  .astype(bfloat16).view(np.uint8) for _ in range(ROWS)] for _ in range(rounds)]
            col.append({"name": name, "w": w, "n_out": n_out, "n_acc": n_acc, "a": a})
        plan.append(col)
    return plan


def emulate_plan(plan):
    """Expected output objects per column, in drain order, as raw bytes.

    psum and scratch persist per core across scenarios because they are per-core BUFFERS on the
    device, not per-packet state - which is exactly what the chunk and hold pairs depend on.
    """
    expected = []
    for c in range(COLS):
        psum = [np.zeros(em.PSUM_FLOATS, np.float32) for _ in range(ROWS)]
        scratch = [np.zeros(em.SCRATCH_ELEMS, np.float32) for _ in range(ROWS)]
        col_out = []
        for sc in plan[c]:
            header, bias, wts = unpack_w(sc["w"])
            for rnd in range(sc["n_out"] + sc["n_acc"]):
                outs = []
                for r in range(ROWS):
                    act = sc["a"][rnd][r].view(bfloat16).astype(np.float32)
                    o = np.zeros(em.OUT_ELEMS, np.float32)
                    em.run_packet(header, act, wts, bias, psum[r], o, scratch[r], core_row=r)
                    if emits(header):
                        outs.append(em.bf16_bits(o).view(np.uint8))
                if outs:
                    col_out.append(np.concatenate(outs))
        expected.append(col_out)
    return expected


def layout_buffers(plan):
    """Pack the plan into the two DDR buffers and return (ws, wp, meta). Byte offsets throughout."""
    wp_parts, ws_parts = [], []
    meta = {"cols": []}
    wp_off = ws_off = 0
    for c in range(COLS):
        col_meta = {"scenarios": []}
        for sc in plan[c]:
            w_off = wp_off
            wp_parts.append(sc["w"])
            wp_off += W_BYTES
            a_offs = []
            for rnd in range(sc["n_out"] + sc["n_acc"]):
                a_offs.append(ws_off)
                ws_parts.append(np.concatenate(sc["a"][rnd]))
                ws_off += ROWS * A_BYTES
            col_meta["scenarios"].append({"name": sc["name"], "w_off": w_off, "n_out": sc["n_out"],
                                          "n_acc": sc["n_acc"], "a_offs": a_offs})
        meta["cols"].append(col_meta)
    out_base = (ws_off + 63) // 64 * 64
    ws_parts.append(np.zeros(out_base - ws_off, np.uint8))
    o_off = out_base
    for c in range(COLS):
        for sc in meta["cols"][c]["scenarios"]:
            sc["o_offs"] = []
            for _ in range(sc["n_out"]):
                sc["o_offs"].append(o_off)
                o_off += ROWS * O_BYTES
    ws = np.concatenate(ws_parts + [np.zeros(o_off - out_base, np.uint8)])
    wp = np.concatenate(wp_parts)
    meta.update({"ws_bytes": int(ws.size), "wp_bytes": int(wp.size), "out_base": int(out_base)})
    return ws, wp, meta


def column_programs(meta):
    from ignite_xdna.compiler.engine_sequence import DmaPattern
    programs = []
    for c in range(COLS):
        items = []
        for sc in meta["cols"][c]["scenarios"]:
            items.append(("w", sc["w_off"], W_BYTES, sc["n_out"] + sc["n_acc"]))
            for i, a_off in enumerate(sc["a_offs"]):
                items.append(("a", [DmaPattern("ws", a_off + r * A_BYTES, (A_BYTES,), (1,))
                                    for r in range(ROWS)]))
                if i < sc["n_out"]:
                    items.append(("o", DmaPattern("ws", sc["o_offs"][i], (ROWS * O_BYTES,), (1,))))
        programs.append(items)
    return programs


def sequence_from_meta(meta):
    from ignite_xdna.compiler.engine_sequence import SequenceEmitter

    from kernels.bf16_conv import design as eng

    programs = column_programs(meta)

    def body(ws, wp):
        emitter = SequenceEmitter(ws, wp, eng.WS_BYTES, eng.WP_BYTES,
                                  {c: eng.fifo_names(c) for c in range(COLS)})
        emitter.run_column_programs(programs, bd_budget=14)
    return body


def compile_engine(build_dir: Path, seed: int):
    import aie.iron as iron
    from aie.iron.device import NPU1
    from aie.utils.compile.utils import compile_mlir_module

    from kernels.bf16_conv import design as eng

    iron.set_current_device(NPU1())
    plan = build_plan(seed)
    ws, wp, meta = layout_buffers(plan)
    build_dir.mkdir(parents=True, exist_ok=True)
    np.save(build_dir / "ws.npy", ws)
    np.save(build_dir / "wp.npy", wp)
    expected = emulate_plan(plan)
    flat = [np.concatenate(col) if col else np.zeros(0, np.uint8) for col in expected]
    meta["expected_lengths"] = [int(f.size) for f in flat]
    np.save(build_dir / "expected.npy", np.concatenate(flat))
    (build_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    t0 = time.perf_counter()
    program = eng.build_program(iron.get_current_device(), sequence_from_meta(meta))
    module = program.resolve_program()
    work = build_dir / "design.prj"
    # IRON reuses an existing engine_bf16.o; a stale one would silently ship an old kernel.
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(exist_ok=True)
    (build_dir / "design.mlir").write_text(str(module), encoding="utf-8")
    compile_mlir_module(module, insts_path=build_dir / "insts.bin",
                        xclbin_path=build_dir / "design.xclbin",
                        work_dir=work, device=iron.get_current_device())
    dt = time.perf_counter() - t0
    obj = work / "engine_bf16.o"
    manifest = {
        "seed": seed, "compile_seconds": round(dt, 1),
        "kernel_sha256": hashlib.sha256(eng.KERNEL_SOURCE.read_bytes()).hexdigest(),
        "kernel_object_sha256": hashlib.sha256(obj.read_bytes()).hexdigest() if obj.exists() else None,
        "xclbin_sha256": hashlib.sha256((build_dir / "design.xclbin").read_bytes()).hexdigest(),
        "insts_bytes": (build_dir / "insts.bin").stat().st_size,
        "ws_bytes": meta["ws_bytes"], "wp_bytes": meta["wp_bytes"],
        "scenarios": [sc["name"] for sc in meta["cols"][0]["scenarios"]],
        "output_objects_per_column": len(expected[0]),
    }
    (build_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("ENGINE_BF16_16CORE_BUILD " + json.dumps(manifest, sort_keys=True), flush=True)
    return manifest


def run_hardware(build_dir: Path, device_idx: int = 0, iters: int = 3) -> int:
    from ignite_xdna.runtime.driver import XrtSiliconHarness

    meta = json.loads((build_dir / "meta.json").read_text(encoding="utf-8"))
    ws = np.load(build_dir / "ws.npy")
    wp = np.load(build_dir / "wp.npy")
    flat = np.load(build_dir / "expected.npy")
    expected, pos = [], 0
    for n in meta["expected_lengths"]:
        expected.append(flat[pos:pos + n])
        pos += n

    bad = 0
    harness = XrtSiliconHarness(device_idx=device_idx)
    try:
        harness.load_xclbin(str(build_dir / "design.xclbin"))
        bo_instr, n_instr = harness.create_instruction_bo(str(build_dir / "insts.bin"))
        bo_ws = harness.create_host_bo(int(ws.size), 3)
        bo_wp = harness.create_host_bo(int(wp.size), 4)
        bo_wp.write(wp.tobytes(), 0)
        bo_wp.sync(harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
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
            for c in range(COLS):
                exp = expected[c]
                offs = [o for sc in meta["cols"][c]["scenarios"] for o in sc["o_offs"]]
                chunks = [got[o:o + ROWS * O_BYTES] for o in offs]
                hw = np.concatenate(chunks) if chunks else np.zeros(0, np.uint8)
                equal = int(np.sum(hw == exp)) if hw.size == exp.size else 0
                # Per scenario, so a failure names the shape rather than a byte offset.
                per_sc, p = {}, 0
                for sc in meta["cols"][c]["scenarios"]:
                    n = sc["n_out"] * ROWS * O_BYTES
                    if n and hw.size == exp.size:
                        per_sc[sc["name"]] = int(np.sum(hw[p:p + n] == exp[p:p + n]))
                    p += n
                print("ENGINE_BF16_16CORE " + json.dumps({
                    "iter": it, "column": c, "ms": round(dt, 3),
                    "bytes_equal": equal, "bytes": int(exp.size),
                    "bytes_equal_frac": (equal / exp.size) if exp.size else 0.0,
                    "cores": ROWS, "per_scenario_bytes_equal": per_sc,
                }, sort_keys=True), flush=True)
                if equal != exp.size:
                    bad += 1
        bo_ws = bo_wp = bo_instr = None
    finally:
        harness.close()
    if bad:
        print(f"FAIL: {bad} column-iterations did not match the emulator byte for byte")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-dir", default=str(ROOT / "scratch" / "bf16_16core"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--compile", action="store_true", help="build the xclbin only")
    ap.add_argument("--run", action="store_true", help="run an already-built xclbin only")
    args = ap.parse_args()

    build_dir = Path(args.build_dir)
    if not args.run:
        compile_engine(build_dir, args.seed)
    if args.compile:
        return 0
    return run_hardware(build_dir, iters=args.iters)


if __name__ == "__main__":
    raise SystemExit(main())

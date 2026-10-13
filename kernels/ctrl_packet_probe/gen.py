"""Generate the control-packet vs write32 microbenchmark design for one (arm, N).

Tile (0,2) holds WORDS i32 at a fixed data-memory offset BUF. The runtime sequence writes N words
(cycling over the buffer, value = f(i)), then releases lock 0 of tile (0,2) the same way; the tile's
MM2S DMA, gated on that lock, then streams the buffer to DDR for an exact check. Arms:
  w32      N aiex.npu.write32 ops in the instruction stream
  ctrlrB   N / B control packets of B consecutive words (B = 1, 2 or 4), each its own stream packet:
           the shim BD's repeat dimension re-runs a (1 + B)-word transfer, <= 64 per task
  ctrl     all control packets inside ONE stream packet. It timed out at N = 4 (Desktop 2,
           2026-10-13): the tile's control port took the first control packet only, so the lock
           release never landed. Kept to reproduce that, not to time.
Control packets reach the tile through aiecc's --generate-ctrl-pkt-overlay (shim MM2S 0, packet id 27
-> tile (0,2) TileControl); the header layout is mlir-aie's test/npu-xrt/add_one_ctrl_packet_col_overlay.
Lock 1 is reset each dispatch because the BD releases it every run: without that it saturates after
~63 dispatches and the run times out.

python gen.py --arm w32|ctrl|ctrlr1|ctrlr2|ctrlr4 --n N --out DIR
Writes DIR/aie.mlir, DIR/ctrl.bin (the packet stream the host loads into a BO) and DIR/expected.bin.
Build with build.sh, run one with run.py, the whole pre-registered sweep with sweep.py
(scripts/research-lowlevel.sh --npu -- <ironenv python> kernels/ctrl_packet_probe/sweep.py).
"""
import argparse
import struct
from pathlib import Path

WORDS = 1024          # buffer size in i32
BUF = 0x1000          # tile-local data-memory offset of the buffer
LOCK0 = 0x1F000       # MEMORY_MODULE_LOCK0_VALUE (aie-rt xaiemlgbl_params.h)
LOCK1 = 0x1F010       # MEMORY_MODULE_LOCK1_VALUE: reset to 0 each dispatch (the BD releases it every run)


def value(i):
    return (i * 2654435761 + 0x5A5A) & 0xFFFFFFFF


def parity_bit(h):
    return 1 if bin(h).count("1") % 2 == 0 else 0


def header(addr, stream_id=0, op=0, beats=0):
    """Control-packet header as in mlir-aie test/npu-xrt/add_one_ctrl_packet_col_overlay/test.cpp."""
    h = stream_id << 24 | op << 22 | beats << 20 | addr
    return h | parity_bit(h) << 31


def writes(n):
    """(address, value) of the N data writes, then the lock release."""
    out = [(BUF + 4 * (i % WORDS), value(i)) for i in range(n)]
    out.append((LOCK0, 1))
    return out


BEATS = 4             # data words per control packet in arm ctrlr (header beats field = BEATS - 1)


def ctrlr_stream(n):
    """Arm ctrlr: n // BEATS packets of header + BEATS consecutive words, then the lock-release packet."""
    assert n % BEATS == 0 and WORDS % BEATS == 0
    out = []
    for k in range(n // BEATS):
        i0 = k * BEATS
        out.append(header(BUF + 4 * (i0 % WORDS), beats=BEATS - 1))
        out += [value(i0 + j) for j in range(BEATS)]
    out += [header(LOCK0), 1]
    return out


def expected(n):
    buf = [0] * WORDS
    for i in range(n):
        buf[i % WORDS] = value(i)
    return buf


def mlir(arm, n, ctrl_words):
    lines = []
    if arm == "w32":
        for k, (a, v) in enumerate(writes(n)):
            sv = v - (1 << 32) if v >= 1 << 31 else v
            lines.append(f"      %a{k} = arith.constant {a} : i32")
            lines.append(f"      %v{k} = arith.constant {sv} : i32")
            lines.append(f"      aiex.npu.write32(%a{k}, %v{k}) {{column = 0 : i32, row = 2 : i32}} : i32, i32")
    elif arm.startswith("ctrlr"):
        # One stream packet per control packet: the repeat dimension (<= 64 per task) re-runs a
        # (1 + BEATS)-word BD, so each repeat carries its own packet header and TLAST.
        p, npk = 1 + BEATS, n // BEATS
        for c0 in range(0, npk, 64):
            r = min(64, npk - c0)
            lines.append(f"      aiex.npu.dma_memcpy_nd(%ctrl[0, 0, 0, {c0 * p}][{r}, 1, 1, {p}][{p}, 0, 0, 1], "
                         f"packet = <pkt_id = 27, pkt_type = 1>) {{id = 1 : i64, issue_token = true, metadata = @ctrlin0}} : memref<{ctrl_words}xi32>")
            lines.append("      aiex.npu.dma_wait {symbol = @ctrlin0}")
        lines.append(f"      aiex.npu.dma_memcpy_nd(%ctrl[0, 0, 0, {npk * p}][1, 1, 1, 2][0, 0, 0, 1], "
                     f"packet = <pkt_id = 27, pkt_type = 1>) {{id = 1 : i64, issue_token = true, metadata = @ctrlin0}} : memref<{ctrl_words}xi32>")
        lines.append("      aiex.npu.dma_wait {symbol = @ctrlin0}")
    else:
        lines.append(f"      aiex.npu.dma_memcpy_nd(%ctrl[0, 0, 0, 0][1, 1, 1, {ctrl_words}][0, 0, 0, 1], "
                     f"packet = <pkt_id = 27, pkt_type = 1>) {{id = 1 : i64, issue_token = true, metadata = @ctrlin0}} : memref<{ctrl_words}xi32>")
        lines.append("      aiex.npu.dma_wait {symbol = @ctrlin0}")
    body = "\n".join(lines)
    return f"""module {{
  aie.device(npu1) {{
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %buf = aie.buffer(%t02) {{address = {BUF} : i32, sym_name = "buf"}} : memref<{WORDS}xi32>
    %lk = aie.lock(%t02, 0) {{init = 0 : i32, sym_name = "lk"}}
    %lk1 = aie.lock(%t02, 1) {{init = 0 : i32, sym_name = "lk1"}}
    aie.flow(%t02, DMA : 0, %t00, DMA : 1)
    aie.shim_dma_allocation @out0(%t00, S2MM, 1)
    aie.shim_dma_allocation @ctrlin0(%t00, MM2S, 0)
    %mem02 = aie.mem(%t02) {{
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:
      %one = arith.constant 1 : i32
      aie.use_lock(%lk, AcquireGreaterEqual, %one)
      aie.dma_bd(%buf : memref<{WORDS}xi32> offset = 0 len = {WORDS})
      aie.use_lock(%lk1, Release, %one)
      aie.next_bd ^bb1
    ^bb2:
      aie.end
    }}
    aie.runtime_sequence @seq(%out: memref<{WORDS}xi32>, %ctrl: memref<{max(ctrl_words, 2)}xi32>) {{
      %l1a = arith.constant {LOCK1} : i32
      %l1v = arith.constant 0 : i32
      aiex.npu.write32(%l1a, %l1v) {{column = 0 : i32, row = 2 : i32}} : i32, i32
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, {WORDS}][0, 0, 0, 1]) {{id = 0 : i64, issue_token = true, metadata = @out0}} : memref<{WORDS}xi32>
{body}
      aiex.npu.dma_wait {{symbol = @out0}}
    }}
  }}
}}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["w32", "ctrl", "ctrlr1", "ctrlr2", "ctrlr4"], required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    global BEATS
    if a.arm.startswith("ctrlr"):
        BEATS = int(a.arm[5:])
    a.out.mkdir(parents=True, exist_ok=True)
    ws = writes(a.n)
    ctrl = ctrlr_stream(a.n) if a.arm.startswith("ctrlr") else [w for addr, v in ws for w in (header(addr), v)]
    (a.out / "aie.mlir").write_text(mlir(a.arm, a.n, len(ctrl) if a.arm != "w32" else 2), encoding="utf-8")
    (a.out / "ctrl.bin").write_bytes(struct.pack(f"<{len(ctrl)}I", *ctrl))
    (a.out / "expected.bin").write_bytes(struct.pack(f"<{WORDS}I", *expected(a.n)))
    print(a.arm, a.n, "ctrl words", len(ctrl))


if __name__ == "__main__":
    main()

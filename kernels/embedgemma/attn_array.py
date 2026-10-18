#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""16-core check of kernels/embedgemma/attn_local.cc: fused local attention for one head, head_dim 256, on the
whole reachable array (4 columns x 4 cores).

    python kernels/embedgemma/attn_array.py build --out DIR                     (ironenv)
    python kernels/embedgemma/attn_array.py run --out DIR --case NAME ...       (pyxrt, NPU, idle device only)

The core program is attn_probe.py's, unchanged: per 16-row query tile, NKB 32-key blocks, the core skipping any
block wholly outside its band. What changes is the feed. Queries go in 64-row groups, assigned to columns round
robin. Per column:
    Q:   shim -> MemTile (4 packets) -> split, one 16-row packet per core
    K/V: shim -> MemTile -> broadcast to the column's four cores: the group's window, NKB blocks from
         s = clamp(floor32(q0 - W), 0, L - 32 NKB), each block four K^T then four V 4 KiB chunks
    O:   four cores -> MemTile (join) -> shim
So a column streams one window per 64 rows, not one per 16, and its four cores share it. NKB is the most blocks
any group's band touches (34 at W = 512), so each core spends one or two of them skipping.

The check is attn_probe.py's: a NumPy emulation of the kernel's arithmetic on the same blocks, row-scaled
criterion (|device - emulation| <= 2 bf16 ulps of the row's max |emulation|), repeat bit-equal, finite, and a fit
of both to_fixed rounding models. Cases (random q/k/v as attn_probe's):
    dense   L = 256,  W = 512   every key visible (global-like)
    band    L = 512,  W = 48    narrow band: most blocks skipped
    model   L = 1088, W = 512   the single-core probe's size
    long    L = 2048, W = 512   the model's local layer at 2048 tokens
"""
import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import attn_probe as P  # noqa: E402
from attention_oracle import MINIT, fixed_block  # noqa: E402

QWORDS, CHUNK, OUTW, STATE, STACK = P.QWORDS, P.CHUNK, P.OUTW, P.STATE, P.STACK
COLS, ROWS, GROUP = 4, 4, 64
CASES = dict(dense=(256, 512, 1, 1), band=(512, 48, 1, 1), model=(1088, 512, 1, 1), long=(2048, 512, 1, 1),
             doc128=(128, 512, 4, 2), doc512=(512, 512, 4, 2), doc2048=(2048, 512, 4, 2))   # doc*: one layer, 4 heads on 2 KV heads


def layout(L, W, heads=1):
    """-> (nkb, groups): groups[c] = list of work items (head, q0 of the 64-row group, window start s) for column
    c. Items in (head, group) order are split into COLS contiguous runs (sizes differ by at most one), so with four
    heads each column takes one head, and along a run the window start moves by a constant step except where it
    is clamped: the runtime sequence reads each such stretch as ONE strided DMA over a single copy of the head's
    K/V (windows overlap in DDR; nothing is duplicated host side)."""
    assert L % GROUP == 0
    starts = list(range(0, L, GROUP))
    span = [(max(0, g - W), min(L - 1, g + GROUP - 1 + W)) for g in starts]
    nkb = max(hi // 32 - lo // 32 + 1 for lo, hi in span)
    items = []
    for h in range(heads):
        for g, (lo, hi) in zip(starts, span):
            s = int(min(max(lo // 32 * 32, 0), L - 32 * nkb))
            assert s <= lo and hi < s + 32 * nkb, (g, s, lo, hi)   # every visible key of the group is in its window
            items.append((h, g, s))
    base, extra = divmod(len(items), COLS)
    groups, i = [], 0
    for c in range(COLS):
        n = base + (c < extra)
        groups.append(items[i:i + n])
        i += n
    assert all(groups), "every column needs a group"
    return nkb, groups


def runs(items, rep_):
    """Maximal stretches of a column's items with one KV head and a constant window step -> (first item index,
    count, kv head, first window block, step in blocks)."""
    out, i = [], 0
    while i < len(items):
        h, _, s = items[i]
        n, step = 1, 0
        if i + 1 < len(items) and items[i + 1][0] // rep_ == h // rep_:
            step = (items[i + 1][2] - s) // 32
            while (i + n < len(items) and items[i + n][0] // rep_ == h // rep_
                   and items[i + n][2] - items[i + n - 1][2] == 32 * step and n < 64):
                n += 1
        out.append((i, n, h // rep_, s // 32, step if n > 1 else 0))
        i += n
    return out


def harness(L, W, heads=1, kv=1):
    nkb, groups = layout(L, W, heads)
    nb, rep_ = L // 32, heads // kv
    parts = ["module {", "  aie.device(npu1) {",
             f"    func.func private @egattn_score(memref<{QWORDS}xi32>, memref<{CHUNK}xi32>, memref<{STATE}xi32>, i32, i32)",
             f"    func.func private @egattn_softmax(memref<{QWORDS}xi32>, memref<{STATE}xi32>, i32)",
             f"    func.func private @egattn_pv(memref<{CHUNK}xi32>, memref<{STATE}xi32>, i32)",
             f"    func.func private @egattn_finish(memref<{STATE}xi32>, memref<{OUTW}xi32>)"]
    cores = []
    for col in range(COLS):
        parts += [f"    %s{col} = aie.tile({col}, 0)", f"    %m{col} = aie.tile({col}, 1)"]
        parts += [f"    %t{col * ROWS + r} = aie.tile({col}, {r + 2})" for r in range(ROWS)]
        consumers = ", ".join(f"%t{col * ROWS + r}" for r in range(ROWS))
        parts += [
            f"    aie.objectfifo @qL2_{col}(%s{col}, {{%m{col}}}, 1 : i32) : !aie.objectfifo<memref<{ROWS * QWORDS}xi32>>",
            f"    aie.objectfifo @kvL2_{col}(%s{col}, {{%m{col}}}, 2 : i32) : !aie.objectfifo<memref<{CHUNK}xi32>>",
            f"    aie.objectfifo @kvF_{col}(%m{col}, {{{consumers}}}, 2 : i32) : !aie.objectfifo<memref<{CHUNK}xi32>>",
            f"    aie.objectfifo.link [@kvL2_{col}] -> [@kvF_{col}]([] [0])",
            f"    aie.objectfifo @oL2_{col}(%m{col}, {{%s{col}}}, 1 : i32) : !aie.objectfifo<memref<{ROWS * OUTW}xi32>>"]
        for r in range(ROWS):
            i = col * ROWS + r
            parts += [
                f'    %state{i} = aie.buffer(%t{i}) {{sym_name = "egattn_state_{i}"}} : memref<{STATE}xi32>',
                f"    aie.objectfifo @qF_{i}(%m{col}, {{%t{i}}}, 1 : i32) : !aie.objectfifo<memref<{QWORDS}xi32>>",
                f"    aie.objectfifo @oF_{i}(%t{i}, {{%m{col}}}, 1 : i32) : !aie.objectfifo<memref<{OUTW}xi32>>"]
            cores.append(f"""    %core{i} = aie.core(%t{i}) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %forever = arith.constant 4294967295 : index
      scf.for %it = %c0 to %forever step %c1 {{
        %sq = aie.objectfifo.acquire @qF_{i}(Consume, 1) : !aie.objectfifosubview<memref<{QWORDS}xi32>>
        %q = aie.objectfifo.subview.access %sq[0] : !aie.objectfifosubview<memref<{QWORDS}xi32>> -> memref<{QWORDS}xi32>
        %nkb32 = memref.load %q[%c3] : memref<{QWORDS}xi32>
        %nkb = arith.index_cast %nkb32 : i32 to index
        scf.for %kb = %c0 to %nkb step %c1 {{
          %kbi = arith.index_cast %kb : index to i32
          scf.for %c = %c0 to %c4 step %c1 {{
            %ci = arith.index_cast %c : index to i32
            %sk = aie.objectfifo.acquire @kvF_{col}(Consume, 1) : !aie.objectfifosubview<memref<{CHUNK}xi32>>
            %k = aie.objectfifo.subview.access %sk[0] : !aie.objectfifosubview<memref<{CHUNK}xi32>> -> memref<{CHUNK}xi32>
            func.call @egattn_score(%q, %k, %state{i}, %ci, %kbi) : (memref<{QWORDS}xi32>, memref<{CHUNK}xi32>, memref<{STATE}xi32>, i32, i32) -> ()
            aie.objectfifo.release @kvF_{col}(Consume, 1)
          }}
          func.call @egattn_softmax(%q, %state{i}, %kbi) : (memref<{QWORDS}xi32>, memref<{STATE}xi32>, i32) -> ()
          scf.for %c = %c0 to %c4 step %c1 {{
            %ci = arith.index_cast %c : index to i32
            %sv = aie.objectfifo.acquire @kvF_{col}(Consume, 1) : !aie.objectfifosubview<memref<{CHUNK}xi32>>
            %v = aie.objectfifo.subview.access %sv[0] : !aie.objectfifosubview<memref<{CHUNK}xi32>> -> memref<{CHUNK}xi32>
            func.call @egattn_pv(%v, %state{i}, %ci) : (memref<{CHUNK}xi32>, memref<{STATE}xi32>, i32) -> ()
            aie.objectfifo.release @kvF_{col}(Consume, 1)
          }}
        }}
        %so = aie.objectfifo.acquire @oF_{i}(Produce, 1) : !aie.objectfifosubview<memref<{OUTW}xi32>>
        %o = aie.objectfifo.subview.access %so[0] : !aie.objectfifosubview<memref<{OUTW}xi32>> -> memref<{OUTW}xi32>
        func.call @egattn_finish(%state{i}, %o) : (memref<{STATE}xi32>, memref<{OUTW}xi32>) -> ()
        aie.objectfifo.release @oF_{i}(Produce, 1)
        aie.objectfifo.release @qF_{i}(Consume, 1)
      }}
      aie.end
    }} {{link_with = "kernel.o", stack_size = {STACK} : i32}}""")
        parts += [
            f"    aie.objectfifo.link [@qL2_{col}] -> [" + ", ".join(f"@qF_{col * ROWS + r}" for r in range(ROWS)) +
            "]([] [" + ", ".join(str(r * QWORDS) for r in range(ROWS)) + "])",
            "    aie.objectfifo.link [" + ", ".join(f"@oF_{col * ROWS + r}" for r in range(ROWS)) +
            f"] -> [@oL2_{col}]([" + ", ".join(str(r * OUTW) for r in range(ROWS)) + "] [])"]
    parts.extend(cores)
    parts.append("    aie.runtime_sequence(%qa: memref<?xi32>, %kva: memref<?xi32>, %oa: memref<?xi32>) {")
    qoff = 0
    blk = 8 * CHUNK   # words per 32-key block (four K^T then four V chunks)
    for col in range(COLS):
        n = len(groups[col])
        parts += [
            f"      aiex.npu.dma_memcpy_nd(%qa[0, 0, 0, {qoff}][1, 1, {n}, {ROWS * QWORDS}][0, 0, {ROWS * QWORDS}, 1]) {{id = 0 : i64, metadata = @qL2_{col}}} : memref<?xi32>",
            f"      aiex.npu.dma_memcpy_nd(%oa[0, 0, 0, {qoff // QWORDS * OUTW}][1, 1, {n}, {ROWS * OUTW}][0, 0, {ROWS * OUTW}, 1]) {{id = 1 : i64, metadata = @oL2_{col}, issue_token = true}} : memref<?xi32>"]
        rr = runs(groups[col], rep_)
        assert len(rr) <= 13, f"column {col}: {len(rr)} window runs exceed the shim's BDs"
        for j, (_, cnt, kvh, b0, step) in enumerate(rr):
            parts.append(
                f"      aiex.npu.dma_memcpy_nd(%kva[0, 0, 0, {(kvh * nb + b0) * blk}][{cnt}, 1, {nkb * 8}, {CHUNK}][{step * blk}, 0, {CHUNK}, 1]) {{id = {2 + j} : i64, metadata = @kvL2_{col}}} : memref<?xi32>")
        qoff += n * ROWS * QWORDS
    parts += [f"      aiex.npu.dma_wait {{symbol = @oL2_{col}}}" for col in range(COLS)]
    return "\n".join(parts + ["    }", "  }", "}"]) + "\n"


def audit(out):
    """Every core's ELF: program within 16 KiB, the same kernel linked."""
    cores = []
    for elf in sorted((out / "prj").rglob("*.elf")):
        col, row = map(int, re.search(r"core_(\d+)_(\d+)\.elf$", elf.name).groups())
        text = int(P.command([P.PEANO / "bin/llvm-size.exe", elf], out).splitlines()[-1].split()[0])
        if not 0 < text <= 16384:
            raise ValueError(f"core {col},{row}: program {text} B")
        cores.append(dict(core=[col, row], text=text))
    if len(cores) != COLS * ROWS:
        raise ValueError(f"{len(cores)} core ELFs, expected {COLS * ROWS}")
    return cores


def build(out: Path, variant="full", cases=None):
    meta = P.compile_kernel(out, variant)
    for name, (L, W, H, KV) in CASES.items():
        if cases and name not in cases:
            continue
        d = out / name
        d.mkdir(exist_ok=True)
        (d / "harness.mlir").write_text(harness(L, W, H, KV), encoding="utf-8")
        (d / "kernel.o").write_bytes((out / "kernel.o").read_bytes())
        P.command([P.IRON / "Scripts/aiecc.exe", d / "harness.mlir", f"--peano={P.PEANO}", "--get-npu-insts",
                   "--npu-insts-name=insts.bin", "--get-xclbin", "--xclbin-name=final.xclbin", f"--tmpdir={d / 'prj'}"], d)
        nkb, groups = layout(L, W, H)
        meta["cases"][name] = dict(L=L, W=W, heads=H, kv=KV, nkb=nkb, transport="strided", groups=[len(g) for g in groups], cores=audit(d),
                                   xclbin_sha256=P.sha(d / "final.xclbin"), insts_sha256=P.sha(d / "insts.bin"))
        print(f"BUILD {name}: L={L} W={W} NKB={nkb} groups per column {[len(g) for g in groups]} ok", flush=True)
    (out / "build.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return 0


def pack(q, k, v, W):
    """q [L, H, 256], k/v [L, KV, 256] (bf16 values) -> (Q packets column-major as the runtime sequence reads them,
    K/V blocks [KV, L/32, 8, CHUNK] (one copy; the DMA reads each window out of it), order): order[i] = (head, q0)
    of output tile i."""
    L, H = q.shape[:2]
    KV = k.shape[1]
    nkb, groups = layout(L, W, H)
    nb = L // 32
    blocks = np.zeros((KV, nb, 8, CHUNK), np.uint32)
    raw = blocks.view(np.uint16).reshape(KV, nb, 8, 2 * CHUNK)
    for c in range(4):   # chunk c of block b: K^T [64 dims, 32 keys] and V [32 keys, 64 dims], each as 8x4 tiles
        kt = P.bits(k[:, :, 64 * c:64 * c + 64]).reshape(nb, 32, KV, 64).transpose(2, 0, 3, 1)   # kv, b, dim, key
        raw[:, :, c] = kt.reshape(KV, nb, 8, 8, 8, 4).transpose(0, 1, 2, 4, 3, 5).reshape(KV, nb, -1)
        vb = P.bits(v[:, :, 64 * c:64 * c + 64]).reshape(nb, 32, KV, 64).transpose(2, 0, 1, 3)   # kv, b, key, dim
        raw[:, :, 4 + c] = vb.reshape(KV, nb, 4, 8, 16, 4).transpose(0, 1, 2, 4, 3, 5).reshape(KV, nb, -1)
    # Q tile: [16 rows, 256] as 4x8 tiles, as attn_probe.tiled(bits(q[q0:q0+16]), 4, 8)
    qt = P.bits(q).transpose(1, 0, 2).reshape(H, L // 16, 4, 4, 32, 8).transpose(0, 1, 2, 4, 3, 5).reshape(H, L // 16, -1)
    items = [it for col in range(COLS) for it in groups[col]]
    qp = np.zeros((len(items), ROWS, QWORDS), np.uint32)
    order = []
    for i, (h, g, s) in enumerate(items):
        for r in range(ROWS):
            q0 = g + 16 * r
            qp[i, r, :4] = [q0, s, W, nkb]
            order.append((h, q0))
        qp[i, :, 16:].view(np.uint16)[:] = qt[h, g // 16:g // 16 + ROWS]
    return qp.reshape(-1, QWORDS), blocks, order


def emulate(q, k, v, W, mode="round"):
    """attn_probe.emulate on the array's windows: each 16-row tile walks its group's NKB blocks. q [L, H, 256],
    k/v [L, KV, 256] -> [L, H, 256]."""
    L, H = q.shape[:2]
    nkb, groups = layout(L, W, H)
    out = np.zeros((L, H, 256), np.float32)
    for col in range(COLS):
        for h, g, s in groups[col]:
            qh, kh, vh = q[:, h], k[:, h // (H // k.shape[1])], v[:, h // (H // k.shape[1])]
            for r in range(ROWS):
                rows = np.arange(g + 16 * r, g + 16 * r + 16)
                m = np.full(16, MINIT, np.int64)
                l = np.zeros(16, np.float32)
                O = np.zeros((16, 256), np.float32)
                for b in range(nkb):
                    cols = np.arange(s + 32 * b, s + 32 * b + 32)
                    vis = np.abs(rows[:, None] - cols[None, :]) <= W
                    if not vis.any():
                        continue
                    S = (qh[rows].astype(np.float64) @ kh[cols].T.astype(np.float64)).astype(np.float32)
                    m, l, O = fixed_block(S, vis, m, l, O, vh[cols], mode)
                out[rows, h] = P.bf16(O * P.reciprocal16(l)[:, None])
    return out


def case_data(name):
    L, W, H, KV = CASES[name]
    rng = np.random.default_rng({"dense": 3, "band": 5, "model": 7, "long": 11}.get(name, L))
    q = P.bf16(rng.standard_normal((L, H, 256)).astype(np.float32) * 0.35)
    k = P.bf16(rng.standard_normal((L, KV, 256)).astype(np.float32) * 0.35)
    v = P.bf16(rng.standard_normal((L, KV, 256)).astype(np.float32))
    return q, k, v, L, W


class Packer:
    """attn_pack.c (egattn_pack.dll, ``native.py build --attn``): pack() and the output reorder, natively."""

    def __init__(self, dll):
        import ctypes
        self.lib = ctypes.CDLL(str(dll))
        vp, i64, i32 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int32
        self.lib.egattn_pack_q.argtypes = [vp, i64, i64, i32, i32, vp, i64, vp]
        self.lib.egattn_pack_kv.argtypes = [vp, vp, i64, i64, vp]
        self.lib.egattn_unpack.argtypes = [vp, vp, i64, i64, vp]
        for f in (self.lib.egattn_pack_q, self.lib.egattn_pack_kv, self.lib.egattn_unpack):
            f.restype = None

    @staticmethod
    def items(L, W, H):
        nkb, groups = layout(L, W, H)
        return nkb, np.ascontiguousarray([it for col in range(COLS) for it in groups[col]], np.int32)

    def pack_into(self, q, k, v, W, qdst, kvdst):
        """q [L, H, 256], k/v [L, KV, 256] fp32 -> Q packets into ``qdst`` and K/V blocks into ``kvdst`` (uint32
        views, e.g. mapped BOs); returns the items array for unpack."""
        L, H = q.shape[:2]
        KV = k.shape[1]
        q, k, v = (np.ascontiguousarray(x, np.float32) for x in (q, k, v))
        nkb, items = self.items(L, W, H)
        if qdst.size < len(items) * ROWS * QWORDS or kvdst.size < KV * (L // 32) * 8 * CHUNK:
            raise ValueError("destination too small")
        self.lib.egattn_pack_q(q.ctypes.data, L, H, W, nkb, items.ctypes.data, len(items), qdst.ctypes.data)
        self.lib.egattn_pack_kv(k.ctypes.data, v.ctypes.data, L, KV, kvdst.ctypes.data)
        return items

    def unpack(self, raw, items, L, H):
        out = np.empty((L, H, 256), np.float32)
        self.lib.egattn_unpack(raw.ctypes.data, items.ctypes.data, len(items), H, out.ctypes.data)
        return out


class Session:
    """One hardware context on one case's xclbin; any case's instruction stream runs on it, since the core images and
    configuration of every case are byte-identical (only main.pdi differs, and the probe's --xclbin-from checks the
    cross-use on device). run(q, k, v, W) packs, dispatches and returns [L, H, 256] fp32 (bf16 values)."""

    def __init__(self, out: Path, xclbin_case, pack_dll=None):
        sys.path.insert(0, "C:/Xilinx/XRT/xrt_sdk/xrt/python")
        import pyxrt
        self.pyxrt, self.out = pyxrt, Path(out)
        self.meta = json.loads((self.out / "build.json").read_text(encoding="utf-8"))
        if self.meta["source_sha256"] != P.sha(HERE / "attn_local.cc") or self.meta["kernel_sha256"] != P.sha(self.out / "kernel.o"):
            raise RuntimeError("build is stale: kernel source or object changed since build")
        for name, c in self.meta["cases"].items():
            d = self.out / name
            if P.sha(d / "final.xclbin") != c["xclbin_sha256"] or P.sha(d / "insts.bin") != c["insts_sha256"]:
                raise RuntimeError(f"{name}: build artefacts changed")
        self.dev = pyxrt.device(0)
        self.xb = pyxrt.xclbin(str(self.out / xclbin_case / "final.xclbin"))
        self.dev.register_xclbin(self.xb)
        self.ctx = pyxrt.hw_context(self.dev, self.xb.get_uuid())
        self.kern = pyxrt.kernel(self.ctx, self.xb.get_kernels()[0].get_name())
        self.to = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self.fr = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        self.shapes = {}   # (L, W, H, KV) -> (case, instruction BO, its bytes, q BO, kv BO, out BO)
        self.times = []
        self.t = dict(attn_pack=0.0, attn_write=0.0, attn_call=0.0, attn_read=0.0, attn_unpack=0.0)
        self.packer = Packer(pack_dll) if pack_dll else None

    def _shape(self, L, W, H, KV):
        key = (L, W, H, KV)
        if key not in self.shapes:
            name = next((n for n, c in self.meta["cases"].items()
                         if (c["L"], c["W"], c.get("heads", 1), c.get("kv", 1)) == key), None)
            if name is None:
                raise KeyError(f"no build for L={L} W={W} heads={H} kv={KV}")
            pyxrt, kern, dev = self.pyxrt, self.kern, self.dev
            words = np.frombuffer((self.out / name / "insts.bin").read_bytes(), dtype=np.uint32)
            ib = pyxrt.bo(dev, words.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
            ib.write(words.tobytes(), 0)
            ib.sync(self.to)
            nkb, groups = layout(L, W, H)
            items = sum(len(g) for g in groups)
            bq = pyxrt.bo(dev, items * ROWS * QWORDS * 4, pyxrt.bo.host_only, kern.group_id(3))
            bkv = pyxrt.bo(dev, KV * (L // 32) * 8 * CHUNK * 4, pyxrt.bo.host_only, kern.group_id(4))
            bo = pyxrt.bo(dev, items * ROWS * OUTW * 4, pyxrt.bo.host_only, kern.group_id(5))
            self.shapes[key] = (name, ib, words.nbytes, bq, bkv, bo)
        return self.shapes[key]

    def dispatch(self, qp, kv, shape, poison=True):
        """Raw dispatch of packed inputs -> (output words as uint16 [tiles, 16, 256], ms)."""
        name, ib, nbytes, bq, bkv, bo = self._shape(*shape)
        t = time.perf_counter()
        np.frombuffer(bq.map(), np.uint32)[:qp.size] = qp.reshape(-1)   # straight into the mapped BOs
        bq.sync(self.to)
        np.frombuffer(bkv.map(), np.uint32)[:kv.size] = kv.reshape(-1)
        bkv.sync(self.to)
        self.t["attn_write"] += time.perf_counter() - t
        n = qp.shape[0] * OUTW
        if poison:
            bo.write(np.full(n, 0x7FC07FC0, np.uint32).tobytes(), 0)   # bf16 NaN poison
            bo.sync(self.to)
        t0 = time.perf_counter()
        st = self.kern(3, ib, nbytes, bq, bkv, bo).wait(10000)
        ms = (time.perf_counter() - t0) * 1000
        if st != self.pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"{name}: dispatch {st}")
        self.t["attn_call"] += ms / 1000
        t = time.perf_counter()
        bo.sync(self.fr)
        raw = np.frombuffer(bo.map(), np.uint16)[:2 * n].reshape(-1, 16, 256).copy()
        self.t["attn_read"] += time.perf_counter() - t
        return raw, ms

    def run(self, q, k, v, W):
        """q [L, H, 256], k/v [L, KV, 256] -> [L, H, 256] fp32 (each value a bf16)."""
        L, H = q.shape[:2]
        if self.packer:   # native: pack straight into the mapped BOs, unpack straight out of the mapped output
            shape = (L, W, H, k.shape[1])
            name, ib, nbytes, bq, bkv, bo = self._shape(*shape)
            t = time.perf_counter()
            items = self.packer.pack_into(q, k, v, W, np.frombuffer(bq.map(), np.uint32), np.frombuffer(bkv.map(), np.uint32))
            self.t["attn_pack"] += time.perf_counter() - t
            t = time.perf_counter()
            bq.sync(self.to)
            bkv.sync(self.to)
            self.t["attn_write"] += time.perf_counter() - t
            t0 = time.perf_counter()
            st = self.kern(3, ib, nbytes, bq, bkv, bo).wait(10000)
            ms = (time.perf_counter() - t0) * 1000
            if st != self.pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
                raise RuntimeError(f"{name}: dispatch {st}")
            self.times.append(ms)
            self.t["attn_call"] += ms / 1000
            t = time.perf_counter()
            bo.sync(self.fr)
            self.t["attn_read"] += time.perf_counter() - t
            t = time.perf_counter()
            out = self.packer.unpack(np.frombuffer(bo.map(), np.uint16), items, L, H)
            self.t["attn_unpack"] += time.perf_counter() - t
            return out
        t = time.perf_counter()
        qp, kv, order = pack(q, k, v, W)
        self.t["attn_pack"] += time.perf_counter() - t
        raw, ms = self.dispatch(qp, kv, (L, W, H, k.shape[1]), poison=False)
        self.times.append(ms)
        t = time.perf_counter()
        rows = np.empty((L, H, 256), np.uint16)
        for i, (h, q0) in enumerate(order):
            rows[q0:q0 + 16, h] = raw[i]
        out = (rows.astype(np.uint32) << 16).view(np.float32)
        self.t["attn_unpack"] += time.perf_counter() - t
        return out

    def close(self):
        self.shapes.clear()
        gc.collect()
        del self.kern
        gc.collect()
        del self.ctx
        gc.collect()
        del self.xb, self.dev
        gc.collect()


def run(out: Path, names, reps, save=None, xclbin_from=None):
    if not P.idle():
        print("NPU not idle: refusing (a foreign context is contention, not a finding)")
        return 3
    ok, saved, sess = True, {}, None
    for name in names:
        c = json.loads((out / "build.json").read_text(encoding="utf-8"))["cases"][name]
        if sess is None or xclbin_from is None:
            if sess is not None:
                sess.close()
            sess = Session(out, xclbin_from or name)
            if xclbin_from:
                print(f"every case runs on {xclbin_from}'s xclbin (cross-use check)", flush=True)
        q, k, v, L, W = case_data(name)
        H = q.shape[1]
        qp, kv, order = pack(q, k, v, W)
        refs = {mode: emulate(q, k, v, W, mode) for mode in ("round", "floor")}
        results, times = [], []
        try:
            for i in range(1 + reps):
                raw, ms = sess.dispatch(qp, kv, (L, W, H, k.shape[1]))
                times.append(ms)
                dev_rows = np.empty((L, H, 256), np.uint16)
                for t, (h, q0) in enumerate(order):
                    dev_rows[q0:q0 + 16, h] = raw[t]
                results.append(dev_rows)
        except RuntimeError as e:
            ok = False
            print(f"CASE {name}: {e}; stopping without retry", flush=True)
            break
        got = (results[0].astype(np.uint32) << 16).view(np.float32)
        saved[name] = results[0]

        def worst_of(ref):
            rowmax = np.abs(ref).max(axis=-1, keepdims=True)
            tol = 2 * np.exp2(np.floor(np.log2(np.maximum(rowmax, 1e-30))) - 7)
            return float((np.abs(got - ref) / tol).max())
        fits = {mode: worst_of(r) for mode, r in refs.items()}
        mode = min(fits, key=fits.get)
        worst, ref = fits[mode], refs[mode]
        same = all(np.array_equal(results[0], r) for r in results[1:])
        finite = bool(np.isfinite(got).all())
        good = finite and same and worst <= 1.0
        ok &= good
        rel = float(np.abs(got - ref).max() / np.abs(ref).max())
        med = float(np.median(times[1:]) if reps else times[0])
        mb = (qp.nbytes + kv.nbytes + qp.shape[0] * OUTW * 4) / 1e6
        print(f"  to_fixed model fit: " + ", ".join(f"{k} {v:.3f}" for k, v in fits.items()) + f" -> {mode}", flush=True)
        print(f"CASE {name}: L={L} W={W} heads={H} NKB={c['nkb']} items/col {c['groups']}: worst |diff| / (2 row ulps) "
              f"{worst:.3f}, rel err {rel:.2e}, finite {finite}, repeat bit-equal {same}: "
              f"{'PASS' if good else 'FAIL'}; dispatch median {med:.3f} ms over {reps} "
              f"(min {min(times[1:] or times):.3f}); {mb:.1f} MB moved, {mb / med:.2f} GB/s", flush=True)
    if sess is not None:
        sess.close()
    if save:
        np.savez_compressed(save, **saved)
        print(f"saved device outputs: {save}", flush=True)
    print(f"idle after: {P.idle()}", flush=True)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "run", "bench", "check-native"])
    ap.add_argument("--pack-dll", type=Path, help="bench/check-native: egattn_pack.dll (native.py build --attn)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--case", nargs="+", default=list(CASES))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--save", type=Path, help="npz of the device outputs per case")
    ap.add_argument("--xclbin-from", help="run: every case on this case's xclbin (one context, per-case instructions)")
    a = ap.parse_args()
    if a.cmd == "check-native":   # offline: the DLL's words == pack()'s, and unpack == the NumPy reorder
        pk, ok = Packer(a.pack_dll), True
        for name in a.case:
            q, k, v, L, W = case_data(name)
            q, k, v = (x + np.float32(1e-3) * np.random.default_rng(L).standard_normal(x.shape).astype(np.float32)
                       for x in (q, k, v))   # not already bf16: exercises the rounding
            qp, kvb, order = pack(q, k, v, W)
            qd, kd = np.zeros(qp.size + 7, np.uint32), np.zeros(kvb.size + 7, np.uint32)
            items = pk.pack_into(q, k, v, W, qd, kd)
            raw = np.random.default_rng(1).integers(0, 1 << 16, (len(order), 16, 256)).astype(np.uint16)
            want = np.empty((L, q.shape[1], 256), np.uint16)
            for i, (h, q0) in enumerate(order):
                want[q0:q0 + 16, h] = raw[i]
            good = (np.array_equal(qd[:qp.size], qp.reshape(-1)) and np.array_equal(kd[:kvb.size], kvb.reshape(-1))
                    and not qd[qp.size:].any() and not kd[kvb.size:].any()
                    and np.array_equal(pk.unpack(raw, items, L, q.shape[1]).view(np.uint32) >> 16, want))
            ok &= good
            print(f"CHECK {name}: {'PASS' if good else 'FAIL'}", flush=True)
        return 0 if ok else 1
    if a.cmd == "bench":   # Session.run end to end per case: host phases (pack, write, call, read, unpack)
        if not P.idle():
            print("NPU not idle: refusing (a foreign context is contention, not a finding)")
            return 3
        sess = Session(a.out.resolve(), a.xclbin_from or "doc2048", a.pack_dll)
        try:
            for name in a.case:
                q, k, v, L, W = case_data(name)
                sess.run(q, k, v, W)   # warm-up: BOs and instructions
                sess.t = dict.fromkeys(sess.t, 0.0)
                t0 = time.perf_counter()
                for _ in range(a.reps):
                    sess.run(q, k, v, W)
                wall = (time.perf_counter() - t0) / a.reps * 1000
                print(f"BENCH {name}: Session.run {wall:.3f} ms/call over {a.reps}; " + ", ".join(
                    f"{k} {v / a.reps * 1000:.3f}" for k, v in sess.t.items()), flush=True)
        finally:
            sess.close()
        print(f"idle after: {P.idle()}", flush=True)
        return 0
    if a.cmd == "build":
        return build(a.out.resolve(), "full", a.case if a.case != list(CASES) else None)
    return run(a.out.resolve(), a.case, a.reps, a.save, a.xclbin_from)


if __name__ == "__main__":
    sys.exit(main())

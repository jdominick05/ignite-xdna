# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native host helpers for EmbeddingGemma 2 on the NPU: build and load ``round.c`` (fp32 -> bf16, AVX2 + OpenMP).

``build(out_dir)`` compiles ``embedgemma_round.dll`` with MSVC (found through vswhere, as Laya's pack.c is built);
``Rounder(dll)`` wraps it. ``round_into(x, dst)`` writes bf16 bits of fp32 ``x`` [rows, cols] into the uint16 view
``dst`` [>= rows, >= cols] (a row-strided view, e.g. a mapped BO), bit-identical to ``model.bf16_bits`` for finite
values. ``python native.py check --dll DLL`` verifies that on random and edge-case data.
"""
import argparse
import ctypes
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def build(out: Path, source="round.c", name="embedgemma_round.dll") -> Path:
    """Compile ``source`` (round.c, or attn_pack.c -> egattn_pack.dll) with MSVC /O2 /arch:AVX2 /openmp."""
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    vswhere = Path("C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe")
    vs = subprocess.run([str(vswhere), "-products", "*", "-latest", "-requires",
                         "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
                        capture_output=True, text=True, check=True).stdout.strip()
    vcvars = Path(vs) / "VC/Auxiliary/Build/vcvars64.bat"
    cmd = out / f"build_{Path(source).stem}.cmd"
    cmd.write_text(f'@echo off\ncall "{vcvars}" >nul && cl /nologo /LD /O2 /arch:AVX2 /openmp '
                   f'"{HERE / source}" /Fe:{name}\n', encoding="utf-8")
    r = subprocess.run(["cmd.exe", "/d", "/c", str(cmd)], cwd=out, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stdout + r.stderr)
    return out / name


class Rounder:
    def __init__(self, dll: Path):
        self.lib = ctypes.CDLL(str(dll))
        f = self.lib.round_bf16_rows
        f.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
        f.restype = None
        self.f = f

    def round_into(self, x: np.ndarray, dst: np.ndarray) -> None:
        if x.dtype != np.float32 or x.ndim != 2 or x.strides[1] != 4:
            x = np.ascontiguousarray(x, dtype=np.float32)
        if dst.dtype != np.uint16 or dst.ndim != 2 or dst.strides[1] != 2 or not dst.flags.writeable:
            raise ValueError("dst must be a writeable row-strided uint16 [rows, cols] view")
        rows, cols = x.shape
        if dst.shape[0] < rows or dst.shape[1] < cols:
            raise ValueError(f"dst {dst.shape} smaller than x {x.shape}")
        self.f(x.ctypes.data, x.strides[0] // 4, dst.ctypes.data, dst.strides[0] // 2, rows, cols)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--attn", action="store_true", help="build attn_pack.c -> egattn_pack.dll instead")
    b.add_argument("--host-geglu", action="store_true", help="build host_geglu.c -> egemm_host_geglu.dll instead")
    b.add_argument("--rowops", action="store_true", help="build rowops.c -> eg_rowops.dll instead")
    b.add_argument("--qkv", action="store_true", help="build qkv_norm_rope.c -> eg_qkv_norm_rope.dll instead")
    sub.add_parser("check").add_argument("--dll", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "build":
        print(build(a.out, "rowops.c", "eg_rowops.dll") if a.rowops else
              build(a.out, "qkv_norm_rope.c", "eg_qkv_norm_rope.dll") if a.qkv else
              build(a.out, "attn_pack.c", "egattn_pack.dll") if a.attn else
              build(a.out, "host_geglu.c", "egemm_host_geglu.dll") if a.host_geglu else build(a.out))
        return 0
    sys.path.insert(0, str(HERE))
    from model import bf16_bits
    r = Rounder(a.dll)
    rng = np.random.default_rng(11)
    edge = np.array([0.0, -0.0, 1.0, -1.0, 1e-40, -1e-40, 3.4e38, -3.4e38, 1.00390625, 1.01171875, 65504.0],
                    np.float32)
    ties = (np.arange(1 << 16, dtype=np.uint32) << 16 | 0x8000).view(np.float32)   # every exact tie
    ties = ties[np.isfinite(ties)]
    ok = True
    for name, x in [("random", rng.standard_normal((517, 3001)).astype(np.float32) * 10),
                    ("bits", rng.integers(0, 0x7F7FFFFF, (300, 1024), dtype=np.uint32).view(np.float32)),
                    ("edge", edge[None, :]), ("ties", ties[None, :])]:
        dst = np.zeros((x.shape[0] + 3, x.shape[1] + 37), np.uint16)
        r.round_into(x, dst)
        good = np.array_equal(dst[:x.shape[0], :x.shape[1]], bf16_bits(x)) and not dst[x.shape[0]:].any() \
            and not dst[:, x.shape[1]:].any()
        ok &= good
        print(f"CHECK {name} {x.shape}: {'PASS' if good else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

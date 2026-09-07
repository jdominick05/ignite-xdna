#!/usr/bin/env python3
"""Verify (and, once patched, prove) conv2dk3_ui8_vector at widths other than 32.

See design.py for why this exists. Weight/scale conventions copied from
kernels/bottleneck_sweep/sweep.py's build_golden -- same INP_SCALE/WEIGHT_SCALE=0.5,
same DataShaper layouts -- applied to just the 3x3 stage in isolation (64 in, 64 out,
uint8 activations in and out, matching conv2's role inside the full bottleneck).

USAGE (mlir-aie ironenv)
    python3 test_width.py --shapes 32x32,32x64,32x96
"""

import argparse
import sys

import numpy as np
import torch
import torch.nn as nn

import aie.iron as iron
from aie.utils.ml import DataShaper

from design import CHANNELS, conv3x3_only

torch.use_deterministic_algorithms(True)
torch.manual_seed(0)

IN_LAYOUT = ("YCXC8", "CYX")
WTS_LAYOUT = ("OIYXI8O8", "OIYX")
OUT_REORDER = ("CDYX", "YCXD")

# Deliberately NOT the bottleneck-wide INP_SCALE/WEIGHT_SCALE=0.5 convention -- at full
# channel width (64 in, 64 out, 9 taps) with weights in [50,100) and inputs in [0,255],
# the raw accumulator saturates the uint8 output at essentially every position, which
# would make this test pass for almost any indexing bug (constant 255 vs constant 255).
# Small weights + a matching right-shift keep the output non-degenerate, so a stride bug
# actually shows up as a mismatch instead of shared saturation.
WEIGHT_LO, WEIGHT_HI = 1, 4
INPUT_LO, INPUT_HI = 0, 16
SCALE = 6  # matches the `scale` kwarg passed to conv3x3_only below


def build_golden(w2):
    class Conv3x3Int8(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv2 = nn.Conv2d(
                CHANNELS, CHANNELS, kernel_size=3, padding=1, padding_mode="zeros", bias=False
            )

        def forward(self, r1):
            # r1/w2 hold literal int8-domain codes (not dequantized), so conv2(r1) IS
            # the raw int32 accumulator -- match the AIE kernel's own
            # sum_srs = (sum + (1<<(scale-1))) >> scale exactly (round-half-up, not
            # torch.round's round-half-to-even) so a tie at .5 can't read as a mismatch.
            c2 = torch.relu(self.conv2(r1))
            rounded = torch.floor(c2 / (2**SCALE) + 0.5)
            return torch.clamp(rounded, 0, 255)

    m = Conv3x3Int8()
    m.conv2.weight.data.copy_(w2)
    m.eval()
    return m


def make_buffers(h, w):
    r1 = torch.randint(INPUT_LO, INPUT_HI, (1, CHANNELS, h, w)).type(torch.FloatTensor)
    w2 = torch.randint(WEIGHT_LO, WEIGHT_HI, (CHANNELS, CHANNELS, 3, 3)).type(torch.FloatTensor)

    ds = DataShaper()
    ifm = ds.reorder_mat(r1.squeeze().data.numpy().astype(np.uint8), *IN_LAYOUT)
    wts = ds.reorder_mat(w2.data.numpy().astype(np.int8), *WTS_LAYOUT)
    golden = build_golden(w2)(r1).detach().numpy()
    return ifm, wts, golden, ds


def check(out_tensor, ds, golden, h, w):
    # golden is the raw uint8-domain r2 value (no final *INP_SCALE dequant --
    # that only applies to the whole bottleneck's LAST stage, not this mid-pipeline
    # activation), so compare the raw AIE output against it directly.
    raw = out_tensor.numpy().view(np.uint8).astype(np.float32)
    temp = raw.reshape((h, CHANNELS // 8, w, 8))
    temp = ds.reorder_mat(temp, *OUT_REORDER).reshape((CHANNELS, h, w))
    return bool(np.allclose(temp[None, ...], golden, rtol=0, atol=1.0))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--shapes", default="32x32,32x64,32x96")
    args = p.parse_args()

    shapes = []
    for tok in args.shapes.split(","):
        hs, ws = tok.lower().split("x")
        shapes.append((int(hs), int(ws)))

    print("conv2dk3_ui8_vector width isolation test (64 -> 64, uint8 act, no skip)")
    print(f"{'HxW':>9}  {'result':>8}  max |diff|")
    print("-" * 40)

    any_fail = False
    for h, w in shapes:
        ifm, wts, golden, ds = make_buffers(h, w)
        a = iron.tensor(ifm, dtype=np.uint8)
        b = iron.tensor(wts, dtype=np.int8)
        c = iron.zeros(h * w * CHANNELS, dtype=np.uint8)
        try:
            conv3x3_only(a, b, c, tensor_w=w, tensor_h=h, scale=SCALE)
        except Exception as e:
            print(f"{h}x{w:>7}  COMPILE-FAIL  {type(e).__name__}: {e}")
            any_fail = True
            continue

        raw = c.numpy().view(np.uint8).astype(np.float32)
        temp = raw.reshape((h, CHANNELS // 8, w, 8))
        temp = ds.reorder_mat(temp, *OUT_REORDER).reshape((CHANNELS, h, w))
        maxdiff = float(np.max(np.abs(temp[None, ...] - golden)))
        ok = check(c, ds, golden, h, w)
        print(f"{h}x{w:>7}  {'PASS' if ok else 'FAIL':>8}  {maxdiff:.3f}")
        if not ok:
            any_fail = True

    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()

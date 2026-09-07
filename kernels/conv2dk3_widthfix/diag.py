import sys
import numpy as np
import torch
import aie.iron as iron
from aie.utils.ml import DataShaper
from design import CHANNELS, conv3x3_only

def run_diag(w=36, h=4):
    # Center tap weight = 64, scale = 6 -> (val * 64 + 32) >> 6 = val
    scale = 6
    w2 = torch.zeros((CHANNELS, CHANNELS, 3, 3), dtype=torch.float32)
    for c in range(CHANNELS):
        w2[c, c, 1, 1] = 64.0

    r1 = torch.zeros((1, CHANNELS, h, w), dtype=torch.float32)
    for c in range(CHANNELS):
        cb = c // 8
        for x in range(w):
            val = cb * 30 + (x % 30) + 1
            r1[0, c, :, x] = val

    ds = DataShaper()
    ifm = ds.reorder_mat(r1.squeeze().data.numpy().astype(np.uint8), "YCXC8", "CYX")
    wts = ds.reorder_mat(w2.data.numpy().astype(np.int8), "OIYXI8O8", "OIYX")

    a = iron.tensor(ifm, dtype=np.uint8)
    b = iron.tensor(wts, dtype=np.int8)
    c = iron.zeros(h * w * CHANNELS, dtype=np.uint8)

    conv3x3_only(a, b, c, tensor_w=w, tensor_h=h, scale=scale)

    raw = c.numpy().view(np.uint8).astype(np.float32)
    temp = raw.reshape((h, CHANNELS // 8, w, 8))
    out = ds.reorder_mat(temp, "CDYX", "YCXD").reshape((CHANNELS, h, w))

    # Examine row 1
    print(f"=== Diagnosis for width={w}, height={h}, row 1 oc-block 0 ===")
    y = 1
    oc_b = 0
    c = oc_b * 8
    got = out[c, y, :].astype(int)
    exp = r1[0, c, y, :].numpy().astype(int)
    print("--- Check within oc-block 0 (channels 0..7) ---")
    for c in range(8):
        m = np.all(out[c, y, :] == r1[0, c, y, :].numpy())
        print(f"c={c}: {'MATCH' if m else 'MISMATCH'}")
    for oc in range(CHANNELS // 8):
        c = oc * 8
        m = np.all(out[c, y, :] == r1[0, c, y, :].numpy())
        print(f"oc={oc}: {'MATCH' if m else 'MISMATCH'}")
        print(f"  oc={oc} Got:      {list(out[c, y, :].astype(int))}")
        print(f"  oc={oc} Expected: {list(r1[0, c, y, :].numpy().astype(int))}")

if __name__ == "__main__":
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 36
    run_diag(w=w)

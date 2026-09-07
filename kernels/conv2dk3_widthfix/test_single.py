import torch
import numpy as np
import aie.iron as iron
from aie.utils.ml import DataShaper
import sys
sys.path.insert(0, "kernels/conv2dk3_widthfix")
from design import CHANNELS, conv3x3_only

def test_single_weight(w=36):
    h = 4
    scale = 6
    # ONLY w2[0, 0, 1, 1] = 64
    w2 = torch.zeros((CHANNELS, CHANNELS, 3, 3), dtype=torch.float32)
    w2[0, 0, 1, 1] = 64.0

    r1 = torch.zeros((1, CHANNELS, h, w), dtype=torch.float32)
    # Give every channel a distinct constant or ramp
    for c in range(CHANNELS):
        for x in range(w):
            r1[0, c, :, x] = x + 1

    ds = DataShaper()
    r1_np = r1.squeeze().data.numpy().astype(np.uint8)
    ifm = ds.reorder_mat(r1_np, "YCXC8", "CYX")
    wts = ds.reorder_mat(w2.data.numpy().astype(np.int8), "OIYXI8O8", "OIYX")
    print("wts non-zero indices:", np.nonzero(wts.flatten())[0])
    print("wts non-zero values:", wts.flatten()[np.nonzero(wts.flatten())[0]])
    print("ifm row 0, first 32 bytes (pixel 0..3 of ic=0):")
    print(ifm[0:32].reshape(4, 8))
    print("ifm row 0, bytes 32..63 (pixel 4..7 of ic=0):")
    print(ifm[32:64].reshape(4, 8))

    a = iron.tensor(ifm, dtype=np.uint8)
    b = iron.tensor(wts, dtype=np.int8)
    c = iron.zeros(h * w * CHANNELS, dtype=np.uint8)

    conv3x3_only(a, b, c, tensor_w=w, tensor_h=h, scale=scale)

    raw = c.numpy().view(np.uint8)
    row_bytes = w * CHANNELS
    row0 = raw[0 : row_bytes]
    row1 = raw[row_bytes : 2 * row_bytes]
    oc0_bytes = row1[0 : 288]

    print(f"=== Single weight w2[0,0,1,1]=64 test at width={w} ===")
    nz0 = np.nonzero(row0)[0]
    print(f"Total non-zero bytes in row 0: {len(nz0)}")
    for idx in nz0[:20]:
        oc = idx // (w * 8)
        pix = (idx % (w * 8)) // 8
        ch = idx % 8
        print(f"  row 0 byte {idx}: oc={oc}, pixel={pix}, ch={ch} -> val={row0[idx]}")
    print("oc=0 Leftmost (pixels 0..3, all 8 channels):")
    print(oc0_bytes[0:32].reshape(4, 8))
    print("oc=0 Remainder chunk 0 (pixels 4..7, all 8 channels):")
    print(oc0_bytes[32:64].reshape(4, 8))
    print("oc=0 Remainder chunk 1 (pixels 8..11, all 8 channels):")
    print(oc0_bytes[64:96].reshape(4, 8))
    print("oc=0 Rightmost (pixels 32..35, all 8 channels):")
    print(oc0_bytes[256:288].reshape(4, 8))

    # Check non-zero elements in the whole row
    nz = np.nonzero(row1)[0]
    print(f"Total non-zero bytes in row 1: {len(nz)}")
    for idx in nz[:20]:
        oc = idx // (w * 8)
        pix = (idx % (w * 8)) // 8
        ch = idx % 8
        print(f"  byte {idx}: oc={oc}, pixel={pix}, ch={ch} -> val={row1[idx]}")

if __name__ == "__main__":
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 36
    test_single_weight(w)

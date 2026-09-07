import torch
import numpy as np
import aie.iron as iron
from aie.utils.ml import DataShaper
import sys
sys.path.insert(0, "kernels/conv2dk3_widthfix")
from design import CHANNELS, conv3x3_only

w = int(sys.argv[1]) if len(sys.argv) > 1 else 36
h = 4
scale = 6
print(f"=== inspect_raw for width={w} ===")
w2 = torch.zeros((CHANNELS, CHANNELS, 3, 3), dtype=torch.float32)
for ch in range(CHANNELS):
    w2[ch, ch, 1, 1] = 64.0

r1 = torch.zeros((1, CHANNELS, h, w), dtype=torch.float32)
for ch in range(CHANNELS):
    cb = ch // 8
    for x in range(w):
        r1[0, ch, :, x] = cb * 30 + (x % 30) + 1

ds = DataShaper()
ifm = ds.reorder_mat(r1.squeeze().data.numpy().astype(np.uint8), "YCXC8", "CYX")
wts = ds.reorder_mat(w2.data.numpy().astype(np.int8), "OIYXI8O8", "OIYX")

a = iron.tensor(ifm, dtype=np.uint8)
b = iron.tensor(wts, dtype=np.int8)
c = iron.zeros(h * w * CHANNELS, dtype=np.uint8)

conv3x3_only(a, b, c, tensor_w=w, tensor_h=h, scale=scale)

raw = c.numpy().view(np.uint8)
row_bytes = w * CHANNELS # 36 * 64 = 2304 bytes
# row 1 is raw[2304 : 4608]
row1 = raw[row_bytes : 2 * row_bytes]
print("Row 1 total bytes:", len(row1))

# Let's inspect oc=0: bytes 0..287 of row 1
oc0_bytes = row1[0 : 288]
print("oc=0, bytes 0..31 (pixels 0..3, Leftmost):")
print(oc0_bytes[0:32].reshape(4, 8))

print("oc=0, bytes 32..63 (pixels 4..7, Remainder chunk 0):")
print(oc0_bytes[32:64].reshape(4, 8))

print("oc=0, bytes 64..95 (pixels 8..11, Remainder chunk 1):")
print(oc0_bytes[64:96].reshape(4, 8))

print("oc=0, bytes 96..127 (pixels 12..15, Remainder chunk 2):")
print(oc0_bytes[96:128].reshape(4, 8))

print("oc=0, bytes 256..287 (pixels 32..35, Rightmost):")
print(oc0_bytes[256:288].reshape(4, 8))

# Now inspect oc=1: bytes 288..575 of row 1
oc1_bytes = row1[288 : 576]
print("oc=1, bytes 0..31 (pixels 0..3, Leftmost):")
print(oc1_bytes[0:32].reshape(4, 8))

print("oc=1, bytes 32..63 (pixels 4..7, Remainder chunk 0):")
print(oc1_bytes[32:64].reshape(4, 8))

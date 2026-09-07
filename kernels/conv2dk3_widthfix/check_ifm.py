import torch
import numpy as np
from aie.utils.ml import DataShaper

h = 4
w = 36
CHANNELS = 64
r1 = torch.zeros((1, CHANNELS, h, w), dtype=torch.float32)
for c in range(CHANNELS):
    cb = c // 8
    for x in range(w):
        r1[0, c, :, x] = cb * 30 + (x % 30) + 1

ds = DataShaper()
r1_np = r1.squeeze().data.numpy().astype(np.uint8)
ifm = ds.reorder_mat(r1_np, "YCXC8", "CYX")
# In YCXC8: shape in memory is (Y, C//8, X, C8) -> (4, 8, 36, 8)
ifm_4d = ifm.reshape(h, CHANNELS // 8, w, 8)
print("ifm_4d shape:", ifm_4d.shape)

# In YCXC8:
# Shape is (Y, C//8, X, C8) -> (4, 8, 36, 8)
# For y=1:
y = 1
# For ic=0 (C//8 = 0):
ic0_data = ifm.reshape(h, CHANNELS // 8, w, 8)[y, 0, :, :]
print("ic=0, y=1, first 8 pixels channel 0:")
print(ic0_data[:8, 0])
print("ic=0, y=1, all 36 pixels channel 0:")
print(ic0_data[:, 0])

# For ic=1 (C//8 = 1):
ic1_data = ifm.reshape(h, CHANNELS // 8, w, 8)[y, 1, :, :]
print("ic=1, y=1, all 36 pixels channel 0:")
print(ic1_data[:, 0])

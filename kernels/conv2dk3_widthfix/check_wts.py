import torch
import numpy as np
from aie.utils.ml import DataShaper

CHANNELS = 64
w2 = torch.zeros((CHANNELS, CHANNELS, 3, 3), dtype=torch.float32)
for c in range(CHANNELS):
    w2[c, c, 1, 1] = 64.0

ds = DataShaper()
wts = ds.reorder_mat(w2.data.numpy().astype(np.int8), "OIYXI8O8", "OIYX")
print("wts shape:", wts.shape)
print("wts dtype:", wts.dtype)
print("total elements:", wts.size)
nz = np.nonzero(wts.flatten())[0]
print("First 16 non-zero indices in wts:", nz[:16])
print("Values at those indices:", wts.flatten()[nz[:16]])

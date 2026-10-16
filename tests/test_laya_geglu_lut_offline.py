# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent Gaussian reference and physical LUT banking, without hardware."""
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import geglu_lut as gl


def test_lookup_banks_duplicate_groups_of_four_not_individual_entries():
    samples=gl.samples(); physical=gl.physical_table().reshape(-1,8)
    assert samples.shape==(512,) and physical.shape==(128,8)
    assert np.array_equal(physical[:,:4],samples.reshape(-1,4))
    assert np.array_equal(physical[:,4:],samples.reshape(-1,4))
    assert samples[0]==.5 and np.all(samples[1:]<=samples[:-1])


def test_formula_against_independent_erfc_including_bin_and_asymptote_edges():
    grid=np.arange(-512,513,dtype=np.float32)/32
    points=np.unique(np.concatenate([np.linspace(-16,16,20001,dtype=np.float32),grid,
      np.nextafter(grid,np.float32(-np.inf)),np.nextafter(grid,np.float32(np.inf)),np.array([-100,100],np.float32)]))
    reference=np.fromiter((float(x)*.5*math.erfc(-float(x)/math.sqrt(2)) for x in points),np.float64)
    observed=gl.approximation(points)
    assert np.isfinite(observed).all()
    assert np.max(np.abs(observed.astype(np.float64)-reference))<5e-5
    assert gl.approximation(np.array([100],np.float32))[0]==100
    assert gl.approximation(np.array([-100],np.float32))[0]==0


def test_formula_retains_signed_zero():
    values=np.array([0.,-0.],np.float32)
    assert np.array_equal(gl.approximation(values).view(np.uint32),values.view(np.uint32))

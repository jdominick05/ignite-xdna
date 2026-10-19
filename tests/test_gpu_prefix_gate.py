"""Prevent numerical diagnostics from weakening the exact prefix-cache gate."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import copy

import numpy as np
import pytest

from tools.qualify_gemma4_prefix_worker import compare


def response():
    return dict(position=2, predictions=[7], tensors={'result_norm': np.array([1.,0.],np.float32)})


@pytest.mark.parametrize('change', ['one_ulp', 'signed_zero', 'prediction', 'shape', 'extra_tensor'])
def test_exact_gate_rejects_each_fidelity_change(change):
    expected = response()
    actual = copy.deepcopy(expected)
    if change == 'one_ulp':
        actual['tensors']['result_norm'][0] = np.nextafter(np.float32(1),np.float32(2))
    elif change == 'signed_zero':
        actual['tensors']['result_norm'][1] = np.float32(-0.)
    elif change == 'prediction':
        actual['predictions'] = [8]
    elif change == 'shape':
        actual['tensors']['result_norm'] = actual['tensors']['result_norm'].reshape(2,1)
    else:
        actual['tensors']['extra'] = np.zeros(1,np.float32)
    result = compare(actual,expected)
    assert not result['passed']
    if change in ('one_ulp','signed_zero'):
        assert result['tensors']['result_norm']['different_elements'] == 1
    if change == 'one_ulp':
        assert result['tensors']['result_norm']['bf16_different'] == 0


def test_exact_gate_accepts_identical_states():
    expected = response()
    assert compare(copy.deepcopy(expected),expected)['passed']

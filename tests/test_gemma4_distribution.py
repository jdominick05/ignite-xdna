"""Distribution diagnostics reject hidden top-1 and completeness failures."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import math

import numpy as np
import pytest

from tools.gemma4_distribution import compare_distribution,log_probs,summarize
from tools.ggml_gpu_target.logits_client import LogitTarget


def test_known_kl_nll_and_shift_invariance():
    control = np.log(np.array([.75,.25]))
    actual = np.log(np.array([.5,.5]))
    result = compare_distribution(actual,control,1)
    assert result['kl'] == pytest.approx(.75*math.log(1.5)+.25*math.log(.5))
    assert result['delta_nll'] == pytest.approx(-math.log(2))
    shifted = compare_distribution(control+100,control,0)
    assert shifted['kl'] < 1e-12 and abs(shifted['delta_nll']) < 1e-12


def test_near_tie_top1_failure_is_not_hidden_by_tiny_kl():
    result = compare_distribution(np.array([0.,1e-8]),np.array([1e-8,0.]),0)
    assert result['kl'] < 1e-10
    assert not summarize([result],1)['diagnostic_passed']
    with pytest.raises(ValueError):
        summarize([result],2)
    with pytest.raises(ValueError):
        log_probs(np.array([1.,np.nan]))


def test_logits_wire_layout_preserves_all_vocabulary_indices():
    expected = np.arange(2*262144,dtype=np.float32).reshape(2,262144)
    wire = expected.reshape(2,8,32768).transpose(2,1,0)[:,:,:,None]
    client = object.__new__(LogitTarget)
    client.request = lambda opcode,values: dict(predictions=[],tensors={'logits':wire})
    np.testing.assert_array_equal(client.logits(2),expected)
    with pytest.raises(ValueError):
        client.logits(17)

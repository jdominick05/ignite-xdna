"""Control-only prefix diagnostic preserves exact bits and token-row indices."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np

from tools.probe_gemma4_prefix_invariance import compare_rows, shared_length


def test_changed_token_row_is_retained_even_when_bf16_rounds_equal():
    control = np.ones((7,4),np.float32)
    value = control.copy()
    value[5,2] = np.nextafter(np.float32(1),np.float32(2))
    result = compare_rows(value,control)
    assert not result['bitwise_equal'] and result['changed_rows']==[5]
    assert result['changed_elements']==1 and result['bf16_different']==0


def test_signed_zero_and_shared_token_prefix_are_exact():
    control = np.zeros((3,2),np.float32)
    value = control.copy()
    value[1,0] = -0.
    result = compare_rows(value,control)
    assert result['changed_rows']==[1] and result['max_abs']==0
    assert shared_length([1,2,3],[1,2,4])==2
    assert shared_length([1,2],[1,2,3])==2
    assert compare_rows(control,control)['bitwise_equal']

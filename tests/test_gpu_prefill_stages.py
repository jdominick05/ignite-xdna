"""Stage layouts and serving fingerprints stay exact under instrumentation."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np
import pytest

from tools.probe_gemma4_prefill_stages import serving_view, token_rows, causal_rows, masked_probability_stats
from tools.profile_gemma4_target import fingerprint


def test_token_rows_handles_hidden_and_multihead_kv():
    hidden = np.arange(28, dtype=np.float32).reshape(4,7,1,1)
    np.testing.assert_array_equal(token_rows(hidden,7), hidden[:,:,0,0].T)
    kv = np.arange(56, dtype=np.float32).reshape(4,2,7,1)
    rows = token_rows(kv,7)
    assert rows.shape == (7,8) and rows.flags.c_contiguous
    np.testing.assert_array_equal(rows[5], kv[:,:,5,0].reshape(-1))
    with pytest.raises(ValueError):
        token_rows(np.zeros((7,7),np.float32),7)


def test_regular_tensor_is_not_removed_when_selected_as_stage():
    original = dict(position=7,predictions=[42], tensors={'result_norm':np.ones((2,1),np.float32),
        'Kcur_pos-13':np.ones((2,7),np.float32)})
    response = dict(original,tensors={**original['tensors'],'l_out-0':np.zeros((2,7),np.float32)})
    assert serving_view(response,'l_out-0') == fingerprint(original)
    assert serving_view(original,'Kcur_pos-13') == fingerprint(original)
    response['predictions'] = [43]
    assert serving_view(response,'l_out-0') != fingerprint(original)


def test_causal_scores_exclude_padding_and_future_but_keep_visible_bits():
    original = np.ones((8,7,2,1),np.float32)
    extended = np.ones((12,10,2,1),np.float32)
    extended[7:,:,:,:] = 99
    extended[3,1,0,0] = 77  # future key for query 1
    np.testing.assert_array_equal(causal_rows(original,7,7),causal_rows(extended,10,7))
    extended[1,3,0,0] = np.nextafter(np.float32(1),np.float32(2))
    assert not np.array_equal(causal_rows(original,7,7).view(np.uint32),
                              causal_rows(extended,10,7).view(np.uint32))


def test_future_probability_guard_covers_extension_padding_and_signed_zero():
    probabilities = np.zeros((12,10,2,1),np.float32)
    probabilities[1,3,0,0] = 1  # visible to query 3
    assert masked_probability_stats(probabilities,10,7)['all_positive_zero']
    probabilities[11,1,0,0] = np.nextafter(np.float32(0),np.float32(1))
    assert masked_probability_stats(probabilities,10,7)['future_and_padding_bits_nonzero'] == 1
    probabilities[11,1,0,0] = -0.
    assert not masked_probability_stats(probabilities,10,7)['all_positive_zero']

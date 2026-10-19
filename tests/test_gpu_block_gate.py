"""A consistent cache cannot override an unchanged-serving numerical failure."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
from tools.qualify_gemma4_block_prefill import evaluate


def case(original=True,cache=True,rows=128):
    comparison = lambda passed: dict(prepare=dict(passed=passed),continuations=[dict(passed=passed)])
    return dict(name='extended',cold_blocks_vs_serving=comparison(original),
        cached_vs_cold_blocks=comparison(cache),evaluated_rows=rows,cold_block_rows=640)


def test_old_numerical_gate_is_required_even_for_consistent_saved_work():
    result = evaluate([case(original=False)])
    assert result['cache_consistency_passed'] and result['changed_prompt_work_avoided']
    assert not result['passed'] and not result['original_state_exact']


def test_recomputed_full_prompt_or_bad_continuation_cannot_pass():
    assert not evaluate([case(rows=640)])['passed']
    value = case()
    value['cached_vs_cold_blocks']['continuations'][0]['passed'] = False
    assert not evaluate([value])['passed']
    assert evaluate([case()])['passed']

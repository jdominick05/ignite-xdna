# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The small-query registration must retain original gates and add bit equivalence."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import small_lookup_accuracy as accuracy


def test_small_lookup_registration_preserves_parent_and_original_gates():
    fields=('criteria','criteria_text','items','reference','decision')
    old={key:{'pinned':key} for key in fields}
    old.update(native_policy={'contexts':5,'dispatches':66,'cpu_neural_fallback':False},arms={'old':{'files':{}}})
    result=accuracy.document(old,{'source':'hash'},{'packing':'hash'})
    for key in fields:assert result[key]==old[key] and result[key] is not old[key]
    assert result['native_policy']['cpu_neural_fallback'] is False
    assert 'small_query_mlp' not in old['native_policy'] and set(old['arms'])=={'old'}
    assert set(result['arms'])=={accuracy.NAME}
    assert result['packing_equivalence_reference']=={'packing':'hash'}
    assert 'bit-exact' in result['packing_equivalence_requirement']
    assert result['development_disclosure']['labelled_final_scores_before_registration']==0
    assert 'known-set' in result['development_disclosure']['heldout']

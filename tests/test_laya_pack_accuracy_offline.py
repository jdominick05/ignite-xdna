# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Packing must retain every original frozen field and add lookup equivalence."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import pack_accuracy as pack


def test_packing_registration_preserves_original_criteria_and_parent_document():
    fields=('criteria','criteria_text','items','reference','decision')
    old={key:{'pinned':key} for key in fields}
    old.update(native_policy={'contexts':5,'dispatches':66,'cpu_neural_fallback':False},arms={'old':{'files':{}}})
    result=pack.document(old,{'source':'hash'},{'lookup':'hash'},{'native':'hash'})
    for key in fields:
        assert result[key]==old[key] and result[key] is not old[key]
    assert result['native_policy']['cpu_neural_fallback'] is False
    assert 'packing' not in old['native_policy']
    assert set(old['arms'])=={'old'} and set(result['arms'])=={pack.NAME}
    assert result['lookup_equivalence_reference']=={'lookup':'hash'}
    assert 'bit-exact' in result['lookup_equivalence_requirement']
    assert result['development_disclosure']['labelled_final_scores_before_registration']==0

# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Attention replacement must preserve full512 control selection and mask addresses."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import laya_attention512_full as full
import attention512_accuracy as accuracy


@pytest.mark.parametrize('length',[1,91,129,512])
def test_selection_restores_control_instructions_and_mask_addresses(monkeypatch,length):
    obj=object.__new__(full.Session)
    obj.full512_attention_original={f'attention_{i}':f'control{i}' for i in range(28)}
    obj.full512_attention={f'attention_{i}':f'candidate{i}' for i in range(28)}
    obj.full512_attention_masks={f'attention_{i}':[i] for i in range(28)}
    obj.full512_attention_masks['head_attention']=[1000]
    obj.full512_candidate_masks={f'attention_{i}':[i+100] for i in range(28)}
    obj.instructions={'mlp_0':'adopted_polynomial'}
    def run(self,ids,markers,qtype,**kwargs):
        assert len(ids)==length
        assert self.instructions['mlp_0']=='adopted_polynomial'
        assert self.original_masks['head_attention']==[1000]
        return 'sentinel'
    monkeypatch.setattr(full.polynomial.Session,'run',run)
    assert obj.run([1]*length,[0,0],0)=='sentinel'
    assert obj.instructions['attention_0']=='candidate0'
    assert obj.original_masks['attention_0']==[100]
    obj.run([1]*length,[0,0],0,attention_layers=())
    assert all(obj.instructions[f'attention_{i}']==f'control{i}' for i in range(28))
    assert obj.original_masks['attention_0']==[0]
    obj.run([1]*length,[0,0],0,attention_layers=(1,))
    assert obj.instructions['attention_0']=='control0'
    assert obj.instructions['attention_1']=='candidate1'
    assert obj.original_masks['attention_0']==[0]
    assert obj.original_masks['attention_1']==[101]


@pytest.mark.parametrize('layers',[(-1,),(28,),(0,0),(True,)])
def test_invalid_layers_rejected_before_dispatch(layers):
    obj=object.__new__(full.Session)
    with pytest.raises(ValueError,match='layer selection'):
        obj.run([1],[0,0],0,attention_layers=layers)


@pytest.mark.parametrize('change',['none','rows','source','arithmetic'])
def test_metadata_cannot_change_rows_source_or_arithmetic(monkeypatch,tmp_path,change):
    folder=tmp_path/'candidate'
    folder.mkdir()
    parent=tmp_path/'chain'/'repeat'
    parent.mkdir(parents=True)
    (parent/'kernel.o').write_bytes(b'original arithmetic')
    meta=dict(sources={'original':'hash'},base_identity='base',
        policy=dict(m=512,q_depth=2,kv_depth=2,l2_q_depth=2),
        kernel_object_sha256=full.c.sha(parent/'kernel.o'))
    if change=='rows':
        meta['policy']['m']=128
    elif change=='source':
        meta['sources']={'changed':'hash'}
    elif change=='arithmetic':
        meta['kernel_object_sha256']='different'
    (folder/'build.json').write_text(json.dumps(meta),encoding='utf-8')
    monkeypatch.setattr(full.attention,'sources',lambda:{'original':'hash'})
    monkeypatch.setattr(full.c.op,'artifacts',lambda *args:b'validated instructions')
    bundle=SimpleNamespace(identity='base',folders={'chain':parent.parent})
    if change=='none':
        assert full.validate(folder,bundle)==(meta,b'validated instructions')
    else:
        with pytest.raises(ValueError):
            full.validate(folder,bundle)


@pytest.mark.parametrize('change',['none','missing_layer','missing_case','nonexact','fallback','incomplete'])
def test_frozen_registration_requires_complete_device_development(tmp_path,change):
    layers=[dict(layer=i,fixture=j,exact=dict(logits=True,act_probs=True)) for i in range(2) for j in range(3)]
    rows=[dict(case=str(i),exact=dict(logits=True,act_probs=True),contexts=5,
        dispatches=66,attention_stages=28) for i in range(25)]
    result=dict(cases=25,all_output_bits_exact=True,cpu_neural_oracle_calls=0,
        idle_after_close=True,registered_files_unchanged=True)
    end=dict(exit_code=0,stop=None)
    if change=='missing_layer':
        layers.pop()
    elif change=='missing_case':
        rows.pop()
    elif change=='nonexact':
        rows[0]['exact']['logits']=False
    elif change=='fallback':
        result['cpu_neural_oracle_calls']=1
    elif change=='incomplete':
        end['exit_code']=1
    lines=['ATTENTION512_LAYER_CHECK '+json.dumps(row) for row in layers]
    lines+=['ATTENTION512_FULL_CHECK '+json.dumps(row) for row in rows]
    lines+=['ATTENTION512_FULL_RESULT '+json.dumps(result),'RESEARCH_PROCESS_RESULT '+json.dumps(end)]
    path=tmp_path/'development.log'
    path.write_text('\n'.join(lines),encoding='utf-8')
    if change=='none':
        assert accuracy.development_gate(path)==full.c.sha(path)
    else:
        with pytest.raises(ValueError):
            accuracy.development_gate(path)


def test_original_frozen_criteria_and_fixed512_policy_are_preserved():
    original=json.loads(accuracy.base.na.ORIGINAL.read_text(encoding='utf-8'))
    before=json.dumps(original,sort_keys=True)
    arm=accuracy.document(original,{'test':'file-map'},{'test':'prior-reference'})
    assert json.dumps(original,sort_keys=True)==before
    for key in ('criteria','criteria_text','items','reference','decision'):
        assert arm[key]==original[key]
    assert set(arm['arms'])=={accuracy.NAME}
    assert arm['native_policy']['encoder_row_bucket'].startswith('fixed512 for EVERY item')
    assert arm['native_policy']['embedding_and_head_schedule']==512
    assert arm['native_policy']['contexts']==5
    assert arm['native_policy']['dispatches']==66
    assert arm['native_policy']['cpu_neural_fallback'] is False
    assert arm['native_policy']['attention_stages']==28

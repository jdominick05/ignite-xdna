# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reject emitted lock-field corruption before a new FIFO layout reaches hardware."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import window_lock_check as check
import window_core_check as control


@pytest.mark.parametrize('row',[1,2])
@pytest.mark.parametrize('field',range(6))
def test_each_compiled_lock_field_is_checked(row,field):
    col=2;bid=3
    shifts=(0,8,15,16,24,31) if row==1 else (0,5,12,13,18,25)
    base=64 if row==1 else 0
    values=(base+4,127,1,base+5,1,1)
    word=sum(value<<shift for value,shift in zip(values,shifts))
    address=(col<<25)|(row<<20)|((0xa0000+28 if row==1 else 0x1d000+20)+32*bid)
    acquire=(col,row,4,0,'acquire');release=(col,row,5,0,'release')
    check.verify_bd_lock_word({address:word},(col,row),bid,acquire,release)
    with pytest.raises(ValueError,match='compiled BD lock fields'):
        check.verify_bd_lock_word({address:word^(1<<shifts[field])},(col,row),bid,acquire,release)


def test_missing_or_nonlocal_lock_rejected():
    with pytest.raises(ValueError,match='missing compiled'):
        check.verify_bd_lock_word({},(0,2),0,(0,2,1),(0,2,2))
    with pytest.raises(ValueError,match='nonlocal'):
        check.verify_bd_lock_word({},(0,2),0,(1,2,1),(0,2,2))


def test_actual_compiled_primitive_lock_relation():
    root=Path.cwd()/'scratch/laya_transformer/window_lookup_primitive_v2'
    if not (root/'build.json').exists():pytest.skip('compiled primitive unavailable')
    result=check.compare(root/'original0',root/'window')
    assert result['hardware_lock_initializations_per_image']==168
    assert result['compiled_bd_lock_fields_checked_per_image']==192
    assert result['per_object_dma_handshake_roles_identical']


@pytest.mark.parametrize('window',[False,True])
def test_actual_core_ledger_wraps_physical_query_and_weight_rings(window):
    root=Path.cwd()/'scratch/laya_transformer/window_lookup_primitive_v2'
    path=root/('window' if window else 'original0')/'prj/input_with_addresses.mlir'
    if not path.exists():pytest.skip('compiled primitive unavailable')
    result=control.verify(path.read_text(),window)
    for row in result['cores']:
        assert row['tokens']==dict(q=6,kv=12 if window else 6,out=6 if window else 3)
        assert row['parked_on_next_query']


@pytest.mark.parametrize('mutation',['reset','lock_amount','slot','unknown_operation','query_ring'])
def test_core_trace_rejects_changed_protocol(mutation):
    root=Path.cwd()/'scratch/laya_transformer/window_lookup_primitive_v2'
    path=root/'window/prj/input_with_addresses.mlir'
    if not path.exists():pytest.skip('compiled primitive unavailable')
    text=path.read_text()
    old,new={
        'reset':('arith.cmpi eq, %19, %c0_i32','arith.cmpi eq, %19, %c1_i32'),
        'lock_amount':('aie.use_lock(%qF_0_cons_cons_lock_0, AcquireGreaterEqual, %c1_i32)',
                       'aie.use_lock(%qF_0_cons_cons_lock_0, AcquireGreaterEqual, %c2_i32)'),
        'slot':('%attn_state_0, %24, %35) :','%attn_state_0, %24, %24) :'),
        'unknown_operation':('%21 = arith.addi','%21 = arith.subi'),
        'query_ring':('cf.br ^bb8(%qF_0_cons_buff_2 :','cf.br ^bb8(%qF_0_cons_buff_1 :'),
    }[mutation]
    assert old in text
    with pytest.raises(ValueError):control.verify(text.replace(old,new,1),True)

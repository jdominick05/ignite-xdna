# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reject corrupt actual CDO fields before the one-shot cache reaches hardware."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import cached_wi_checks as checks


def images():
    root=Path.cwd()/'scratch/laya_transformer';folder=root/'cached_wi_v5_col0'
    if not (folder/'build.json').exists():pytest.skip('compiled cache prototype unavailable')
    parent=root/'small_lookup_v3_m128'
    return parent,folder


def test_actual_code_load_and_every_compiled_replay_word():
    parent,folder=images();proof=checks.verify(parent,folder)
    assert proof['code_data_and_enable_cdos_bit_identical']
    assert proof['delivered_query_word_addresses_checked']==1397760
    assert proof['compiled_memtile_descriptors_checked']==16


@pytest.mark.parametrize('address,bit',[(0x1a0000,1),(0x1a0004,1),(0x1a001c,1<<8),
    (0x1a0644,1<<16),(0x1c0020,1),(0x1c03f0,1),(0x21a0000,1)])
def test_reject_corrupted_cache_length_address_lock_repeat_or_other_column(address,bit):
    parent,folder=images();old=checks.state(parent/'prj/cdo_main/main_aie_cdo_init.bin')
    new=checks.state(folder/'prj/cdo_main/main_aie_cdo_init.bin');new[address]=new.get(address,0)^bit
    with pytest.raises(ValueError):checks.verify_states(old,new,checks.q.sq.bm.b.plan())

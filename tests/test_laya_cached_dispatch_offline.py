# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every dispatch initializes its own complete finite query task state."""
import json
from pathlib import Path
import struct
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import cached_mlp_dispatch as dispatch


@pytest.fixture(scope='module')
def image():
    root=Path.cwd()/'scratch/laya_transformer';parent=root/'small_lookup_v3_m128';folder=root/'cached_mlp_dispatch_v3_m128'
    if not (folder/'build.json').exists():pytest.skip('compiled self-contained dispatch image unavailable')
    return parent,folder


def test_actual_image_has_no_query_startup_tasks_and_complete_runtime_setup(image):
    proof=dispatch.verify(*image)
    assert proof['runtime']['query_setup_writes']==768
    assert proof['runtime']['finite_task_writes']==80
    assert proof['runtime']['queues']==439


@pytest.mark.parametrize('index',[0,7,127,256,767,768])
def test_runtime_rejects_descriptor_enable_or_phase_task_corruption(image,index):
    parent,folder=image;raw=bytearray((folder/'insts.bin').read_bytes())
    ops=dispatch.previous.decoder.parse(raw);op=ops[index];offset=op['offset']+16
    value=struct.unpack_from('<I',raw,offset)[0];struct.pack_into('<I',raw,offset,value^1)
    text=(parent/'prj/input_with_addresses.mlir').read_text(encoding='utf-8')
    old=dispatch.c.checks.state(parent/'prj/cdo_main/main_aie_cdo_init.bin')
    meta=json.loads((parent/'build.json').read_text(encoding='utf-8'))
    with pytest.raises(ValueError):dispatch.runtime_proof(bytes(raw),old,text,dispatch.q.sq.bm.b.plan(),meta['audits'])

# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The specialized arithmetic image must inherit the actual working transport."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels/laya_transformer'))
import gemm_specialized_primitive as candidate


def compiled():
    root=Path.cwd()/'scratch/laya_transformer'
    folder=root/'gemm_specialized_primitive_v2'
    if not (folder/'build.json').exists():pytest.skip('compiled specialized image unavailable')
    meta=json.loads((folder/'build.json').read_text())
    return folder,root/'window_lookup_primitive_v2',meta


def test_compiled_image_inherits_working_transport_and_fits():
    folder,reference,meta=compiled()
    assert candidate.qualify(folder,reference)==meta
    for row in meta['layouts'].values():
        assert max(a['code_bytes'] for a in row['audits'])<=16384
        assert max(sum(a['frames'])+256 for a in row['audits'])<=5376


def test_changed_transaction_bytes_are_rejected(tmp_path):
    folder,reference,meta=compiled()
    raw=bytearray((folder/'original0/insts.bin').read_bytes());raw[-1]^=1
    (tmp_path/'insts.bin').write_bytes(raw)
    with pytest.raises(ValueError,match='transaction stream'):
        candidate.check_layout(tmp_path,reference/'original0',meta['layouts']['original0']['audits'],0)


def test_changed_sram_span_is_rejected():
    folder,reference,meta=compiled()
    audits=deepcopy(meta['layouts']['original0']['audits'])
    audits[0]['spans'][0][0]+=4
    with pytest.raises(ValueError,match='SRAM buffers'):
        candidate.check_layout(folder/'original0',reference/'original0',audits,0)

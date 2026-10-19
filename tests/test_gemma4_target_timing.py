"""Decode diagnostic protocol bounds without GPU hardware."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import copy
import io
import json
import struct
from types import SimpleNamespace

import pytest

from tools.ggml_gpu_target.client import Target


def sample():
    return dict(version=1,groups={'1':dict(calls=2,tokens=4,capture_calls=12,
        captured_bytes=32768,response_bytes=34000,process_ms=80.0,capture_ms=0.2,
        logits_ms=1.0,tensors_pack_ms=0.01,send_ms=0.3)})


def receive(value,*,count=0,tensors=0,tail=b'',encoded=None):
    data=json.dumps(value).encode('ascii') if encoded is None else encoded
    body=struct.pack('<6I',0,8,12,count,tensors,len(data))+data+tail
    target=object.__new__(Target)
    target.timeout=5;target.context_tokens=512
    target.process=SimpleNamespace(stdout=io.BytesIO(struct.pack('<I',len(body))+body),kill=lambda:None)
    return target.receive()


def test_timing_roundtrip_has_no_predictions_or_tensors():
    actual=receive(sample())
    assert actual==dict(opcode=8,position=12,predictions=[],tensors={},timing=sample())


@pytest.mark.parametrize('kind',['version','group','missing','negative','nan','bool','fraction','oversize','trailing'])
def test_bad_timing_payload_rejected(kind):
    value=copy.deepcopy(sample())
    if kind=='version':value['version']=True
    elif kind=='group':value['groups']['99']=value['groups'].pop('1')
    elif kind=='missing':del value['groups']['1']['calls']
    elif kind=='negative':value['groups']['1']['send_ms']=-1
    elif kind=='nan':value['groups']['1']['process_ms']=float('nan')
    elif kind=='bool':value['groups']['1']['capture_ms']=True
    elif kind=='fraction':value['groups']['1']['calls']=1.5
    with pytest.raises(ValueError):
        receive(value,encoded=b' '*4097 if kind=='oversize' else None,tail=b'x' if kind=='trailing' else b'')

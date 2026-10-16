#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Captured complete-layer gate for one-state small-query lookup/KV2."""
import json
from pathlib import Path
import sys
from types import FunctionType

import laya_window_layer_probe as original
sys.path.insert(0,str(Path.cwd()/'kernels/laya_transformer'))
import small_lookup_primitive as primitive
full=original.full;c=full.c;candidate=primitive.candidate


def gates(path,build_sha):
    segments=[]
    for line in path.read_text().splitlines():
        if line.startswith('RAW_ARCHIVE '):segments.append([])
        if segments:segments[-1].append(line)
    verdicts={}
    for lines in segments:
        rows=[(line.split(' ',1)[0],json.loads(line.split(' ',1)[1])) for line in lines
              if line.startswith(('WINDOW_PRIMITIVE_RESULT ','WINDOW_VECTOR_RESULT '))]
        if not rows:continue
        terminal=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('RESEARCH_PROCESS_RESULT ')]
        starts=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('SMALL_LOOKUP_PRIMITIVE_GATES ')]
        checks=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith(('WINDOW_PRIMITIVE_CHECK ','WINDOW_VECTOR_CHECK '))]
        tag,row=rows[0];key=('gemm:' if tag=='WINDOW_PRIMITIVE_RESULT' else 'vector:')+row['variant']
        if (len(rows)!=1 or len(terminal)!=1 or terminal[0]['exit_code']!=0 or terminal[0]['stop'] is not None or
            len(starts)!=1 or starts[0]['build_sha256']!=build_sha or row['cases']!=6 or not row['idle_after_close'] or
            not row['all_output_bits_and_guards_pass' if tag=='WINDOW_PRIMITIVE_RESULT' else 'all_checks_pass'] or
            len(checks)!=6 or not all(all(check['checks'].values()) for check in checks) or key in verdicts):
            raise ValueError('small lookup primitive prerequisite failed')
        verdicts[key]=row
    if set(verdicts)!={'gemm:original0','gemm:original1','vector:original','vector:window'}:
        raise ValueError('small lookup requires all four primitive profiles')
    if verdicts['vector:window']['reference_sha256']!=verdicts['vector:original']['output_sha256']:
        raise ValueError('small lookup vector baseline not pinned')
    return c.sha(path)


def identity():
    paths,bundle,fixtures,files,proof=full.identity();folder=paths.base/'small_lookup_v3_m128'
    meta=json.loads((folder/'build.json').read_text());prim=paths.base/'small_lookup_primitive_v1'
    pm=primitive.qualify(prim,'vector');row=pm['layouts']['window']
    if meta['sources']!=candidate.sources() or meta['base_identity']!=bundle.identity:raise ValueError('small lookup full image source/base')
    raw=full.op.artifacts(folder,meta)
    if meta['audits']!=row['audits'] or meta['cdo']!=row['cdo'] or meta['kernel_object_sha256']!=row['kernel_sha256']:
        raise ValueError('small lookup full image changes guarded core/init')
    transport=candidate.verify(raw,candidate.events(bundle.plan),meta['audits'],bundle.plan['words'])
    if transport!=meta['transport']:raise ValueError('small lookup full transport changed')
    evidence=c.ROOT/'results/laya/small_lookup_v3_primitives_desktop2_20261016.log'
    proof=dict(proof,small_lookup_transport=transport,small_lookup_primitives=gates(evidence,c.sha(prim/'build.json')))
    for file in ('build.json','kernel.o','insts.bin','final.xclbin'):files['small_lookup/'+file]=c.sha(folder/file)
    for file,digest in primitive.sources().items():files['small_lookup_source/'+file]=digest
    for module in (original,sys.modules[__name__]):
        file=Path(module.__file__);files['small_lookup_source/'+str(file.relative_to(c.ROOT)).replace('\\','/')]=c.sha(file)
    files['small_lookup_primitives']=c.sha(evidence);files['small_lookup_primitive_build']=c.sha(prim/'build.json')
    return paths,bundle,fixtures,folder,meta,raw,files,proof


if __name__=='__main__':
    FunctionType(original.main.__code__,dict(original.main.__globals__,identity=identity,__file__=__file__))()

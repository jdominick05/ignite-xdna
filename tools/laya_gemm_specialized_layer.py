#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare specialized complete MLP layers against captured qualified NPU layers."""
import json
from pathlib import Path
import sys
from types import FunctionType

import laya_window_layer_probe as original
sys.path.insert(0,str(Path.cwd()/'kernels/laya_transformer'))
import gemm_specialized_image as image
full=original.full;c=full.c


def primitive_gates(path):
    segments=[]
    for line in path.read_text().splitlines():
        if line.startswith('RAW_ARCHIVE '):segments.append([])
        if segments:segments[-1].append(line)
    variants=[]
    for lines in segments:
        rows=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('WINDOW_PRIMITIVE_RESULT ')]
        if not rows:continue
        terminal=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('RESEARCH_PROCESS_RESULT ')]
        checks=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('WINDOW_PRIMITIVE_CHECK ')]
        if (len(rows)!=1 or len(terminal)!=1 or terminal[0]['exit_code']!=0 or terminal[0]['stop'] is not None or
            rows[0]['cases']!=6 or not rows[0]['idle_after_close'] or not rows[0]['all_output_bits_and_guards_pass'] or
            len(checks)!=6 or not all(all(row['checks'].values()) for row in checks)):
            raise ValueError('specialized primitive qualification failed')
        variants.append(rows[0]['variant'])
    if sorted(variants)!=['original0','original1']:raise ValueError('both guarded output groups required')
    return c.sha(path)


def identity():
    paths,bundle,fixtures,files,proof=full.identity()
    folder=paths.base/'gemm_specialized_mlp_v1_m128'
    guarded=paths.base/'gemm_specialized_primitive_v2';reference=paths.base/'window_lookup_primitive_v2'
    meta=image.validate(folder,paths.fast_folder,guarded,reference,bundle)
    evidence=c.ROOT/'results/laya/gemm_specialized_primitive_v2_desktop2_20261016.log'
    proof=dict(proof,guarded_primitives=primitive_gates(evidence),
        same_qualified_primitive_core_init=True,original_complete_runtime_bytes_identical=True)
    for name in ('build.json','kernel.o','insts.bin','final.xclbin'):files['specialized/'+name]=c.sha(folder/name)
    for file,digest in image.sources().items():files['specialized_source/'+file]=digest
    files['specialized_primitives']=c.sha(evidence);files['specialized_primitive_build']=c.sha(guarded/'build.json')
    return paths,bundle,fixtures,folder,meta,(folder/'insts.bin').read_bytes(),files,proof


if __name__=='__main__':
    FunctionType(original.main.__code__,dict(original.main.__globals__,identity=identity,__file__=__file__))()

#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Complete-model development parity for the guarded specialized MLP image."""
import json
from pathlib import Path
from types import FunctionType

import laya_gemm_specialized_layer as layer
import laya_window_full_probe as reference
full=layer.full;c=full.c;op=full.op


class Session(full.Session):
    def __init__(self,bundle,mlp_folder,attention_folder,mlp_candidate,attention_candidate,compact_folder,fixtures,head_folder,
                 specialized_folder,parent_folder,pack_folder):
        folder=Path(specialized_folder);base=folder.parent
        meta=layer.image.validate(folder,parent_folder,base/'gemm_specialized_primitive_v2',base/'window_lookup_primitive_v2',bundle)
        raw=op.artifacts(folder,meta)
        super().__init__(bundle,mlp_folder,attention_folder,mlp_candidate,attention_candidate,compact_folder,fixtures,head_folder,
                         fast_folder=None,pack_folder=pack_folder,mlp_kind='lookup')
        try:op.Session.install(self,'mlp',folder,meta,raw)
        except BaseException:self.close();raise


def layer_gate(path):
    lines=path.read_text().splitlines()
    results=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('WINDOW_LAYER_RESULT ')]
    terminals=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('RESEARCH_PROCESS_RESULT ')]
    if (len(results)!=1 or results[0]['cases']!=6 or not results[0]['all_arena_bits_exact'] or
        results[0]['cpu_neural_oracle_calls'] or not results[0]['idle_after_close'] or not terminals or
        any(row['exit_code']!=0 or row['stop'] is not None for row in terminals)):
        raise ValueError('specialized complete layer prerequisite failed')
    return c.sha(path)


def identity():
    paths,bundle,fixtures,folder,meta,raw,files,proof=layer.identity()
    evidence=c.ROOT/'results/laya/gemm_specialized_mlp_v1_layer_desktop2_20261016.log'
    proof=dict(proof,specialized_layer_qualification=layer_gate(evidence))
    files['specialized_layer_qualification']=c.sha(evidence)
    files['specialized_source/tools/laya_gemm_specialized_full.py']=c.sha(Path(__file__))
    files['specialized_source/tools/laya_window_full_probe.py']=c.sha(Path(reference.__file__))
    return paths,bundle,fixtures,folder,files,proof


def make(paths,bundle,fixtures,folder,specialized):
    if not specialized:return full.make(paths,bundle,fixtures,True)
    return Session(bundle,paths.mlp_folder,paths.attention_folder,paths.mlp_candidate,paths.attention_candidate,
        paths.compact_folder,fixtures,paths.head_folder,folder,paths.fast_folder,paths.base/'compact_pack_wide_v2_m128')


if __name__=='__main__':
    FunctionType(reference.qualify.__code__,dict(reference.qualify.__globals__,identity=identity,make=make,__file__=__file__))()

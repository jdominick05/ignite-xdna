#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Full-model development parity for one-state smaller-query lookup/KV2."""
import json
from pathlib import Path
from types import FunctionType

import laya_small_lookup_layer as layer
import laya_window_full_probe as reference
full=layer.full;c=full.c;op=full.op


class Session(full.Session):
    def __init__(self,bundle,mlp_folder,attention_folder,mlp_candidate,attention_candidate,compact_folder,fixtures,head_folder,
                 small_folder,pack_folder):
        folder=Path(small_folder);meta=json.loads((folder/'build.json').read_text())
        if meta['sources']!=layer.candidate.sources() or meta['base_identity']!=bundle.identity:raise ValueError('stale complete small lookup image')
        raw=op.artifacts(folder,meta)
        super().__init__(bundle,mlp_folder,attention_folder,mlp_candidate,attention_candidate,compact_folder,fixtures,head_folder,
            fast_folder=None,pack_folder=pack_folder,mlp_kind='lookup')
        try:op.Session.install(self,'mlp',folder,meta,raw)
        except BaseException:self.close();raise


def layer_gate(path):
    lines=path.read_text().splitlines()
    results=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('WINDOW_LAYER_RESULT ')]
    terminal=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('RESEARCH_PROCESS_RESULT ')]
    if (len(results)!=1 or results[0]['cases']!=6 or not results[0]['all_arena_bits_exact'] or
        results[0]['cpu_neural_oracle_calls'] or not results[0]['idle_after_close'] or
        len(terminal)!=1 or terminal[0]['exit_code']!=0 or terminal[0]['stop'] is not None):
        raise ValueError('small lookup complete layer prerequisite failed')
    return c.sha(path)


def identity():
    paths,bundle,fixtures,folder,meta,raw,files,proof=layer.identity()
    evidence=c.ROOT/'results/laya/small_lookup_v3_layer_desktop2_20261016.log'
    proof=dict(proof,small_lookup_layer=layer_gate(evidence));files['small_lookup_layer_qualification']=c.sha(evidence)
    for file in (Path(__file__),Path(reference.__file__)):
        files['small_lookup_source/'+str(file.relative_to(c.ROOT)).replace('\\','/')]=c.sha(file)
    return paths,bundle,fixtures,folder,files,proof


def make(paths,bundle,fixtures,folder,small):
    if not small:return full.make(paths,bundle,fixtures,True)
    return Session(bundle,paths.mlp_folder,paths.attention_folder,paths.mlp_candidate,paths.attention_candidate,
        paths.compact_folder,fixtures,paths.head_folder,folder,paths.base/'compact_pack_wide_v2_m128')


if __name__=='__main__':
    FunctionType(reference.qualify.__code__,dict(reference.qualify.__globals__,identity=identity,make=make,__file__=__file__))()

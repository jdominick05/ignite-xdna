#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reuse guarded runners for small-query GEMM and paired norm/lookup streams."""
import argparse
from pathlib import Path
import sys
from types import FunctionType,SimpleNamespace

sys.path.insert(0,str(Path.cwd()/'kernels/laya_transformer'))
import small_lookup_primitive as candidate
import laya_window_primitive as gemm
import laya_window_vector_primitive as vector


if __name__=='__main__':
    ap=argparse.ArgumentParser(add_help=False);ap.add_argument('--kind',choices=('gemm','vector'),required=True)
    ap.add_argument('--variant',choices=('original0','original1','original','window'),required=True)
    args,remaining=ap.parse_known_args()
    if (args.kind=='gemm' and args.variant not in ('original0','original1') or
        args.kind=='vector' and args.variant not in ('original','window')):ap.error('variant does not match primitive kind')
    sys.argv=[sys.argv[0],'--variant',args.variant,*remaining]
    original=gemm if args.kind=='gemm' else vector;fixture=candidate.gemm if args.kind=='gemm' else candidate.vector
    def qualify(folder):
        meta=candidate.qualify(folder,args.kind)
        original.emit('SMALL_LOOKUP_PRIMITIVE_GATES',dict(build_sha256=candidate.p.attention.sha(folder/'build.json'),
            kind=args.kind,one_output_state=True,physical_kv_buffers=2,all_compiled_checks_pass=True))
        return meta
    adapter=SimpleNamespace(p=candidate.p,plan=fixture.plan,fixture=fixture.fixture,sources=candidate.sources)
    namespace=dict(original.main.__globals__,qualify=qualify,primitive=adapter,__file__=__file__)
    FunctionType(original.main.__code__,namespace)()

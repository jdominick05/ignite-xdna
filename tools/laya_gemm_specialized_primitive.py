#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run the guarded primitive with a separately qualified specialized GEMM image."""
import json
from pathlib import Path
import sys
from types import FunctionType,SimpleNamespace

sys.path.insert(0,str(Path.cwd()/'kernels/laya_transformer'))
import gemm_specialized_primitive as candidate
import laya_window_primitive as original


def qualify(folder):
    reference=Path.cwd()/'scratch/laya_transformer/window_lookup_primitive_v2'
    meta=candidate.qualify(folder,reference)
    print('SPECIALIZED_PRIMITIVE_OFFLINE '+json.dumps(dict(build_sha256=candidate.p.attention.sha(folder/'build.json'),
        all_transport_memory_lock_gates_pass=True)),flush=True)
    return meta


if __name__=='__main__':
    # Private globals retain the existing fixture/guard/timeout/cleanup harness.
    namespace=dict(original.main.__globals__,qualify=qualify,
        primitive=SimpleNamespace(p=candidate.p,plan=candidate.original.plan,
            fixture=candidate.original.fixture,sources=candidate.sources),__file__=__file__)
    FunctionType(original.main.__code__,namespace)()

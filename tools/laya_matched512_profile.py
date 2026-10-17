#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Diagnostic attribution of the registered matched-512 arms; no adoption decision."""
from collections import Counter, defaultdict
from contextlib import ExitStack
import gc
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import laya_native_hybrid_compare as c
from laya_stage_profile import group


def exact(output, reference):
    if set(output) != set(reference) or any(
        not np.array_equal(value.view(np.uint32), reference[key].view(np.uint32))
        for key, value in output.items()
    ):
        raise ValueError('complete output repeatability failed')


def product_group(i, product):
    if i >= 112:
        return 'head_products'
    expected = [(1024, 3072), (1024, 1024), (1024, 5248), (2624, 1024)]
    if (product['K'], product['N']) != expected[i % 4]:
        raise ValueError('encoder product order/shape changed')
    return ['encoder_qkv', 'encoder_attention_projection', 'encoder_wi', 'encoder_wo'][i % 4]


def segment_group(i):
    # Boundaries remain explicit: these are exported segment slots, not pure op timings.
    if i == 0:
        return 'stem_segment'
    if i == 112:
        return 'last_encoder_residual_and_head_boundary'
    if i > 112:
        return 'head_segments'
    return {1: 'attention_between_products', 2: 'attention_residual_norm',
            3: 'geglu_between_products', 0: 'mlp_residual_norm'}[i % 4]


def totals(values, names):
    result = defaultdict(float)
    for value, name in zip(values, names, strict=True):
        result[name] += value
    return dict(result)


def main():
    prereg = c.ROOT / 'results/laya/native_hybrid_compare_v1_prereg_desktop2_20261016.json'
    pre, paths, bundle, fixtures = c.load_registration(prereg)
    item = fixtures[0]
    if item['ids'] != pre['fixture']['ids'] or c.hybrid.SEQ != 512:
        raise ValueError('registered fixture or schedule changed')
    c.emit('MATCHED512_START', dict(runner_sha256=c.sha(Path(__file__)),
        registration_sha256=c.sha(prereg),valid_tokens=len(item['ids']),scheduled_rows=512,
        warmups=2,repetitions=3,expected_gain_ms=0,
        kill_line='reject diagnostic if coverage, repeatability, cleanup or accounting fails',
        scope='instrumented wall attribution; dispatch times include driver/wait/DMA, not core cycles'))
    if not c.hybrid.idle():
        raise ValueError('device occupied before native')
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError('CPU neural/oracle call in native arm')
    n = c.op.n
    functions = [(c.ort.InferenceSession,'run'),(n.cr,'references'),
        (n.embedded.mask_reference,'extract'),(c.p.norm,'oracle'),(c.p.attention,'oracle'),
        (n.action,'reference'),(n.action,'verify'),(n.action.s,'reference'),(n.action.s,'verify'),
        (n.action.s.second,'reference'),(n.action.s.second.post,'reference')]
    common = (bundle, paths.mlp_folder, paths.attention_folder, paths.mlp_candidate,
              paths.attention_candidate, paths.compact_folder, fixtures, paths.head_folder)
    native_rows = []
    with ExitStack() as stack:
        for module, method in functions:
            stack.enter_context(patch.object(module, method, forbidden))
        with c.op.hp.Session(*common) as session:
            for i in range(5):
                start = time.perf_counter()
                output = session.run(item['ids'],item['markers'],item['qtype'],baseline=True,profile=i>=2)
                wall = (time.perf_counter()-start)*1000
                if i == 0:
                    native_reference = {key:value.copy() for key,value in output.items()}
                exact(output, native_reference)
                stats = session.last_stats
                if len(session.states)!=5 or stats['dispatches']!=66 or calls:
                    raise ValueError('native execution policy changed')
                if i < 2:
                    continue
                stages = stats['stages']
                counts = Counter(group(stage['name']) for stage in stages)
                if len(stages)!=66 or counts['encoder_mlp']!=28 or counts['encoder_attention']!=28:
                    raise ValueError('missing native stages')
                groups = totals([stage['ms'] for stage in stages],[group(stage['name']) for stage in stages])
                accounted = sum(groups.values())+stats['prepare_ms']+stats['readback_ms']
                if accounted > wall+0.01:
                    raise ValueError('native accounting exceeds external wall')
                row = dict(index=i-2,wall_ms=wall,groups_ms=groups,stats=stats,
                    outside_stage_prepare_read_ms=wall-accounted,group_counts=dict(counts))
                native_rows.append(row)
                c.emit('MATCHED512_NATIVE_SAMPLE',row)
    session = None
    gc.collect()
    if not c.hybrid.idle():
        raise ValueError('native cleanup failure')
    c.emit('MATCHED512_IDLE',dict(phase='after_native',idle=True,cpu_neural_oracle_calls=len(calls)))
    plan = c.hybrid.Plan(paths.model,threads=8)
    pgroups = [product_group(i,product) for i,product in enumerate(plan.products)]
    sgroups = [segment_group(i) for i in range(len(plan.segments))]
    c.emit('MATCHED512_HYBRID_LAYOUT',dict(products=[dict(index=i,name=p['name'],K=p['K'],N=p['N'],group=pgroups[i])
        for i,p in enumerate(plan.products)],segments=[dict(index=i,ops=ops,group=sgroups[i],
        active=plan.segments[i] is not None) for i,ops in enumerate(plan.seg_ops)],
        caveat='segment group names describe graph position; mixed boundaries retained, no per-op time inferred'))
    feed = c.hybrid.padded_feed(item['ids'],item['markers'],item['qtype'])
    products = c.hybrid.NpuProducts(plan,paths.gemm_dir)
    hybrid_rows = []
    try:
        for i in range(5):
            plan.seg_t = [0.0]*len(plan.segments)
            products.t = dict.fromkeys(products.t,0.0)
            before = products.counts.copy()
            times = [0.0]*len(plan.products)
            seen = []
            host_calls = []
            original = c.ort.InferenceSession.run
            def host_run(session,*args,**kwargs):
                host_calls.append(id(session))
                return original(session,*args,**kwargs)
            def product(index,x):
                start = time.perf_counter()
                result = products(index,x)
                times[index] += (time.perf_counter()-start)*1000
                seen.append(index)
                return result
            with patch.object(c.ort.InferenceSession,'run',host_run):
                start = time.perf_counter()
                output = dict(zip(plan.graph_outputs,plan.run(feed,product),strict=True))
                wall = (time.perf_counter()-start)*1000
            if i == 0:
                hybrid_reference = {key:value.copy() for key,value in output.items()}
            exact(output,hybrid_reference)
            counts = {key:products.counts[key]-before[key] for key in before}
            if counts!={'npu':118,'host':2} or seen!=list(range(120)):
                raise ValueError('hybrid product coverage changed')
            expected_calls = Counter(id(seg['sess']) for seg in plan.segments if seg is not None)
            if Counter(host_calls)!=expected_calls:
                raise ValueError('hybrid host segment coverage changed')
            segments = [value*1000 for value in plan.seg_t]
            residual = wall-sum(times)-sum(segments)
            if residual < -0.01:
                raise ValueError('hybrid accounting exceeds external wall')
            if i < 2:
                continue
            row = dict(index=i-2,wall_ms=wall,product_ms=times,segment_ms=segments,
                product_groups_ms=totals(times,pgroups),segment_groups_ms=totals(segments,sgroups),
                product_internal_ms={key:value*1000 for key,value in products.t.items()},
                orchestration_residual_ms=residual,counts=counts,host_segment_calls=len(host_calls))
            hybrid_rows.append(row)
            c.emit('MATCHED512_HYBRID_SAMPLE',row)
    finally:
        products.close()
    products = None
    plan = None
    gc.collect()
    if not c.hybrid.idle():
        raise ValueError('hybrid cleanup failure')
    c.load_registration(prereg)
    difference = {key:float(np.max(np.abs(value-hybrid_reference[key]))) for key,value in native_reference.items()}
    def aggregate(rows,field):
        return {key:c.stats([row[field][key] for row in rows]) for key in rows[0][field]}
    c.emit('MATCHED512_RESULT',dict(native_wall=c.stats([r['wall_ms'] for r in native_rows]),
        hybrid_wall=c.stats([r['wall_ms'] for r in hybrid_rows]),native_groups=aggregate(native_rows,'groups_ms'),
        hybrid_products=aggregate(hybrid_rows,'product_groups_ms'),hybrid_segments=aggregate(hybrid_rows,'segment_groups_ms'),
        cross_arm_max_abs=difference,repeat_outputs_bit_exact=True,registered_artifacts_unchanged=True,
        idle_after_close=True,adoption_decision=False))


if __name__ == '__main__':
    main()

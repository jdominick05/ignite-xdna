#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Layer-first wide-packing integration after guarded transport qualification."""
from contextlib import ExitStack
import argparse
import gc
import json
from pathlib import Path
from unittest.mock import patch

import laya_native_hybrid_compare as compare
import pack_sweep
import transport_walk
op=compare.op; n=op.n; p=compare.p


class Session(op.Session):
    def run(self,*args,pack_layers=(),**kwargs):
        selected={f'attention_{index}_m128' for index in pack_layers}
        instructions=self.candidate_attention; masks=self.candidate_mask_fields
        self.candidate_attention={name:(inst if name in selected else self.original_attention[name]) for name,inst in instructions.items()}
        self.candidate_mask_fields={name:(fields if name in selected else self.original_mask_fields[name]) for name,fields in masks.items()}
        try:
            return super().run(*args,**kwargs)
        finally:
            self.candidate_attention=instructions;self.candidate_mask_fields=masks


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--packing-build',type=Path,required=True)
    ap.add_argument('--packing-evidence',type=Path,required=True);args=ap.parse_args()
    packing=json.loads((args.packing_build/'build.json').read_text())
    if packing['sources']!=pack_sweep.sources(): raise ValueError('packing qualification sources changed')
    lines=args.packing_evidence.read_text().splitlines()
    rows=lambda tag:[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith(tag+' ')]
    identities=rows('PACK_PRIMITIVE_IDENTITY');results=rows('PACK_PRIMITIVE_RESULT');terminal=rows('RESEARCH_PROCESS_RESULT')
    checks=rows('PACK_PRIMITIVE_CHECK')
    if len(identities)!=1 or identities[0]['variant']!='compact' or identities[0]['build_sha256']!=compare.sha(args.packing_build/'build.json'):
        raise ValueError('packing evidence identity differs')
    if len(results)!=1 or results[0]['features']!=16 or not results[0]['all_guards_and_output_bits_pass'] or not results[0]['idle_after_close']:
        raise ValueError('all-feature packing not qualified')
    if len(terminal)!=1 or terminal[0]['exit_code']!=0 or len(checks)!=6 or not all(all(row['checks'].values()) for row in checks):
        raise ValueError('packing qualification incomplete')
    evidence_sha=compare.sha(args.packing_evidence)
    prereg=compare.ROOT/'results/laya/native_hybrid_compare_v1_prereg_desktop2_20261016.json'
    _,paths,bundle,fixtures=compare.load_registration(prereg)
    folder=paths.base/'compact_pack_wide_v2_m128'
    wide=json.loads((folder/'build.json').read_text());parent=json.loads((paths.attention_candidate/'build.json').read_text())
    if wide['sources']!=op.cw.sources(): raise ValueError('wide attention source changed')
    init=pack_sweep.init_proof(paths.attention_candidate/'prj/cdo_main/main_aie_cdo_init.bin',
        folder/'prj/cdo_main/main_aie_cdo_init.bin',parent['audits'],wide['audits'])
    walk=transport_walk.verify_pair((paths.attention_candidate/'insts.bin').read_bytes(),(folder/'insts.bin').read_bytes(),bundle.plan)
    compare.emit('PACK_LAYER_IDENTITY',dict(runner_sha256=compare.sha(Path(__file__)),walk_sha256=compare.sha(Path(transport_walk.__file__)),
        packing_evidence_sha256=evidence_sha,wide_build_sha256=compare.sha(folder/'build.json'),init=init,transport=walk,
        scope='one global/local layer at a time; complete fixture outputs, no accuracy or performance claim'))
    if not compare.hybrid.idle(): raise ValueError('device occupied')
    calls=[]
    def forbidden(*unused,**kw): calls.append(True);raise AssertionError('CPU neural/oracle call')
    functions=[(compare.ort.InferenceSession,'run'),(n.cr,'references'),(n.embedded.mask_reference,'extract'),
        (p.norm,'oracle'),(p.attention,'oracle'),(n.action,'reference'),(n.action,'verify'),
        (n.action.s,'reference'),(n.action.s,'verify'),(n.action.s.second,'reference'),(n.action.s.second.post,'reference')]
    common=(bundle,paths.mlp_folder,paths.attention_folder,paths.mlp_candidate,paths.attention_candidate,
        paths.compact_folder,fixtures,paths.head_folder)
    refs=[]
    with ExitStack() as stack:
        for module,name in functions:stack.enter_context(patch.object(module,name,forbidden))
        with op.Session(*common,fast_folder=paths.fast_folder,mlp_kind='lookup') as session:
            for item in fixtures:
                refs.append({key:value.copy() for key,value in session.run(item['ids'],item['markers'],item['qtype']).items()})
        session=None;gc.collect()
        if not compare.hybrid.idle():raise ValueError('reference context cleanup')
        with Session(*common,fast_folder=paths.fast_folder,mlp_kind='lookup',pack_folder=folder) as session:
            representatives={spec['mode']:index for index,spec in reversed(list(enumerate(bundle.meta['chain']['spec'])))}
            for selected in [(),*((index,) for index in representatives.values())]:
                for index,item in enumerate(fixtures):
                    result=session.run(item['ids'],item['markers'],item['qtype'],pack_layers=selected)
                    exact={key:bool(compare.np.array_equal(value.view(compare.np.uint32),refs[index][key].view(compare.np.uint32))) for key,value in result.items()}
                    compare.emit('PACK_LAYER_CHECK',dict(layers=selected,fixture=index,exact=exact,contexts=len(session.states),dispatches=session.last_stats['dispatches']))
                    if not all(exact.values()) or len(session.states)!=5 or session.last_stats['dispatches']!=66:
                        raise ValueError('layer integration changed outputs or execution policy')
        session=None;gc.collect()
    if calls or not compare.hybrid.idle(): raise ValueError('CPU neural call or final context cleanup')
    compare.load_registration(prereg)
    if compare.sha(args.packing_evidence)!=evidence_sha: raise ValueError('qualification evidence changed')
    compare.emit('PACK_LAYER_RESULT',dict(cpu_neural_oracle_calls=len(calls),all_fixture_bits_exact=True,idle_after_close=True,
        scope='baseline plus one global/local layer only; all-layer and boundary qualification pending'))


if __name__=='__main__':main()

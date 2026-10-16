# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""ABBA host-packing comparison on one unchanged, resident attention program."""
import argparse
import gc
import json
from pathlib import Path
from statistics import mean
import time

import numpy as np
import probe as p
from shared import pack_native, pack


def register(source,out):
    contract=json.loads(source.read_text())
    contract['builds']={name:contract['builds'][name] for name in ('base_global_h16_v512','host_global_h16_v512')}
    contract['sources'][Path(__file__).name]=p.sha(__file__)
    contract['paired_plan']=dict(order='ABBA',blocks=25,samples_per_arm=50,
        modes=['global_512','local_512'],warmups=5,cpu_reps=25,
        target='at least .15ms lower packing; report smaller savings without promoting target',
        timing='same loaded NPU program and BOs, fresh input packing each sample, sync and detached output included',
        correctness='three seeds, four distributions, two poisons, both packers and masks; immutable inputs and guards')
    if out.exists(): raise FileExistsError(out)
    p.dump(out,contract)
    print('preregistration_sha256',p.sha(out),flush=True)


def run(builds,masks,registration):
    contract=json.loads(registration.read_text())
    for name,digest in contract['sources'].items():
        if p.sha(p.HERE/name)!=digest: raise ValueError('source changed: '+name)
    artifacts={}
    for label in ('base','host'):
        name=label+'_global_h16_v512'; directory=builds/name
        meta=json.loads((directory/'build.json').read_text())
        if meta!=contract['builds'][name]: raise ValueError('unregistered artifact')
        for name,digest in meta['dependencies'].items():
            if contract['sources'].get(name)!=digest: raise ValueError('build dependency changed')
        if meta['kernel_sha256']!=contract['sources'][meta['kernel_source']]: raise ValueError('kernel changed')
        if meta['masks_sha256']!=p.sha(masks): raise ValueError('mask changed')
        for filename,key in (('final.xclbin','xclbin_sha256'),('insts.bin','instructions_sha256'),('attention_pack.dll','pack_dll_sha256')):
            if p.sha(directory/filename)!=meta[key]: raise ValueError('binary changed')
        artifacts[label]=(directory,meta)
    base,host=artifacts['base'][1],artifacts['host'][1]
    if base['fast_packing'] or not host['fast_packing']: raise ValueError('incorrect controls')
    for key in ('instructions_sha256','kernel_sha256','core_audits','input_words','outputs'):
        if base[key]!=host[key]: raise ValueError('device configuration differs: '+key)
    print('PREREGISTRATION',json.dumps(contract,separators=(',',':')),flush=True)
    print('preregistration_sha256',p.sha(registration),flush=True)
    print('SAME_DEVICE_CONFIG',json.dumps(dict(instructions=base['instructions_sha256'],core_audits_equal=True)),flush=True)
    from ignite_xdna.runtime.driver import XrtSiliconHarness
    import ignite_xdna
    assert Path(ignite_xdna.__file__).resolve().is_relative_to(p.ROOT/'src')
    if 'No hardware contexts running' not in p.command(['C:/Windows/System32/AMD/xrt-smi.exe','examine','-r','aie-partitions'],p.ROOT):
        raise RuntimeError('NPU busy')
    with np.load(masks) as archive, XrtSiliconHarness() as h:
        ib=bi=bo=imap=omap=output=None
        try:
            h.load_xclbin(str(artifacts['base'][0]/'final.xclbin'))
            ib,nbytes=h.create_instruction_bo(str(artifacts['base'][0]/'insts.bin'))
            bi=h.create_host_bo(base['input_words']*4,3); bo=h.create_host_bo(base['outputs']*p.OUT*4,4)
            imap=np.frombuffer(bi.map(),np.uint32); omap=np.frombuffer(bo.map(),np.uint32)
            to=h.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
            fr=h.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
            for mode in ('global','local'):
                mask=archive[mode+'_512']
                def execute(label,q,k,v,poison):
                    options={'mask_params':(512,mode=='local')} if label=='host' else {}
                    start=time.perf_counter()
                    pack_native(q,k,v,mask,poison,artifacts[label][0]/'attention_pack.dll',imap,**options)
                    packed=time.perf_counter()
                    bi.sync(to)
                    ds=time.perf_counter()
                    result,status=h.dispatch_kernel(ib,nbytes,bi,bo,timeout_ms=5000)
                    de=time.perf_counter()
                    if 'COMPLETED' not in str(status): raise RuntimeError('dispatch failed; do not retry')
                    bo.sync(fr)
                    got=omap.reshape(-1,p.OUT)[:,16:-16].copy().view(np.float32).reshape(16,512,64)
                    elapsed=(time.perf_counter()-start)*1e3
                    return got,dict(end_to_end_ms=elapsed,packing_ms=(packed-start)*1e3,dispatch_ms=(de-ds)*1e3)
                for seed in p.SEEDS:
                    for kind in contract['cases']:
                        q,k,v=p.case(16,seed,kind); ref=p.oracle(q,k,v,mask)
                        previous=None
                        for label in ('base','host'):
                            for poison in contract['poisons']:
                                omap.fill(poison); bo.sync(to)
                                # Obtain reference bytes before dispatch with an independent layout oracle.
                                reference_input=pack(q,k,v,mask,poison).tobytes()
                                got,_=execute(label,q,k,v,poison)
                                bi.sync(fr)
                                immutable=imap.tobytes()==reference_input
                                output=omap.reshape(-1,p.OUT)
                                guards=bool(np.all(output[:,:16]==poison) and np.all(output[:,-16:]==poison))
                                written=bool(np.all(output[:,16:-16]!=poison))
                                independent=previous is None or np.array_equal(previous,got)
                                verdict=p.numeric(got,ref)
                                ok=verdict['ok'] and immutable and guards and written and independent
                                print('PAIR_CHECK',json.dumps(dict(mode=mode,seed=seed,kind=kind,packer=label,
                                      poison=hex(poison),inputs_immutable=immutable,guards=guards,written=written,
                                      independent=independent,**verdict,verdict='PASS' if ok else 'FAIL')),flush=True)
                                if not ok: raise ValueError('paired correctness failed; timing withheld')
                                previous=got
                for _ in range(contract['paired_plan']['warmups']): execute('host',q,k,v,poison)
                samples={'base':[],'host':[]}; deltas=[]
                for block in range(contract['paired_plan']['blocks']):
                    rows={'base':[],'host':[]}
                    for label in ('base','host','host','base'):
                        got,row=execute(label,q,k,v,poison)
                        if not p.numeric(got,ref)['ok']: raise ValueError('timed output incorrect')
                        samples[label].append(row); rows[label].append(row['end_to_end_ms'])
                    deltas.append(mean(rows['base'])-mean(rows['host']))
                cpu=p.cpu_session(16,mask); feed=dict(q=q,k=k,v=v)
                for _ in range(3): cpu.run(None,feed)
                cpu_times=[]
                for _ in range(contract['paired_plan']['cpu_reps']):
                    start=time.perf_counter(); result=cpu.run(None,feed)[0]; cpu_times.append((time.perf_counter()-start)*1e3)
                if not p.numeric(result,ref)['ok']: raise ValueError('CPU baseline incorrect')
                delta=mean(deltas); sem=float(np.std(deltas,ddof=1)/np.sqrt(len(deltas)))
                report=dict(mode=mode,samples=samples,cpu_ms=cpu_times,paired_deltas_ms=deltas,
                    paired_delta_mean_ms=delta,paired_delta_ci95_normal_approx=[delta-1.96*sem,delta+1.96*sem],
                    base_mean_ms=mean(x['end_to_end_ms'] for x in samples['base']),
                    host_mean_ms=mean(x['end_to_end_ms'] for x in samples['host']),
                    base_pack_mean_ms=mean(x['packing_ms'] for x in samples['base']),
                    host_pack_mean_ms=mean(x['packing_ms'] for x in samples['host']))
                print('HOST_PAIR_RESULT',json.dumps(report),flush=True)
                del execute
        finally:
            imap=omap=ib=bi=bo=output=None
            gc.collect()
    witness=p.command(['C:/Windows/System32/AMD/xrt-smi.exe','examine','-r','aie-partitions'],p.ROOT)
    print(witness,flush=True)
    if 'No hardware contexts running' not in witness: raise RuntimeError('context remained after close')
    print('SUITE_RESULT',json.dumps(dict(verdict='PASS',failed=[])),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); sub=ap.add_subparsers(dest='cmd',required=True)
    reg=sub.add_parser('register'); reg.add_argument('--from-prereg',type=Path,required=True); reg.add_argument('--out',type=Path,required=True)
    run_parser=sub.add_parser('run'); run_parser.add_argument('--builds',type=Path,required=True)
    run_parser.add_argument('--masks',type=Path,required=True); run_parser.add_argument('--prereg',type=Path,required=True)
    a=ap.parse_args()
    if a.cmd=='register': register(a.from_prereg,a.out)
    else: run(a.builds.resolve(),a.masks,a.prereg)

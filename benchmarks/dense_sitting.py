"""Serialize dense-model evidence jobs through the bounded research wrapper."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASH = 'C:/Program Files/Git/bin/bash.exe'
IRON_PYTHON = Path.home()/'miniforge3/envs/mlir-aie-iron/python.exe'


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('phase',choices=['inputs','baseline','bench','verify'])
    ap.add_argument('--tag',required=True)
    ap.add_argument('--families',nargs='+',default=['bisenetv2','modnet_cut'])
    ap.add_argument('--backends',nargs='+',default=['cpu','amd','ignite'])
    ap.add_argument('--runs',type=int,default=2)
    ap.add_argument('--pins-tag',default='20260919')
    ap.add_argument('--container-tag',default='dense')
    args = ap.parse_args()
    for family in args.families:
        for run in range(args.runs if args.phase == 'bench' else 1):
            for backend in args.backends:
                if args.phase == 'verify' and backend != 'ignite':
                    continue
                tag = f'{args.phase}_{family}_{backend}_{args.tag}_{run+1}'
                log = f'results/dense/{tag}.log'
                python = str(IRON_PYTHON) if backend in ('ignite','ignite_initial','cpu_current') else 'python'
                cmd = [BASH,'scripts/research-lowlevel.sh','--log',log,
                       '--seconds','1200','--rss-gib','10','--wait-clear','60']
                if args.phase != 'inputs':
                    cmd += ['--npu']
                if args.phase != 'bench':
                    cmd += ['--checks-only']
                cmd += ['--',python,'benchmarks/dense_compare.py',args.phase,'--family',family,
                        '--pins',f'results/dense/{family}_pins_{args.pins_tag}.json',
                        '--artifacts',f'build/dense_validation/{args.tag}',
                        '--backend','ignite' if backend == 'ignite_initial' else backend,
                        '--container',f'build/{family}_{"dense" if backend == "ignite_initial" else args.container_tag}.ignite']
                if backend == 'amd' and args.phase == 'baseline':
                    cmd += ['--fresh']
                print('START',tag,flush=True)
                result = subprocess.run(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                        text=True,encoding='utf-8',errors='replace',timeout=1260)
                print('END',tag,'returncode',result.returncode,flush=True)
                for line in result.stdout.splitlines():
                    if line.startswith(('BENCHMARK','VERDICT','RESEARCH_PROCESS_RESULT','RESEARCH_STOP')):
                        label,_,payload = line.partition(' ')
                        value = json.loads(payload)
                        value.pop('samples_ms',None)
                        print(label,json.dumps(value),flush=True)
                if result.returncode:
                    print(result.stdout[-5000:],flush=True)
                    return result.returncode
    return 0


if __name__ == '__main__':
    sys.exit(main())

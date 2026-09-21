"""Summarize new dense evidence logs without treating unlabeled agreement as accuracy."""
import argparse
import json
from pathlib import Path
import numpy as np


def records(path):
    rows = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        kind,_,payload = line.partition(' ')
        if payload.startswith('{'):
            try:
                value = json.loads(payload)
            except json.JSONDecodeError:
                continue
            rows.setdefault(kind,[]).append(value)
    return rows


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('logs',nargs='*')
    ap.add_argument('--artifacts',type=Path)
    ap.add_argument('--inputs-tag')
    ap.add_argument('--sitting-tag')
    args = ap.parse_args()
    if args.inputs_tag:
        for family in ('bisenetv2','modnet_cut'):
            paths = [Path(f'results/dense/inputs_{family}_{backend}_{args.inputs_tag}_1.log')
                     for backend in ('cpu','cpu_current')]
            values = [{r['image']:r for r in records(p)['INPUT']} for p in paths]
            good = values[0] == values[1] and len(values[0]) == 50
            print(json.dumps(dict(family=family,input_parity=good,images=len(values[0]),logs=[str(p) for p in paths])))
            if not good:
                raise SystemExit('input tensor hashes differ')
    if args.sitting_tag:
        for family in ('bisenetv2','modnet_cut'):
            rows = {}
            for backend in ('cpu','cpu_current','amd','ignite'):
                rows[backend] = []
                for run in (1,2):
                    path = Path(f'results/dense/bench_{family}_{backend}_{args.sitting_tag}_{run}.log')
                    row = records(path)
                    assert row['RESEARCH_PROCESS_RESULT'][-1]['timing_eligible'],path
                    timing = row['BENCHMARK'][0]
                    assert timing['frames'] == 500 and timing['warmup'] == 50
                    timing.pop('samples_ms',None)
                    timing['output_exact'] = row['BENCHMARK_OUTPUT'][0]['oracle']['exact']
                    timing['log'] = str(path)
                    rows[backend].append(timing)
            def beats(backend):
                return all(n['mean_ms'] <= .9*c['mean_ms'] and n['p95_ms'] <= c['p95_ms']
                           for n,c in zip(rows['ignite'],rows[backend]))
            print(json.dumps(dict(family=family,rows=rows,
                cpu_acceptance=beats('cpu') and beats('cpu_current'),
                amd_speed_win=all(n['mean_ms'] < a['mean_ms'] for n,a in zip(rows['ignite'],rows['amd'])))))
    if args.artifacts:
        for family in ('bisenetv2','modnet_cut'):
            cpu = sorted(p for p in args.artifacts.glob(f'{family}_cpu_*.npy') if p.stem[-4:].isdigit())
            for backend in ('amd','ignite'):
                exact,maximum = 0,0.
                for p in cpu:
                    a = np.load(p)
                    b = np.load(p.with_name(p.name.replace('_cpu_',f'_{backend}_')))
                    exact += bool(np.array_equal(a,b))
                    maximum = max(maximum,float(np.max(np.abs(a-b))))
                print(json.dumps(dict(family=family,backend=backend,images=len(cpu),
                                      pinned_cpu_exact=exact,pinned_cpu_max_abs=maximum)),flush=True)
                for suffix in ('oracle','fp32'):
                    pairs = [(p.with_stem(p.stem+'_'+suffix),p.with_name(
                        p.name.replace('_cpu_',f'_{backend}_')).with_stem(
                            p.stem.replace('_cpu_',f'_{backend}_')+'_'+suffix)) for p in cpu]
                    if not all(a.exists() and b.exists() for a,b in pairs):
                        continue
                    exact,maximum = 0,0.
                    for a,b in pairs:
                        a,b = np.load(a),np.load(b)
                        exact += bool(np.array_equal(a,b))
                        maximum = max(maximum,float(np.max(np.abs(a-b))))
                    print(json.dumps(dict(family=family,backend=backend,reference=suffix,images=len(pairs),
                        cross_environment_exact=exact,max_abs=maximum)))
    for pattern in args.logs:
        for path in sorted(Path('.').glob(pattern)):
            rows = records(path)
            summary = {'log':path.as_posix()}
            agreements = rows.get('AGREEMENT',[])
            if agreements:
                summary.update(images=len(agreements),cpu_exact=sum(r['cpu']['exact'] for r in agreements),
                    cpu_max_abs=max(r['cpu']['max_abs'] for r in agreements),
                    fp32_mean_abs=float(np.mean([r['fp32']['mean_abs'] for r in agreements])),
                    fp32_post_mean_abs=float(np.mean([r['postprocessed_fp32']['mean_abs'] for r in agreements])))
                if 'fp32_pixel_agreement' in agreements[0]:
                    summary['fp32_pixel_agreement'] = float(np.mean([r['fp32_pixel_agreement'] for r in agreements]))
                if 'cut_fp32_vs_stock_postprocessed' in agreements[0]:
                    summary['cut_fp32_vs_stock_alpha_mad'] = float(np.mean([
                        r['cut_fp32_vs_stock_postprocessed']['mean_abs'] for r in agreements]))
                if 'optimized_cpu_vs_oracle' in agreements[0]:
                    summary['optimized_cpu_exact'] = sum(r['optimized_cpu_vs_oracle']['exact'] for r in agreements)
                    summary['optimized_cpu_max_abs'] = max(r['optimized_cpu_vs_oracle']['max_abs'] for r in agreements)
            for label in ('BENCHMARK','VERDICT','RESEARCH_PROCESS_RESULT'):
                if label in rows:
                    summary[label] = [{k:v for k,v in r.items() if k != 'samples_ms'} for r in rows[label]]
            if rows.get('AMD_REPORT'):
                summary['placement'] = rows['AMD_REPORT'][0]['report'].get('deviceStat')
            print(json.dumps(summary))


if __name__ == '__main__':
    main()

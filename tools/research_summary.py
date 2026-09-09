"""Summarize immutable research logs and archive their small numerical evidence.

Run through research-lowlevel.sh --checks-only. An archive is created exclusively,
never overwritten. Calibration ONNX weights are not copied into the evidence ZIP.
"""
import argparse
import glob
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np

from research_process import ROOT, scrub


def records(path):
    result = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        label, sep, value = line.partition(' ')
        if sep and value.startswith(('{', '[')):
            try:
                result.setdefault(label, []).append(json.loads(value))
            except json.JSONDecodeError:
                continue
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--logs', nargs='+', required=True)
    ap.add_argument('--archive', type=Path, required=True)
    a = ap.parse_args()
    paths = sorted({Path(p).resolve() for pattern in a.logs for p in glob.glob(pattern)})
    if not paths or a.archive.exists():
        ap.error('need matching logs and a new archive')
    summary, archive_items, arithmetic = [], {}, []
    for path in paths:
        if not path.is_relative_to(ROOT / 'results'):
            raise ValueError('log escaped results')
        rs = records(path)
        row = {'log': path.relative_to(ROOT).as_posix(), 'log_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        process = rs.get('RESEARCH_PROCESS_RESULT', [{}])[-1]
        row['completed'] = process.get('exit_code') == 0 and process.get('stop') is None
        row['timing_eligible'] = process.get('timing_eligible', False)
        row['foreign_npu_observed'] = any(r['foreign_pids'] for r in rs.get('NPU_OWNERSHIP', []))
        for label in ('PLACEMENT_RESULT', 'GEMM_PLACEMENT_RESULT', 'ALPHABET_MODEL_RESULT', 'ALPHABET_RESULT'):
            if label in rs:
                row[label] = rs[label][-1]
        if 'ARITHMETIC_RESULT' in rs:
            row['ARITHMETIC_RESULT'] = rs['ARITHMETIC_RESULT'][-1]
        platforms = rs.get('RESEARCH_PLATFORM', [])
        if platforms and row['completed']:
            command = platforms[-1]['command']
            if '--out' in command:
                out = (ROOT / command[command.index('--out')+1]).resolve()
                if not out.is_relative_to(ROOT / 'scratch'):
                    raise ValueError('run output escaped scratch')
                stem = path.stem
                if out.suffix == '.onnx':
                    candidates = [Path(str(out)+'.quant.json')]
                else:
                    candidates = list(out.glob('*'))
                for artifact in candidates:
                    if artifact.suffix not in ('.json', '.npz', '.onnx', '.txt'):
                        continue
                    data = artifact.read_bytes()
                    if artifact.suffix in ('.json', '.txt'):
                        data = scrub(data.decode('utf-8')).encode('utf-8')
                    archive_items[stem+'/'+artifact.name] = data
                    if artifact.suffix == '.txt' and 'elf' in artifact.name:
                        raw = ''.join(re.findall(r'^\s*[0-9a-fA-F]+:\s+((?:[0-9a-fA-F]{2}\s+)+)', data.decode('utf-8'), re.M))
                        if raw:
                            binary = bytes.fromhex(raw)
                            archive_items[stem+'/'+artifact.name+'.function.bin'] = binary
                if 'ARITHMETIC_RESULT' in rs:
                    r = json.loads((out/'result.json').read_text(encoding='utf-8'))
                    rawpath = out / 'raw_outputs.npz'
                    if hashlib.sha256(rawpath.read_bytes()).hexdigest() != rs['ARITHMETIC_RESULT'][-1]['raw_outputs_sha256']:
                        raise AssertionError('numerical evidence hash mismatch')
                    with np.load(rawpath) as arrays:
                        if r['status'] == 'placed' and not row['foreign_npu_observed']:
                            names = [n.removeprefix('npu_') for n in arrays.files if n.startswith('npu_')]
                            mismatches = sum(np.count_nonzero(arrays['npu_'+n] != arrays['reference_'+n]) for n in names)
                            recorded = sum(v['mismatches'] for v in r['comparisons']['npu'].values())
                            assert mismatches == recorded
                            row['npu_mismatches_recounted'] = int(mismatches)
                            row['npu_elements'] = sum(arrays['npu_'+n].size for n in names)
                            row['cpu_mismatches'] = {p: sum(v['mismatches'] for v in comp.values())
                                for p, comp in r['comparisons'].items() if p.startswith('cpu_')}
                            arithmetic.append((r, out, row))
        summary.append(row)
        print('RESEARCH_CASE', json.dumps(row), flush=True)
    common = None
    for r, _, _ in arithmetic:
        candidates = {(c['input_round'], c['bias_round'], c['output_round']) for c in r['heldout_survivors']}
        common = candidates if common is None else common & candidates
    permutations = []
    for r, out, _ in arithmetic:
        if not r['reverse_channels']:
            continue
        matches = [(rr, pp) for rr, pp, _ in arithmetic if not rr['reverse_channels'] and
                   all(rr[k] == r[k] for k in ('channels', 'shift_cut', 'shift_bias'))]
        if len(matches) != 1:
            raise ValueError('permutation has no unique baseline')
        with np.load(out/'raw_outputs.npz') as actual, np.load(matches[0][1]/'raw_outputs.npz') as baseline:
            names = [n for n in actual.files if n.startswith('npu_')]
            assert all(np.array_equal(actual[n], baseline[n]) for n in names)
            permutations.append({'channels': r['channels'], 'identical_npu_elements': sum(actual[n].size for n in names)})
    final = {'cases': len(summary), 'completed': sum(r['completed'] for r in summary),
             'timing_eligible_cases': sum(r['timing_eligible'] for r in summary),
             'placed_arithmetic_models': len(arithmetic),
             'arithmetic_elements': sum(row['npu_elements'] for _, _, row in arithmetic),
             'arithmetic_mismatches_to_onnx': sum(row['npu_mismatches_recounted'] for _, _, row in arithmetic),
             'global_arithmetic_candidates': sorted(common or []), 'permutation_checks': permutations}
    archive_items['manifest.json'] = json.dumps({'summary': final, 'cases': summary}, indent=2).encode('utf-8')
    with zipfile.ZipFile(a.archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in sorted(archive_items.items()):
            z.writestr(name, data)
    final['archive'] = str(a.archive)
    final['archive_sha256'] = hashlib.sha256(a.archive.read_bytes()).hexdigest()
    final['archive_bytes'] = a.archive.stat().st_size
    print('RESEARCH_SUMMARY', json.dumps(final), flush=True)


if __name__ == '__main__':
    main()

"""REG_0 sensitivity map: which bytes of the DPU parameter memory change the fixture Conv's output?

The xmodel's REG_0 (reg_id_to_parameter_value, 1891 B) is the parameter memory the VitisAI DPU overlay
loads (tools/dpu_mc_fixtures.py). This tool:
- compiles the 32-channel shift_cut 8 / shift_bias 0 fixture of tools/xint8_arithmetic_probe.py once;
- for each byte offset in --lo..--hi and each XOR mask in --masks, rewrites that one REG_0 byte in the CACHED
  xmodel, reloads the session (the EP must not recompile or rewrite it), and runs the 7 cases;
- records the arm as "same" (output codes bit-equal to the unpatched base), "changed" (with the count of
  differing codes) or an error.

**Stop rule.** The first error or exception stops the sweep (some bytes are probably DMA sizes or
addresses): the original xmodel is restored and the stop is recorded. A sweep resumes only through
--resume-from, on the user's OK.

**Restore.** The original xmodel is restored at the end, and copied to --out first, so a killed run can be
restored by hand.

This is EVIDENCE, not a witness. Several overlay handlers can read the same byte, and a byte can be read
without changing these 7 outputs. Scoring against handler read sets is a separate, pre-registered step.
What has been run: see results/aie/dpu_reg0_sensitivity_desktop2_*.log; until such a log exists, nothing.

  python tools/dpu_reg0_sensitivity.py --out RUN_DIR --lo 0 --hi 255 --masks 0x01,0x80
"""
import argparse
import gc
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from npu.ep_report import read_report  # noqa: E402
from npu.paths import CACHE_DIR, RESNET_CACHE_KEY  # noqa: E402
from npu.session import build_session, clear_cache, resolve_xclbin  # noqa: E402
import xint8_arithmetic_probe as probe  # noqa: E402
from dpu_mc_fixtures import reg0_span  # noqa: E402

XRT_SMI = 'C:/Windows/System32/AMD/xrt-smi.exe'
SC, SB, CHANNELS = 8, 0, 32


def idle():
    out = subprocess.run([XRT_SMI, 'examine', '-r', 'aie-partitions'], capture_output=True, text=True,
                         timeout=60).stdout
    return 'No hardware contexts running' in out


def sha(b):
    return hashlib.sha256(b).hexdigest()


def run_cases(s, cases):
    outs = {}
    for name, x in cases.items():
        first = probe.output_codes(s.run(None, {'input': x})[0], SC)
        again = probe.output_codes(s.run(None, {'input': x})[0], SC)
        if not np.array_equal(first, again):
            raise RuntimeError('nondeterministic repeated output on ' + name)
        outs[name] = first
    return outs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--lo', type=lambda s: int(s, 0), default=0)
    ap.add_argument('--hi', type=lambda s: int(s, 0), default=0xFF, help='inclusive')
    ap.add_argument('--masks', default='0x01,0x80')
    ap.add_argument('--resume-from', type=lambda s: int(s, 0), help='skip offsets below this (after a stop)')
    a = ap.parse_args()
    masks = [int(m, 0) for m in a.masks.split(',')]
    if a.out.exists():
        ap.error('--out must be new')
    cache = (CACHE_DIR / RESNET_CACHE_KEY).resolve()
    if cache.parent != ROOT.resolve():
        raise SystemExit('EP cache escaped this checkout: %s' % cache)
    if not idle():
        raise SystemExit('NPU not idle: refusing (a foreign context is contention, not a finding)')
    a.out.mkdir(parents=True)
    g, w, b = probe.model(CHANNELS, SC, SB, False)
    onnx_path = a.out / 'fixture.onnx'
    g.save(onnx_path)
    cases = probe.fixtures(CHANNELS, False)
    xclbin = str(resolve_xclbin())

    clear_cache(RESNET_CACHE_KEY)
    s = build_session(onnx_path, 'npu', RESNET_CACHE_KEY, xclbin, log_severity=3)
    ep = read_report(cache / 'vitisai_ep_report.json').summary()
    base = run_cases(s, cases)
    del s
    gc.collect()
    xms = sorted(cache.glob('*.xmodel'))
    if len(xms) != 1:
        raise SystemExit('expected one cached xmodel, found %d' % len(xms))
    xm = xms[0]
    original = xm.read_bytes()
    shutil.copy2(xm, a.out / 'original.xmodel')
    i0, n = reg0_span(original)
    if a.hi >= n:
        raise SystemExit('--hi beyond REG_0 (%d B)' % n)
    cache_files = sorted(p.name for p in cache.rglob('*') if p.is_file())
    report = {'fixture': {'channels': CHANNELS, 'shift_cut': SC, 'shift_bias': SB}, 'reg0_len': n,
              'reg0_head_hex': original[i0:i0 + 0x100].hex(), 'xmodel_sha256': sha(original),
              'ep': {'npu': ep['npu'], 'total': ep['total'], 'conv_on_npu': ep['ops_by_device'].get('NPU:Conv', 0)},
              'masks': masks, 'arms': [], 'stopped': None}
    print('BASE: NPU:Conv %d, REG_0 %d B at xmodel offset %d' % (report['ep']['conv_on_npu'], n, i0), flush=True)
    t0 = time.time()
    try:
        for off in range(a.lo, a.hi + 1):
            if a.resume_from is not None and off < a.resume_from:
                continue
            for m in masks:
                if not idle():
                    report['stopped'] = 'not idle before 0x%02x^0x%02x' % (off, m)
                    raise StopIteration
                patched = bytearray(original)
                patched[i0 + off] ^= m
                xm.write_bytes(bytes(patched))
                arm = {'off': off, 'mask': m, 'orig': original[i0 + off], 'new': patched[i0 + off]}
                try:
                    s = build_session(onnx_path, 'npu', RESNET_CACHE_KEY, xclbin, log_severity=3)
                    arm['xmodel_unchanged_by_ep'] = sha(xm.read_bytes()) == sha(bytes(patched))
                    arm['cache_file_set_unchanged'] = sorted(p.name for p in cache.rglob('*') if p.is_file()) == cache_files
                    arm['conv_on_npu'] = read_report(cache / 'vitisai_ep_report.json').summary()['ops_by_device'].get('NPU:Conv', 0)
                    if not (arm['xmodel_unchanged_by_ep'] and arm['cache_file_set_unchanged']):
                        arm['verdict'] = 'REJECTED'
                    else:
                        out = run_cases(s, cases)
                        diff = sum(int(np.count_nonzero(out[k] != base[k])) for k in base)
                        arm['diff_codes'] = diff
                        arm['verdict'] = 'same' if diff == 0 else 'changed'
                    del s
                except Exception as e:  # noqa: BLE001 -- an error is the finding, and it stops the sweep
                    # The whole text: vaip prefixes every error with a long frame signature, so a short
                    # cut keeps only that and loses the cause (the 2026-10-12 220151Z/221209Z stops).
                    arm['verdict'] = 'ERROR: %s' % type(e).__name__
                    arm['error_text'] = str(e)
                gc.collect()
                report['arms'].append(arm)
                print('0x%02x ^ 0x%02x (%02x->%02x): %s%s' % (off, m, arm['orig'], arm['new'], arm['verdict'],
                      (' (%d codes)' % arm['diff_codes']) if arm.get('verdict') == 'changed' else ''), flush=True)
                if 'error_text' in arm:
                    print('  full error text:\n' + arm['error_text'], flush=True)
                if arm['verdict'].startswith(('ERROR', 'REJECTED')):
                    report['stopped'] = 'stop rule at 0x%02x ^ 0x%02x: %s' % (off, m, arm['verdict'])
                    raise StopIteration
    except StopIteration:
        pass
    finally:
        xm.write_bytes(original)
        report['restored'] = sha(xm.read_bytes()) == sha(original)
        report['elapsed_s'] = round(time.time() - t0, 1)
        report['idle_after'] = idle()
        (a.out / 'report.json').write_text(json.dumps(report, indent=1))
        np.savez(a.out / 'base_outputs.npz', **base)
    print('arms %d, stopped: %s, restored %s, idle after %s, %.0f s'
          % (len(report['arms']), report['stopped'], report['restored'], report['idle_after'], report['elapsed_s']), flush=True)
    return 1 if report['stopped'] else 0


if __name__ == '__main__':
    sys.exit(main())

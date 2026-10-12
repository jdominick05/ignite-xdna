"""Does the VitisAI DPU execute a shift_cut outside the producer's [0, 16] contract?

The EP refuses to COMPILE a Conv whose shift_cut is outside [0, 16] (sigma = shift_cut + 14,
BENCHMARKS "Theorem 1's sigma window"), so the edges have never run on the DPU. mc_code does
not carry the shift (tools/dpu_mc_fixtures.py). It sits in two places in the compiled .xmodel:
  reg0          byte 0x7C of the subgraph's REG_0 (reg_id_to_parameter_value), the parameter
                memory the DPU loads; byte 0x7B is shift_bias << 2 | 1. DEFAULT.
  param_buffer  the weight tensor's attr varint shift_cut << 8 | shift_bias << 2 | 1. The
                device ignores it: patched alone, in-contract controls read the base shift
                (MEASURED 2026-10-12, run 1, kept as the record of that dead end).
This test compiles an in-contract fixture, rewrites the chosen site in the CACHED xmodel, and
reloads the cache WITHOUT --fresh, so the EP's compile-time refusal never runs.

Arms, all in one process, with the cache under this checkout (asserted, as in the probe):
  base      compile shift_cut --base fresh and check it against the exact oracle
  control   patch to in-contract values (--controls) whose correct answer is known
  patched   patch to out-of-contract values (--patches)
  restore   reload the unpatched xmodel at the end, which must equal base again
For every reload, the xmodel's sha256 is checked after the session builds. If the EP
recompiled or rewrote the cache, the arm is reported REJECTED and its outputs are not
scored.

Each arm's NPU output codes (dequantized with the base fixture's output scale, which the
patch does not touch) are compared exactly against:
  H_exact  predict(shift_cut = patched value), output rounding half_up (measured on the
           in-contract fixtures, results/aie/dpu_sigma_field_fixtures_*.log)
  H_base   predict(shift_cut = base): the patch had no effect, or was clamped back
A patched value only counts as "executes exactly" if H_exact holds on every case.

The device runs a hand-edited binary. Run with an outer timeout, from an idle NPU.
  python tools/dpu_sigma_patch_test.py --out ../../../scratch/dpu_re/patch --base 16 \
      --controls 15,1 --patches 17,18,20,24,31,45,46
"""
import argparse
import gc
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from npu.paths import CACHE_DIR, RESNET_CACHE_KEY  # noqa: E402
from npu.session import build_session, clear_cache, resolve_xclbin  # noqa: E402
from npu.ep_report import read_report  # noqa: E402
import xint8_arithmetic_probe as probe  # noqa: E402
from dpu_mc_fixtures import REG0_SHIFT_CUT, reg0_span  # noqa: E402

XRT_SMI = 'C:/Windows/System32/AMD/xrt-smi.exe'


def enc(v):
    out = []
    while True:
        b = v & 0x7F
        v >>= 7
        out.append(b | (0x80 if v else 0))
        if not v:
            return bytes(out)


def field(sc, sb=0):
    return enc(sc << 8 | sb << 2 | 1)


PRE, POST = bytes.fromhex('c4 88 01'), bytes.fromhex('80 0c')
def patch(original, site, base, v):
    """The xmodel with shift_cut rewritten from base to v at the chosen site."""
    if site == 'param_buffer':
        return original.replace(PRE + field(base) + POST, PRE + field(v) + POST)
    i, n = reg0_span(original)
    assert original[i + REG0_SHIFT_CUT] == base, original[i + REG0_SHIFT_CUT]
    b = bytearray(original)
    b[i + REG0_SHIFT_CUT] = v
    return bytes(b)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def idle():
    w = subprocess.run([XRT_SMI, 'examine', '-r', 'aie-partitions'], capture_output=True, text=True, timeout=60)
    return 'No hardware contexts running' in (w.stdout + w.stderr)


def run_cases(session, cases, base_sc):
    out = {}
    for name, x in cases.items():
        y = session.run(None, {'input': x})[0]
        out[name] = probe.output_codes(y, base_sc)
    return out


def score(out, cases, w, b, sc, sb):
    mism = 0
    maxd = 0
    for name, x in cases.items():
        ref = probe.predict(x, w, b, sc, sb, 'half_up')
        d = np.abs(out[name].astype(np.int16) - ref.astype(np.int16))
        mism += int(np.count_nonzero(d))
        maxd = max(maxd, int(d.max()))
    return mism, maxd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--base', type=int, default=16)
    ap.add_argument('--controls', default='15,1')
    ap.add_argument('--patches', default='17,18,20,24,31,45,46')
    ap.add_argument('--channels', type=int, default=32)
    ap.add_argument('--site', choices=('reg0', 'param_buffer'), default='reg0',
                    help='reg0: the REG_0 parameter byte (what the DPU loads); param_buffer: the '
                         'weight attr varint, which run 1 showed the device ignores')
    a = ap.parse_args()
    controls = [int(x) for x in a.controls.split(',') if x]
    patches = [int(x) for x in a.patches.split(',') if x]
    for v in controls + patches + [a.base]:
        if a.site == 'reg0' and not 0 <= v <= 255:
            ap.error('REG_0 holds shift_cut in one byte')
        if a.site == 'param_buffer' and len(field(v)) != len(field(a.base)):
            ap.error('shift_cut %d changes the varint length; only same-length patches are supported' % v)
    if a.out.exists():
        ap.error('--out must be new')
    a.out.mkdir(parents=True)
    cache = (CACHE_DIR / RESNET_CACHE_KEY).resolve()
    if cache.parent != ROOT.resolve():
        raise SystemExit('EP cache escaped this checkout: %s' % cache)
    if not idle():
        raise SystemExit('NPU not idle: refusing (a foreign context is contention, not a finding)')

    g, w, b = probe.model(a.channels, a.base, 0, False)
    onnx_path = a.out / 'fixture.onnx'
    g.save(onnx_path)
    cases = probe.fixtures(a.channels, False)
    xclbin = str(resolve_xclbin())
    report = {'site': a.site, 'base': a.base, 'controls': controls, 'patches': patches, 'arms': []}

    clear_cache(RESNET_CACHE_KEY)
    s = build_session(onnx_path, 'npu', RESNET_CACHE_KEY, xclbin, log_severity=3)
    ep = read_report(cache / 'vitisai_ep_report.json').summary()
    base_out = run_cases(s, cases, a.base)
    del s
    gc.collect()
    xms = sorted(cache.glob('*.xmodel'))
    if len(xms) != 1:
        raise SystemExit('expected one cached xmodel, found %d' % len(xms))
    xm = xms[0]
    original = xm.read_bytes()
    shutil.copy2(xm, a.out / 'original.xmodel')
    n_sites = (original.count(PRE + field(a.base) + POST) if a.site == 'param_buffer'
               else int(original[reg0_span(original)[0] + REG0_SHIFT_CUT] == a.base))
    cache_files = sorted(p.name for p in cache.rglob('*') if p.is_file())
    m, d = score(base_out, cases, w, b, a.base, 0)
    print('BASE shift_cut %d: NPU:Conv %s, sites %d, H_exact mismatches %d (max %d LSB)'
          % (a.base, ep['ops_by_device'].get('NPU:Conv', 0), n_sites, m, d), flush=True)
    report['base_arm'] = {'npu_conv': ep['ops_by_device'].get('NPU:Conv', 0), 'sites': n_sites,
                          'h_exact_mismatches': m, 'max_lsb': d, 'xmodel_sha256': hashlib.sha256(original).hexdigest()}
    if n_sites < 1 or m != 0 or ep['ops_by_device'].get('NPU:Conv', 0) != 1:
        raise SystemExit('base arm is not a clean placed fixture; nothing to patch')

    for kind, values in (('control', controls), ('patched', patches), ('restore', [a.base])):
        for v in values:
            if not idle():
                print('NPU not idle before arm %s %d: stopping' % (kind, v), flush=True)
                report['stopped'] = 'not idle'
                break
            patched = patch(original, a.site, a.base, v)
            xm.write_bytes(patched)
            want = hashlib.sha256(patched).hexdigest()
            arm = {'kind': kind, 'shift_cut': v, 'sigma': v + 14,
                   'field': field(v).hex(' ') if a.site == 'param_buffer' else '%02x' % v}
            try:
                s = build_session(onnx_path, 'npu', RESNET_CACHE_KEY, xclbin, log_severity=3)
                now_files = sorted(p.name for p in cache.rglob('*') if p.is_file())
                arm['xmodel_unchanged_by_ep'] = sha(xm) == want
                arm['cache_file_set_unchanged'] = now_files == cache_files
                arm['npu_conv'] = read_report(cache / 'vitisai_ep_report.json').summary()['ops_by_device'].get('NPU:Conv', 0)
                if not arm['xmodel_unchanged_by_ep']:
                    arm['verdict'] = 'REJECTED (EP rewrote the cached xmodel)'
                else:
                    out = run_cases(s, cases, a.base)
                    arm['h_exact'] = score(out, cases, w, b, v, 0)
                    arm['h_base'] = score(out, cases, w, b, a.base, 0)
                    np.savez_compressed(a.out / ('outputs_%s_sc%d.npz' % (kind, v)), **out)
                    arm['verdict'] = ('EXACT at shift_cut %d' % v if arm['h_exact'][0] == 0 else
                                      'NO EFFECT / base' if arm['h_base'][0] == 0 else 'NEITHER')
                del s
            except Exception as e:  # noqa: BLE001 -- a refusal or device error is the finding
                arm['verdict'] = 'ERROR: %s: %s' % (type(e).__name__, str(e)[:300])
            gc.collect()
            print('%-8s shift_cut %2d (sigma %2d) field [%s]: %s  %s' % (
                kind, v, v + 14, arm['field'], arm['verdict'],
                {k: arm[k] for k in ('h_exact', 'h_base', 'npu_conv', 'xmodel_unchanged_by_ep',
                                     'cache_file_set_unchanged') if k in arm}), flush=True)
            report['arms'].append(arm)
    xm.write_bytes(original)
    (a.out / 'report.json').write_text(json.dumps(report, indent=1))
    print('cache xmodel restored to the original bytes; NPU idle after: %s' % idle(), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

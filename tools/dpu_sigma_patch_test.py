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

An arm is `shift_cut` or `shift_cut:shift_bias`. At the reg0 site the shift_bias goes into byte
0x7B as shift_bias << 2 | 1 (the low two bits are left as compiled). A negative shift_cut is
written as its two's-complement byte. That it reads back as signed is a hypothesis, scored as
H_signed (= H_exact) against H_unsigned.

Every arm is also scored against named alternative models (--dry-run prints the table without
the device), so a pre-registration can name which outcomes each arm separates:
  exact          shift_cut and shift_bias applied as written, accumulator wide enough
  sat32, wrap32  the accumulator (in the oracle's units) saturates / wraps at 32 bits
  sb_clamp15     shift_bias clamped to the producer's 15
  sc_clamp<N>    shift_cut clamped to N
  sigma_mask5    sigma = shift_cut + 14 masked to 5 bits
  vsrs6          sigma masked to 6 bits and read as the AIE2 vsrs code (60..63 shift left 4..1)
  code6_sb4      FITTED on run 3: the shift_cut byte & 63 is the vsrs code (60..63 shift left by
                 4..1, saturating only when a one-bit shift would overflow, else wrapping to 8 bits),
                 and the effective shift_bias is shift_bias & 15
  sigma6_sb4     the rival reading: sigma = shift_cut + 14 wrapped to 6 bits, shift = sigma - 14
                 (so bytes with low bits 50..59 shift LEFT by 14..5), same bias mask and rule
  unsigned       a negative shift_cut's byte read as unsigned
  sc_floor0      a negative shift_cut clamped to 0
  signed         a byte >= 128 read as a negative shift_cut
  base           the patch had no effect
With a bias of 127 << shift_bias, a 32-bit accumulator holds the arm exactly only for
shift_bias <= 23. Arms past that are width probes: which model matched, not "exact".

The device runs a hand-edited binary. Run with an outer timeout, from an idle NPU.
  python tools/dpu_sigma_patch_test.py --out ../../../scratch/dpu_re/patch --base 16 \
      --controls 15,1 --patches 17,18,20,24,31,45,46
  python tools/dpu_sigma_patch_test.py --dry-run --reference code6_sb4 --controls 16:8 --patches 20:15,-1,60
  python tools/dpu_sigma_patch_test.py --rescore ../../../scratch/dpu_re/patch3     # re-score a saved run
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
from dpu_mc_fixtures import REG0_SHIFT_BIAS, REG0_SHIFT_CUT, reg0_span  # noqa: E402

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
def patch(original, site, base, v, base_sb=0, sb=0):
    """The xmodel with (shift_cut, shift_bias) rewritten from the base pair at the chosen site."""
    if site == 'param_buffer':
        assert sb == base_sb, 'the param_buffer site patches shift_cut only'
        return original.replace(PRE + field(base, base_sb) + POST, PRE + field(v, sb) + POST)
    i, n = reg0_span(original)
    assert original[i + REG0_SHIFT_CUT] == base & 0xFF, original[i + REG0_SHIFT_CUT]
    assert original[i + REG0_SHIFT_BIAS] == (base_sb << 2 | 1), original[i + REG0_SHIFT_BIAS]
    b = bytearray(original)
    b[i + REG0_SHIFT_CUT] = v & 0xFF
    b[i + REG0_SHIFT_BIAS] = sb << 2 | 1
    return bytes(b)


def model_out(x, w, b, sc, sb, acc='wide'):
    """Output codes for shift_cut sc and shift_bias sb, in integers (half_up output rounding,
    as measured). acc: 'wide' exact, 'sat32'/'wrap32' bound the accumulator to 32 bits."""
    codes = np.clip(np.rint(x.astype(np.float64)), -128, 127).astype(np.int64)
    t = w @ codes.reshape(w.shape[1], -1) + (b.astype(np.int64) << sb)[:, None]
    if acc == 'sat32':
        t = np.clip(t, -2**31, 2**31 - 1)
    elif acc == 'wrap32':
        t = (t + 2**31) % 2**32 - 2**31
    if sc > 0:
        sc = min(sc, 62)                       # |t| < 2**61, so any larger shift gives the same 0/-1
        r = (t + (1 << (sc - 1))) >> sc
    elif acc == 'lsat1' and sc < 0:
        # The vsrs left-shift rule (fitted on run 3): saturate only if a ONE-bit shift would
        # overflow int8, otherwise t << k wrapped to 8 bits. Same shape as the acc64 rule in
        # BENCHMARKS "The AIE2 shifter reads a 6-bit code".
        k = min(-sc, 16)
        wrapped = ((np.clip(t, -64, 63) << k) + 128) % 256 - 128
        r = np.where(t > 63, 127, np.where(t < -64, -128, wrapped))
    else:
        r = np.clip(t, -256, 256) << min(-sc, 16)   # past +-256, or shifted 16+, a nonzero t saturates
    return np.clip(r + 128, 0, 255).astype(np.uint8).reshape(1, 32, 16, 16)


def hypotheses(sc, sb, base=None):
    """Named alternatives as (shift_cut, shift_bias, accumulator) triples; see the docstring."""
    h = {'exact': (sc, sb, 'wide'), 'sat32': (sc, sb, 'sat32'), 'wrap32': (sc, sb, 'wrap32'),
         'sb_clamp15': (sc, min(sb, 15), 'wide')}
    for n in (16, 17, 31):
        h['sc_clamp%d' % n] = (min(sc, n), sb, 'wide')
    h['sigma_mask5'] = (((sc + 14) & 31) - 14, sb, 'wide')
    code = (sc + 14) & 63
    h['vsrs6'] = ((code if code < 60 else code - 64) - 14, sb, 'wide')
    code = sc & 63                             # fitted on run 3: the byte itself is the vsrs code
    h['code6_sb4'] = (code if code < 60 else code - 64, sb & 15, 'lsat1')
    h['code6_sb4_sat'] = (h['code6_sb4'][0], sb & 15, 'wide')     # same shift, plain saturation
    h['sigma6_sb4'] = (((sc + 14) & 63) - 14, sb & 15, 'lsat1')   # its closest rival on run 3
    if sc < 0:
        h['unsigned'] = (sc & 0xFF, sb, 'wide')
        h['sc_floor0'] = (0, sb, 'wide')
    if sc >= 128:
        h['signed'] = (sc - 256, sb, 'wide')
    if base is not None:
        h['base'] = (base[0], base[1], 'wide')
    return h


def score_all(out, cases, w, b, sc, sb, base=None):
    """Mismatched elements of out against every hypothesis."""
    return {k: sum(int(np.count_nonzero(out[n] != model_out(x, w, b, *t))) for n, x in cases.items())
            for k, t in hypotheses(sc, sb, base).items()}


def separation(cases, w, b, sc, sb, base=None, reference='exact'):
    """For a pre-registration: elements by which each alternative differs from the reference
    model's prediction, and how many distinct output codes that prediction has (1 means the
    arm is degenerate)."""
    t = hypotheses(sc, sb, base)[reference]
    ref = {n: model_out(x, w, b, *t) for n, x in cases.items()}
    return score_all(ref, cases, w, b, sc, sb, base), len(np.unique(np.concatenate([v.ravel() for v in ref.values()])))


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
        ref = model_out(x, w, b, sc, sb)
        d = np.abs(out[name].astype(np.int16) - ref.astype(np.int16))
        mism += int(np.count_nonzero(d))
        maxd = max(maxd, int(d.max()))
    return mism, maxd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--base', type=int, default=16)
    ap.add_argument('--base-sb', type=int, default=0, help='shift_bias the base fixture is compiled with')
    ap.add_argument('--controls', default='15,1', help='arms: shift_cut or shift_cut:shift_bias, comma-separated')
    ap.add_argument('--patches', default='17,18,20,24,31,45,46')
    ap.add_argument('--channels', type=int, default=32)
    ap.add_argument('--site', choices=('reg0', 'param_buffer'), default='reg0',
                    help='reg0: the REG_0 parameter byte (what the DPU loads); param_buffer: the '
                         'weight attr varint, which run 1 showed the device ignores')
    ap.add_argument('--dry-run', action='store_true',
                    help='print each arm\'s separation from the alternative models; no device')
    ap.add_argument('--reference', default='exact',
                    help='--dry-run: the model whose prediction the alternatives are measured against')
    ap.add_argument('--rescore', type=Path, metavar='RUN_DIR',
                    help='re-score a finished run\'s saved outputs against every model; no device')
    a = ap.parse_args()

    def arms(spec):
        out = []
        for x in spec.split(','):
            if x:
                sc, _, sb = x.partition(':')
                out.append((int(sc), int(sb) if sb else a.base_sb))
        return out
    controls, patches = arms(a.controls), arms(a.patches)
    for v, sb in controls + patches + [(a.base, a.base_sb)]:
        if a.site == 'reg0' and not (-128 <= v <= 255 and 0 <= sb <= 55):
            ap.error('REG_0 holds shift_cut in one byte and shift_bias << 2 | 1 in another (oracle: sb <= 55)')
        if a.site == 'param_buffer' and (sb != a.base_sb or len(field(v)) != len(field(a.base))):
            ap.error('shift_cut %d changes the varint length; only same-length patches are supported' % v)
    g, w, b = probe.model(a.channels, a.base, a.base_sb, False)
    cases = probe.fixtures(a.channels, False)
    for sc, sb in [(a.base, a.base_sb), (15, 0), (1, 0), (16, 8)]:   # the integer oracle against the probe's
        assert all(np.array_equal(model_out(x, w, b, sc, sb), probe.predict(x, w, b, sc, sb, 'half_up'))
                   for x in cases.values()), (sc, sb)
    if a.rescore:
        rep = json.loads((a.rescore / 'report.json').read_text())
        bad = 0
        for arm in rep['arms']:
            v, sb = arm['shift_cut'], arm.get('shift_bias', 0)
            f = a.rescore / ('outputs_%s_sc%d_sb%d.npz' % (arm['kind'], v, sb))
            if not f.exists():
                print('%-8s %4d:%-3d no outputs (%s)' % (arm['kind'], v, sb, arm['verdict']))
                continue
            out = dict(np.load(f))
            models = score_all(out, cases, w, b, v, sb, (rep['base'], rep.get('base_sb', 0)))
            bad += models['code6_sb4'] != 0
            print('%-8s %4d:%-3d code6_sb4 %5d  matches %s' % (
                arm['kind'], v, sb, models['code6_sb4'], sorted(k for k, n in models.items() if n == 0)))
        print('arms code6_sb4 misses: %d' % bad)
        return 0
    if a.dry_run:
        for kind, values in (('control', controls), ('patched', patches)):
            for sc, sb in values:
                sep, distinct = separation(cases, w, b, sc, sb, (a.base, a.base_sb), a.reference)
                print('%-8s %4d:%-3d sigma %3d  %s distinct codes %3d  alternatives differ by %s' % (
                    kind, sc, sb, sc + 14, a.reference, distinct, {k: v for k, v in sep.items() if k != a.reference}))
        return 0
    if not a.out or a.out.exists():
        ap.error('--out must be new')
    a.out.mkdir(parents=True)
    cache = (CACHE_DIR / RESNET_CACHE_KEY).resolve()
    if cache.parent != ROOT.resolve():
        raise SystemExit('EP cache escaped this checkout: %s' % cache)
    if not idle():
        raise SystemExit('NPU not idle: refusing (a foreign context is contention, not a finding)')

    onnx_path = a.out / 'fixture.onnx'
    g.save(onnx_path)
    xclbin = str(resolve_xclbin())
    report = {'site': a.site, 'base': a.base, 'base_sb': a.base_sb, 'controls': controls,
              'patches': patches, 'arms': []}

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
    n_sites = (original.count(PRE + field(a.base, a.base_sb) + POST) if a.site == 'param_buffer'
               else int(original[reg0_span(original)[0] + REG0_SHIFT_CUT] == a.base & 0xFF
                        and original[reg0_span(original)[0] + REG0_SHIFT_BIAS] == a.base_sb << 2 | 1))
    cache_files = sorted(p.name for p in cache.rglob('*') if p.is_file())
    m, d = score(base_out, cases, w, b, a.base, a.base_sb)
    print('BASE shift_cut %d: NPU:Conv %s, sites %d, H_exact mismatches %d (max %d LSB)'
          % (a.base, ep['ops_by_device'].get('NPU:Conv', 0), n_sites, m, d), flush=True)
    report['base_arm'] = {'npu_conv': ep['ops_by_device'].get('NPU:Conv', 0), 'sites': n_sites,
                          'h_exact_mismatches': m, 'max_lsb': d, 'xmodel_sha256': hashlib.sha256(original).hexdigest()}
    if n_sites < 1 or m != 0 or ep['ops_by_device'].get('NPU:Conv', 0) != 1:
        raise SystemExit('base arm is not a clean placed fixture; nothing to patch')

    for kind, values in (('control', controls), ('patched', patches), ('restore', [(a.base, a.base_sb)])):
        for v, sb in values:
            if not idle():
                print('NPU not idle before arm %s %d:%d: stopping' % (kind, v, sb), flush=True)
                report['stopped'] = 'not idle'
                break
            patched = patch(original, a.site, a.base, v, a.base_sb, sb)
            xm.write_bytes(patched)
            want = hashlib.sha256(patched).hexdigest()
            arm = {'kind': kind, 'shift_cut': v, 'shift_bias': sb, 'sigma': v + 14,
                   'field': (field(v, sb).hex(' ') if a.site == 'param_buffer'
                             else '%02x %02x' % (sb << 2 | 1, v & 0xFF))}
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
                    arm['h_exact'] = score(out, cases, w, b, v, sb)
                    arm['h_base'] = score(out, cases, w, b, a.base, a.base_sb)
                    arm['models'] = score_all(out, cases, w, b, v, sb, (a.base, a.base_sb))
                    arm['matches'] = sorted(k for k, n in arm['models'].items() if n == 0)
                    np.savez_compressed(a.out / ('outputs_%s_sc%d_sb%d.npz' % (kind, v, sb)), **out)
                    arm['verdict'] = ('EXACT at %d:%d' % (v, sb) if arm['h_exact'][0] == 0 else
                                      'NO EFFECT / base' if arm['h_base'][0] == 0 else 'NEITHER')
                del s
            except Exception as e:  # noqa: BLE001 -- a refusal or device error is the finding
                arm['verdict'] = 'ERROR: %s: %s' % (type(e).__name__, str(e)[:300])
            gc.collect()
            print('%-8s %4d:%-3d (sigma %3d) bytes [%s]: %s  %s' % (
                kind, v, sb, v + 14, arm['field'], arm['verdict'],
                {k: arm[k] for k in ('h_exact', 'h_base', 'matches', 'npu_conv', 'xmodel_unchanged_by_ep',
                                     'cache_file_set_unchanged') if k in arm}), flush=True)
            if 'models' in arm and not arm['matches']:
                print('         nearest models: %s' % sorted(arm['models'].items(), key=lambda kv: kv[1])[:4], flush=True)
            report['arms'].append(arm)
    xm.write_bytes(original)
    (a.out / 'report.json').write_text(json.dumps(report, indent=1))
    print('cache xmodel restored to the original bytes; NPU idle after: %s' % idle(), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

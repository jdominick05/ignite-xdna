"""Differential mc_code fixtures: which bytes of the DPU instruction stream carry sigma?

The VitisAI DPU overlay (phoenix 4x4.xclbin) runs AMD core programs that read 40-byte
instruction packets from the per-model mc_code (tools/xclbin_overlay_extract.py; BENCHMARKS,
"The AIE2 shifter reads a 6-bit code"). This compiles single-Conv fixtures that differ ONLY
in shift_cut (sigma = shift_cut + 14), so any mc_code byte that moves with shift_cut is a
sigma field candidate. The other fields stay fixed: weights, input, channels.

Each fixture is built and run by tools/xint8_arithmetic_probe.py (exact references, an
idle-device precheck, --fresh, an EP placement check). After each one, the compiled .xmodel
is copied out of the cache before the next fixture clears it. mc_code is pulled from the
xmodel's protobuf: map entry "mc_code" -> value (field 2) -> field 12 -> bytes (field 1).

Run from a worktree in resnet_env17: the probe refuses an EP cache outside its own repo
root, so the cache it clears is the worktree's, never the main checkout's.
Outputs go to --out (keep under git-ignored scratch/: compiled AMD artifacts):
  sc<N>/  probe run dir, compiled.xmodel, mc_code_<k>.bin
  and a diff table on stdout: every byte offset whose value differs across fixtures,
  with its value per shift_cut.

  python tools/dpu_mc_fixtures.py --out ../../../scratch/dpu_re/fixtures --shift-cuts 0,1,2,4,8,15,16
  python tools/dpu_mc_fixtures.py --out <same> --diff-only     # re-read without running
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'modelcachekey'


def varint(b, i):
    v = s = 0
    while True:
        c = b[i]
        i += 1
        v |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return v, i


def mc_code_blobs(xmodel_bytes):
    """Every mc_code payload in an xmodel, in file order."""
    key = b'\x0a\x07mc_code\x12'
    out, start = [], 0
    while True:
        k = xmodel_bytes.find(key, start)
        if k < 0:
            return out
        i = k + len(key)
        _, i = varint(xmodel_bytes, i)          # AttrValue length
        assert xmodel_bytes[i] == 0x62, hex(xmodel_bytes[i])
        _, i = varint(xmodel_bytes, i + 1)      # field 12 length
        assert xmodel_bytes[i] == 0x0A, hex(xmodel_bytes[i])
        n, i = varint(xmodel_bytes, i + 1)      # bytes length
        out.append(xmodel_bytes[i:i + n])
        start = i + n


def run_fixture(out, sc, channels, sb=0):
    d = out / ('sc%d' % sc if sb == 0 else 'sc%d_sb%d' % (sc, sb))
    if (d / 'compiled.xmodel').exists():
        print('sc%d: already built, skipping' % sc)
        return True
    if d.exists():
        shutil.rmtree(d)
    cmd = [sys.executable, str(ROOT / 'tools' / 'xint8_arithmetic_probe.py'), '--fresh',
           '--channels', str(channels), '--shift-cut', str(sc), '--shift-bias', str(sb), '--out', str(d / 'probe')]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    d.mkdir(parents=True, exist_ok=True)
    (d / 'probe.log').write_text(r.stdout + r.stderr, encoding='utf-8')
    placed = 'tested_conv_on_npu' in r.stdout or '"NPU:Conv": 1' in r.stdout
    xm = sorted(CACHE.glob('*.xmodel'))
    print('%s: probe exit %d, %d xmodel(s) in cache, NPU Conv placed %s'
          % (d.name, r.returncode, len(xm), '"NPU:Conv": 1' in r.stdout))
    if r.returncode != 0 or len(xm) != 1:
        return False
    shutil.copy2(xm[0], d / 'compiled.xmodel')
    return placed


def diff(out, scs):
    blobs = {}
    for sc in scs:
        p = out / ('sc%d' % sc) / 'compiled.xmodel'
        if not p.exists():
            continue
        bl = mc_code_blobs(p.read_bytes())
        for k, b in enumerate(bl):
            (p.parent / ('mc_code_%d.bin' % k)).write_bytes(b)
        blobs[sc] = bl
    if len(blobs) < 2:
        print('fewer than two fixtures with mc_code; nothing to diff')
        return 1
    scs = sorted(blobs)
    print('mc_code blobs per fixture:', {sc: [len(b) for b in blobs[sc]] for sc in scs})
    for k in range(min(len(blobs[sc]) for sc in scs)):
        lens = {len(blobs[sc][k]) for sc in scs}
        print('\n== blob %d, lengths %s' % (k, sorted(lens)))
        n = min(lens)
        varying = [o for o in range(n) if len({blobs[sc][k][o] for sc in scs}) > 1]
        print('bytes that differ across shift_cut: %d of %d' % (len(varying), n))
        print('offset  word@  ' + '  '.join('sc%-3d' % sc for sc in scs))
        for o in varying[:200]:
            print('0x%05x  0x%05x ' % (o, o & ~3) + '  '.join('%02x   ' % blobs[sc][k][o] for sc in scs))
    return 0


def param_buffers(xmodel_bytes):
    """Every `param_buffer` attr payload in file order. In a single-Conv fixture there are
    three. The second belongs to the weight tensor and carries the quantization shifts."""
    key = b'\x0a\x0cparam_buffer\x12'
    out, start = [], 0
    while True:
        k = xmodel_bytes.find(key, start)
        if k < 0:
            return out
        i = k + len(key)
        n, i = varint(xmodel_bytes, i)
        end = i + n
        assert xmodel_bytes[i] in (0x62, 0x72), hex(xmodel_bytes[i])   # bytes-value wrapper
        _, i = varint(xmodel_bytes, i + 1)
        assert xmodel_bytes[i] == 0x0A, hex(xmodel_bytes[i])
        m, i = varint(xmodel_bytes, i + 1)
        out.append(xmodel_bytes[i:i + m])
        start = end


SHIFT_FIELD_PRE, SHIFT_FIELD_POST = bytes.fromhex('c4 88 01'), bytes.fromhex('80 0c')


def shift_field(buf):
    """The varint between the fixture's fixed neighbours in the weight param_buffer.
    MEASURED 2026-10-12: value = shift_cut << 8 | shift_bias << 2 | 1. The shift_cut part
    held 5 of 5 pre-registered held-out values; the shift_bias part is fitted on 3 points."""
    i = buf.find(SHIFT_FIELD_PRE) + len(SHIFT_FIELD_PRE)
    j = buf.find(SHIFT_FIELD_POST, i)
    raw = buf[i:j]
    v = s = 0
    for c in raw:
        v |= (c & 0x7F) << s
        s += 7
    return raw, v


REG0_SHIFT_BIAS, REG0_SHIFT_CUT = 0x7B, 0x7C


def reg0_span(x):
    """(start, length) of the subgraph's REG_0 bytes, the parameter memory the DPU loads: the
    first map<string, bytes> entry `0a 05 'REG_0' 12 <len>` after the one
    reg_id_to_parameter_value key. Later REG_0 entries belong to sibling maps (context types).
    MEASURED 2026-10-12: byte 0x7B = shift_bias << 2 | 1, byte 0x7C = shift_cut, and these are
    the only bytes that vary across the sweep. Patching 0x7C alone moves the device's shift
    (tools/dpu_sigma_patch_test.py); the param_buffer varint does not."""
    k = x.find(b'reg_id_to_parameter_value')
    entry = bytes.fromhex('0a 05') + b'REG_0' + bytes.fromhex('12')
    j = x.find(entry, k)
    if k < 0 or j < 0 or x.find(b'reg_id_to_parameter_value', k + 1) >= 0:
        raise SystemExit('reg_id_to_parameter_value/REG_0 not found exactly once')
    n, i = varint(x, j + len(entry))
    return i, n


def summary(out):
    """One row per built fixture: placement, mc_code identity, the weight attr shift field and
    the REG_0 shift bytes; then every REG_0 byte offset that varies across the fixtures."""
    rows = sorted(p.parent for p in out.glob('*/compiled.xmodel'))
    ref = None
    print('fixture   NPU:Conv  mc_code sha256        param_buffer#1 field  value   '
          'sc<<8|sb<<2|1  buffers 0/2 == first  REG_0 0x7B 0x7C  == (sb<<2|1, sc)')
    ok = True
    regs = {}
    for d in rows:
        x = (d / 'compiled.xmodel').read_bytes()
        i, n = reg0_span(x)
        regs[d.name] = r = x[i:i + n]
        mc = {hashlib.sha256(b).hexdigest()[:16] for b in mc_code_blobs(x)}
        pb = param_buffers(x)
        raw, v = shift_field(pb[1])
        ref = ref or pb
        log = (d / 'probe.log').read_text(encoding='utf-8')
        res = json.loads(next(l for l in log.splitlines() if l.startswith('ARITHMETIC_RESULT ')).split(' ', 1)[1])
        sc, sb = res['shift_cut'], res['shift_bias']
        pred = (sc << 8) | (sb << 2) | 1
        reg_ok = (r[REG0_SHIFT_BIAS], r[REG0_SHIFT_CUT]) == ((sb << 2) | 1, sc)
        ok &= pred == v and reg_ok
        print('%-9s %-8s  %-20s  [%-8s]            0x%04x  %-13s  %-20s  %4d B  %02x   %02x    %s' % (
            d.name, res['tested_conv_on_npu'], ','.join(sorted(mc)), raw.hex(' '), v,
            'yes' if pred == v else 'NO (0x%04x)' % pred, pb[0] == ref[0] and pb[2] == ref[2],
            n, r[REG0_SHIFT_BIAS], r[REG0_SHIFT_CUT], 'yes' if reg_ok else 'NO'))
    lens = {len(r) for r in regs.values()}
    varying = [o for o in range(min(lens)) if len({r[o] for r in regs.values()}) > 1]
    print('\nREG_0 lengths %s; byte offsets that vary across all %d fixtures: %s'
          % (sorted(lens), len(regs), ['0x%02x' % o for o in varying]))
    ok &= len(lens) == 1 and varying == [REG0_SHIFT_BIAS, REG0_SHIFT_CUT]
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--summary', action='store_true',
                    help='table of every built fixture: placement, mc_code hash, weight shift field')
    ap.add_argument('--shift-cuts', default='0,1,2,4,8,15,16')
    ap.add_argument('--channels', type=int, default=32)
    ap.add_argument('--shift-bias', type=int, default=0, help='held fixed across the shift-cut sweep')
    ap.add_argument('--diff-only', action='store_true')
    a = ap.parse_args()
    if a.summary:
        return summary(a.out)
    scs = [int(x) for x in a.shift_cuts.split(',')]
    a.out.mkdir(parents=True, exist_ok=True)
    if not a.diff_only:
        for sc in scs:
            if not run_fixture(a.out, sc, a.channels, a.shift_bias):
                print('sc%d did not produce a placed, compiled fixture; stopping' % sc)
                return 1
    return diff(a.out, scs)


if __name__ == '__main__':
    sys.exit(main())

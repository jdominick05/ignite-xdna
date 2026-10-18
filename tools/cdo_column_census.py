"""Which AIE columns does an overlay's PDI set actually program?

Built for the 5th-column question (results/aie/notes_column0_architecture_audit.md): the
driver hardcodes first_col = 1 on Phoenix, and AMD ships overlays named 5x4_*.xclbin.
This reads a PDI set dumped with
    xclbinutil --dump-section PDI:RAW:<f> --input <xclbin>
and reports every AIE column addressed by any CDO command, plus program-memory words
per (column, row). Columns are partition-relative, so add the xclbin's AIE_PARTITION
start column (xclbinutil --dump-section AIE_PARTITION:JSON) for the physical index.

Two CDO encodings occur on this machine, and both are handled:
  * PLM CDO (the 5x4_* overlays in C:\\Windows\\System32\\AMD): each command header is
    len << 16 | module << 8 | cmd, and len 0xff means the next word is the length.
    Module 1 cmd 3 is write, 2 mask-write, 5 dma-write, 7/8 the 64-bit forms.
  * the record format of the Ryzen AI 1.7.1 phoenix overlays (tools/xclbin_overlay_extract.py)
Any command it cannot place makes it exit 1. Like that tool's, the encodings are inferred
from the bytes and not from a spec.
"""
import argparse
import collections
import glob
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xclbin_overlay_extract import parse_pdi  # noqa: E402

PM_LO, PM_HI = 0x20000, 0x24000


def walk_plm(blob):
    m = blob.find(struct.pack('<I', 0x004F4443))
    o = m + 16
    n = (len(blob) - o) // 4
    w = struct.unpack_from('<%dI' % n, blob, o)
    i, cols, pm, cmds = 0, collections.Counter(), collections.Counter(), collections.Counter()
    while i < n:
        h = w[i]
        ln, mod, cmd = (h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF
        i += 1
        if ln == 0xFF:
            ln = w[i]
            i += 1
        p = w[i:i + ln]
        i += ln
        cmds[(mod, cmd)] += 1
        if mod != 1:
            continue
        if cmd in (2, 3) and ln >= 2:
            addr = p[0]
        elif cmd in (5, 7, 8) and ln >= 3:
            addr = p[0] << 32 | p[1]
        else:
            continue
        a = addr & 0xFFFFFFFF
        cols[a >> 25] += 1
        if PM_LO <= (a & 0xFFFFF) < PM_HI:
            pm[(a >> 25, (a >> 20) & 0x1F)] += ln
    if i != n:
        raise ValueError('command walk overran the PDI (%d of %d words)' % (i, n))
    return cmds, cols, pm


def walk_records(blob):
    _, dma, writes, mws, _ = parse_pdi(blob)
    cols, pm = collections.Counter(), collections.Counter()
    for a, _ in writes:
        cols[a >> 25] += 1
    for a, _, _ in mws:
        cols[a >> 25] += 1
    for a, _, f in dma:
        lo = a & 0xFFFFFFFF
        cols[lo >> 25] += 1
        if PM_LO <= (lo & 0xFFFFF) < PM_HI:
            pm[(lo >> 25, (lo >> 20) & 0x1F)] += f & 0xFFFF
    return collections.Counter({'records': 1}), cols, pm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pdi_dirs', nargs='+', help='directories holding one overlay\'s dumped *.pdi files')
    a = ap.parse_args()
    ok = True
    for d in a.pdi_dirs:
        files = sorted(glob.glob(os.path.join(d, '*.pdi')))
        cols, pm, kinds = collections.Counter(), collections.Counter(), collections.Counter()
        for f in files:
            blob = open(f, 'rb').read()
            try:
                k, c, p = walk_records(blob)
                kinds['record format'] += 1
            except ValueError:
                try:
                    k, c, p = walk_plm(blob)
                    kinds['PLM CDO'] += 1
                except ValueError as e:
                    print('  %s: cannot parse (%s)' % (os.path.basename(f), e))
                    ok = False
                    continue
            cols += c
            pm += p
        print('%s: %d PDIs %s' % (os.path.basename(os.path.normpath(d)), len(files), dict(kinds)))
        print('  columns addressed (partition-relative): %s' % dict(sorted(cols.items())))
        print('  cores given program memory: %d, in columns %s' % (len(pm), sorted({c for c, _ in pm})))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

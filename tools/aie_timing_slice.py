"""Timing-aware backward slice of one register at one bundle of an AIE2 disassembly listing.

The AIE2 core has no interlocks: a register read sees whichever write has COMMITTED by the
cycle it reads, so a plain def-use walk can name the wrong producer when a long-latency load
and a short-latency op write the same register close together. This walks back with the
latencies of llvm-aie's own scheduling model instead.

Model (SPEC(Peano): llvm-aie `AIE2Schedule.td`, snapshot a36c62b9d26291fb06604bc976c897791f0bb578,
not measured here):
  * bundles in the range issue one per cycle in listing order, so the range must be one
    straight-line path: the tool does not follow branches, and a taken branch or a hardware
    loop's back edge inside the range makes the answer wrong;
  * a write is visible to a read iff  t_def + def_cycle <= t_use + use_cycle - 1;
    among visible writes the latest commit wins;
  * def cycles: scalar `lda*` 7 (II_LDA*, operand cycle 7; a post-modified pointer 1),
    `mul` 2 (II_MUL), everything else 1;
  * use cycles: the shift scalar of `vlda*.ups` 7 (II_VLDA_UPS / II_VLDA_2D_UPS), everything
    else 1.
Writes before the range print as PRE. Pointer registers and sp are not followed.

Used on the VitisAI overlay's PDI 2 (pm_0002, git-ignored scratch) for
results/aie/ups_fused_probe_desktop2_20261012T203701Z.log.

  python tools/aie_timing_slice.py LISTING.s --lo 0x1c00 --hi 0x24d8 --at 0x24d4 --reg s3
"""
import argparse
import re

REG = r'(?:r\d+|p\d+|m\d+|s\d+|dn\d+|dj\d+|dc\d+|lc|ls|le|sp|lr)'
NOWRITE = re.compile(r'^(st|vst|j|ret|done|acq|rel|nop|event|lock|paddb|padda)')
VECTOR = ('vlda', 'vldb', 'vst', 'vsrs', 'vups', 'vmac', 'vmul', 'vmov', 'vshift', 'vmin', 'vmax', 'vbcst',
          'vshuffle', 'vpack', 'vadd', 'vsub', 'vsel', 'vext', 'vinsert', 'vclr', 'vlup', 'vconv')


def read_bundles(path, lo, hi):
    bundles = []
    for line in open(path):
        m = re.match(r'\s*([0-9a-f]+):\s+((?:[0-9a-f]{2} )+)\s*(.*)', line)
        if not m:
            continue
        a = int(m.group(1), 16)
        if lo <= a < hi:
            body = re.sub(r'^[0-9a-f]{2}\s+', '', m.group(3).replace('\t', ' '))
            slots = [re.sub(r'\s+', ' ', s.strip()) for s in body.split(';')]
            bundles.append((a, [s for s in slots if s and not re.match(r'nop[abmsvx]*$', s)]))
    return bundles


def analyse(ins):
    """-> ([(reg, def_cycle)], [(reg, use_cycle)])"""
    op, _, rest = ins.partition(' ')
    ops = [o.strip() for o in rest.split(',')] if rest else []
    regs = lambda xs: [r for o in xs for r in re.findall(r'\b' + REG + r'\b', o)]  # noqa: E731
    if op.startswith(VECTOR):
        rc = 7 if '.ups' in op and op.startswith('vlda') else 1
        return [], [(r, rc if re.fullmatch(r's\d', r) else 1) for r in regs(ops)]
    if NOWRITE.match(op) or not ops:
        return [], [(r, 1) for r in regs(ops)]
    d = ops[0]
    if op.startswith(('lda', 'ldb')):
        addr = ','.join(ops[1:])
        defs = [(d, 7)]
        if re.search(r'\]\s*,', addr):
            p = re.search(r'\[(p\d+)', addr)
            if p:
                defs.append((p.group(1), 1))
        return defs, [(r, 1) for r in regs([addr])]
    defs = [(d, 2 if op == 'mul' else 1)] if re.fullmatch(REG, d) else []
    return defs, [(r, 1) for r in regs(ops[1:])]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('listing')
    ap.add_argument('--lo', required=True, type=lambda s: int(s, 16))
    ap.add_argument('--hi', required=True, type=lambda s: int(s, 16))
    ap.add_argument('--at', required=True, type=lambda s: int(s, 16), help='address of the reading bundle')
    ap.add_argument('--reg', required=True)
    ap.add_argument('--depth', type=int, default=12)
    a = ap.parse_args()

    bundles = read_bundles(a.listing, a.lo, a.hi)
    insts = []
    for t, (addr, slots) in enumerate(bundles):
        for s in slots:
            defs, uses = analyse(s)
            insts.append((t, addr, s, defs, uses))
    writes = {}
    for i, (t, _, _, defs, _) in enumerate(insts):
        for r, dc in defs:
            writes.setdefault(r, []).append((t + dc, i))

    def provider(reg, t_use, uc):
        vis = [(c, i) for c, i in writes.get(reg, []) if c <= t_use + uc - 1]
        if not vis:
            return None, []
        best = max(c for c, _ in vis)
        return best, [i for c, i in vis if c == best]

    def show(i, indent, d, seen):
        t, addr, s, _, uses = insts[i]
        print('%s%5x [t=%3d] %s' % (indent, addr, t, s))
        if d == 0 or i in seen:
            return
        seen.add(i)
        for r, uc in dict.fromkeys(uses):
            if r == 'sp' or re.fullmatch(r'p\d+', r):
                continue
            c, ids = provider(r, t, uc)
            if not ids:
                print('%s  %s (read c%d) <- PRE' % (indent, r, uc))
            for j in ids:
                print('%s  %s (read c%d) <- commit t=%d%s' % (indent, r, uc, c, '  CONFLICT' if len(ids) > 1 else ''))
                show(j, indent + '    ', d - 1, seen)

    t_at = next((t for t, (addr, _) in enumerate(bundles) if addr == a.at), None)
    if t_at is None:
        raise SystemExit('no bundle at 0x%x in the range' % a.at)
    hits = [i for i, x in enumerate(insts) if x[0] == t_at and a.reg in dict(x[4])]
    if not hits:
        raise SystemExit('%s is not read at 0x%x' % (a.reg, a.at))
    for i in hits:
        t, addr, s, _, uses = insts[i]
        uc = dict(uses)[a.reg]
        print('target %5x [t=%d] %s   (%s read at cycle %d)' % (addr, t, s, a.reg, uc))
        _, ids = provider(a.reg, t, uc)
        for j in ids:
            show(j, '  ', a.depth, set())
        if not ids:
            print('  PRE')


if __name__ == '__main__':
    main()

"""Extract the AIE core programs of a VitisAI DPU overlay xclbin (e.g. phoenix/4x4.xclbin).

The VitisAI EP's "DPU" on XDNA1 is a fixed overlay: an xclbin whose PDIs program every
core of the partition with an AMD-built program that interprets the per-model mc_code.
This tool reads those programs back out so tools/aie_disasm.py / Peano llvm-objdump can
disassemble them.

Inputs: the PDI files written by `xclbinutil --dump-section PDI:RAW:<f> --input <xclbin>`
(xclbinutil writes one <uuid>.pdi per AIE_PARTITION PDI next to <f>).

PDI layout as decoded here (2026-10-12, Phoenix 4x4.xclbin, 15 PDIs). This is inferred from
the bytes, not taken from a spec, and is checked only by the round trip described below:
  * a boot header, then a CDO v2 header: 4, 0x004F4443 ("CDO"), 0x200, length, checksum
  * then records [type, count, payload]:
      type 3: count x (addr, value)               register writes
      type 2: count x (addr, mask, value)         masked writes
      type 5: count x (addr_hi, addr_lo, off, f)  DMA writes of (f & 0xffff) words taken
                                                  from the payload region at byte `off`
      type 0, count 0xffffffff: end of records; the payload region starts right after it
  * AIE2 tile address: col = addr >> 25, row = (addr >> 20) & 0x1f, offset = addr & 0xfffff;
    core program memory is offset 0x20000-0x23fff (as in compiler/scheduler.py).

Checks the decode must pass (any failure exits 1):
  * the record walk ends on the type-0 terminator, never on an unknown type
  * every program-memory image disassembles with Peano with no <unknown> encodings
    (only with --objdump; the payload base is what this test pins down)

Writes into --out (keep it under git-ignored scratch/: these are AMD binaries, and the
repo has a public mirror): pm_<pdi>_c<col>r<row>.bin, a wrapping .elf, and a census on stdout.
"""
import argparse
import collections
import hashlib
import os
import struct
import subprocess
import sys

CDO_MAGIC = 0x004F4443
PM_LO, PM_HI = 0x20000, 0x24000


def parse_pdi(blob):
    """Return (records summary, dma list, payload base) for one PDI."""
    magic_at = blob.find(struct.pack('<I', CDO_MAGIC))
    if magic_at < 4:
        raise ValueError('no CDO header')
    o = magic_at + 16  # past magic, version, length, checksum
    n = (len(blob) - o) // 4
    w = struct.unpack_from('<%dI' % n, blob, o)
    i, kinds, dma, writes, mask_writes = 0, collections.Counter(), [], [], []
    while i < n:
        t, c = w[i], w[i + 1]
        if t == 0 and c == 0xFFFFFFFF:
            return kinds, dma, writes, mask_writes, o + 4 * (i + 2)
        kinds[t] += 1
        if t == 3:
            writes += list(zip(w[i + 2:i + 2 + 2 * c:2], w[i + 3:i + 3 + 2 * c:2]))
            i += 2 + 2 * c
        elif t == 2:
            mask_writes += [tuple(w[i + 2 + 3 * k:i + 5 + 3 * k]) for k in range(c)]
            i += 2 + 3 * c
        elif t == 5:
            for k in range(c):
                hi, lo, off, fl = w[i + 2 + 4 * k:i + 6 + 4 * k]
                dma.append((hi << 32 | lo, off, fl))
            i += 2 + 4 * c
        else:
            raise ValueError('unknown record type 0x%x at word %d' % (t, i))
    raise ValueError('no terminator')


def wrap_elf(raw, ref_elf):
    """Minimal ELF32 with one .text, e_machine/e_flags copied from a Peano-built ELF."""
    ref = open(ref_elf, 'rb').read()
    machine = struct.unpack_from('<H', ref, 18)[0]
    flags = struct.unpack_from('<I', ref, 36)[0]
    shstr = b'\0.text\0.shstrtab\0'
    text = raw + b'\0' * ((-len(raw)) % 4)
    text_off = 52
    shstr_off = text_off + len(text)
    sh_off = (shstr_off + len(shstr) + 3) & ~3
    eh = b'\x7fELF' + bytes([1, 1, 1, 0]) + b'\0' * 8
    eh += struct.pack('<HHIIIIIHHHHHH', 2, machine, 1, 0, 0, sh_off, flags, 52, 0, 0, 40, 3, 2)

    def sh(name, typ, fl, off, size, align):
        return struct.pack('<10I', name, typ, fl, 0, off, size, 0, 0, align, 0)
    data = eh + text + shstr
    data += b'\0' * (sh_off - len(data))
    data += sh(0, 0, 0, 0, 0, 0) + sh(1, 1, 6, text_off, len(text), 16) + sh(7, 3, 0, shstr_off, len(shstr), 1)
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pdis', nargs='+', help='PDI files from xclbinutil --dump-section PDI:RAW')
    ap.add_argument('--out', required=True, help='output dir (keep under scratch/)')
    ap.add_argument('--ref-elf', help='a Peano-built AIE2 ELF to copy e_machine/e_flags from')
    ap.add_argument('--objdump', help='Peano llvm-objdump; if given, every image is disassembled and checked')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ok = True
    images = {}
    for pdi in a.pdis:
        blob = open(pdi, 'rb').read()
        tag = os.path.splitext(os.path.basename(pdi))[0][-4:]
        kinds, dma, writes, mws, base = parse_pdi(blob)
        need = max(off + 4 * (f & 0xFFFF) for _, off, f in dma)
        over = base + need - len(blob)
        print('PDI %s: %d B, records %s, %d reg writes, %d mask writes, %d DMA writes, payload base 0x%x, '
              'payload overrun %d B' % (tag, len(blob), dict(kinds), len(writes), len(mws), len(dma), base, max(over, 0)))
        pm = {}
        for addr, off, fl in dma:
            t = addr & 0xFFFFF
            if PM_LO <= t < PM_HI:
                col, row = addr >> 25, (addr >> 20) & 0x1F
                n = 4 * (fl & 0xFFFF)
                img = pm.setdefault((col, row), bytearray(PM_HI - PM_LO))
                img[t - PM_LO:t - PM_LO + n] = blob[base + off:base + off + n]
        for (col, row), img in sorted(pm.items()):
            used = bytes(img).rstrip(b'\0')
            h = hashlib.sha256(used).hexdigest()
            images.setdefault(h, []).append('%s:c%dr%d' % (tag, col, row))
            stem = os.path.join(a.out, 'pm_%s_c%dr%d' % (tag, col, row))
            open(stem + '.bin', 'wb').write(used)
            if a.ref_elf:
                open(stem + '.elf', 'wb').write(wrap_elf(used, a.ref_elf))
            line = '  c%d r%d: %d B of %d, sha256 %s' % (col, row, len(used), PM_HI - PM_LO, h[:16])
            if a.objdump and a.ref_elf:
                r = subprocess.run([a.objdump, '-d', stem + '.elf'], capture_output=True, text=True)
                open(stem + '.s', 'w').write(r.stdout)
                bad = r.stdout.count('<unknown>') + (r.returncode != 0)
                line += ', %d disasm lines, %d unknown, objdump exit %d' % (r.stdout.count('\n'), bad, r.returncode)
                ok &= bad == 0
            print(line)
    print('distinct program images: %d' % len(images))
    for h, where in images.items():
        print('  %s used by %d tiles: %s' % (h[:16], len(where), ' '.join(where)))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

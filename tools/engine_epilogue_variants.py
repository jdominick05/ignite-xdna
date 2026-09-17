"""Offline sizing of a piecewise-linear sigmoid SiLU epilogue for the graph engine's core program. Writes variant
copies of kernels/aie2/conv_engine/engine.cc into OUT_DIR (the repository's engine.cc is never modified), compiles
each with Peano (aie2, -O2, object only, no device) and prints .text bytes, instruction count and a census of stack
loads and stores by register class. An accumulator register (am*) stored to or loaded from the stack is a spill:
engine.cc's header records that spilled pass accumulators made the frame's core time 2.6 to 3 times longer.

The candidate epilogue, K lines (line 1 in H_A1/H_B1, lines 2..K in the free header words from 26, K <= 4):

    u = |t|;  g = clip(min_i rne((u * A_i + B_i) >> S1), 0, 64);  y = rne(((t << 6) + u * g) >> YSH)

which is t * (64 + sign(t) * g): tools/silu_integer_oracle.py --mode pl evaluates the same integer function on COCO.

Variants:
  inline_add<K>      silu_pl_u8 beside hswish_u8 inside the pass epilogue, selected by F_SIGMOID = 64
  inline_replace<K>  hswish_u8's body replaced by the PL form
  post<K>            the passes emit q1 (F_HSWISH clear); with F_SIGMOID a separate loop over the emitted or held
                     tile applies the PL form after all passes, outside every loop that holds pass accumulators

Run in the mlir-aie ironenv (source scripts/research-iron.sh, or any environment where aie.utils.config resolves):

    python tools/engine_epilogue_variants.py OUT_DIR [--lines 3 4] [--source ENGINE_CC]

The variants patch the program as it was before the sigmoid epilogue landed (the committed census ran on engine.cc
at 3b5be0e); engine.cc now contains the post4 form, so pass that older source, e.g.
``git show 3b5be0e:kernels/aie2/conv_engine/engine.cc > engine_3b5be0e.cc``.
"""
import argparse
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "kernels" / "aie2" / "conv_engine" / "engine.cc"
HS_START = "inline V32u hswish_u8(V32u q1, const Hdr &d) {\n"
HS_END = "    return sat_u8_from_i16(aie::add(y, int16_t(128)));\n}\n"


def _replace(s, old, new):
    assert s.count(old) == 1, old
    return s.replace(old, new)


def pl_body(K):
    body = ["    V32i16 t = unpack_centered(q1);", "    V32i16 u = aie::abs(t);", "    Acc l;",
            "    l.from_vector(aie::broadcast<int32, 32>(d.b1));",
            "    V32i16 g = aie::mac(l, u, int16_t(d.a1)).template to_vector<int16>(d.s1);"]
    for i in range(2, K + 1):
        body += [f"    l.from_vector(aie::broadcast<int32, 32>(d.b{i}));",
                 f"    g = aie::min(g, aie::mac(l, u, int16_t(d.a{i})).template to_vector<int16>(d.s1));"]
    body += ["    g = aie::max(g, int16_t(0));", "    g = aie::min(g, int16_t(64));", "    l.from_vector(t, 6);",
             "    V32i16 y = aie::mac(l, u, g).template to_vector<int16>(d.ysh);",
             "    return sat_u8_from_i16(aie::add(y, int16_t(128)));"]
    return "\n".join(body) + "\n"


def header_words(s, K):
    extra = range(2, K + 1)
    if not extra:
        return s
    words = ", ".join(f"H_A{i} = {24 + 2 * i}, H_B{i} = {25 + 2 * i}" for i in extra)
    s = _replace(s, "    H_RLSH_M = 24, H_RLSH_R = 25,\n", f"    H_RLSH_M = 24, H_RLSH_R = 25, {words},\n")
    s = _replace(s, "    int a1, b1, s1, qmax, k2, s2, ysh, rsh;\n",
                 "    int a1, b1, s1, qmax, k2, s2, ysh, rsh;\n    int " + ", ".join(f"a{i}, b{i}" for i in extra) + ";\n")
    return _replace(s, "    d.k2 = h[H_K2]; d.s2 = h[H_S2]; d.ysh = h[H_YSH]; d.rsh = h[H_RSH];\n",
                    "    d.k2 = h[H_K2]; d.s2 = h[H_S2]; d.ysh = h[H_YSH]; d.rsh = h[H_RSH];\n    "
                    + " ".join(f"d.a{i} = h[H_A{i}]; d.b{i} = h[H_B{i}];" for i in extra) + "\n")


def silu_fn(K):
    return ("// Piecewise-linear sigmoid SiLU on 32 uint8 values at the header's line constants: q1 -> q2.\n"
            "__attribute__((always_inline))\ninline V32u silu_pl_u8(V32u q1, const Hdr &d) {\n" + pl_body(K) + "}\n\n")


def variants(base, K):
    i0 = base.index(HS_START) + len(HS_START)
    i1 = base.index(HS_END, i0) + len(HS_END)
    out = {f"inline_replace{K}": header_words(base[:i0] + pl_body(K) + "}\n" + base[i1:], K)}

    add = base[:i1] + "\n" + silu_fn(K).rstrip("\n") + "\n" + base[i1:]
    add = _replace(add, "F_RES_SHIFTS = 32 };", "F_RES_SHIFTS = 32, F_SIGMOID = 64 };")
    old_ep = "    if (!(d.flags & F_HSWISH))\n        return q1;\n    return hswish_u8(q1, d);\n"
    add = _replace(add, old_ep, "    if (d.flags & F_SIGMOID)\n        return silu_pl_u8(q1, d);\n" + old_ep)
    out[f"inline_add{K}"] = header_words(add, K)

    post = _replace(base, "F_RES_SHIFTS = 32 };", "F_RES_SHIFTS = 32, F_SIGMOID = 64 };")
    anchor = "// HardSwish epilogue on one 4x8 accumulator"
    post = _replace(post, anchor, silu_fn(K) + anchor)
    post = _replace(post, "// Nearest 2x upsampling", "// The PL sigmoid SiLU over a whole emitted or held tile, in place.\n"
                    "void silu_pl_tile(const Hdr &d, uint8_t *dst) {\n"
                    "    for (int off = 0; off < NCO * OUT_BLOCK_BYTES; off += 32)\n"
                    "        aie::store_v(dst + off, silu_pl_u8(aie::load_v<32>(dst + off), d));\n}\n\n"
                    "// Nearest 2x upsampling")
    loop = "            conv_pass<false>(d, apkt, w, bias, psum, out, r, 4);\n        }\n"
    post = _replace(post, loop, loop + "        if ((d.flags & F_SIGMOID) && (d.flags & (F_EMIT | F_HOLD)))\n"
                    "            silu_pl_tile(d, (d.flags & F_EMIT) ? out : "
                    "reinterpret_cast<uint8_t *>(psum) + HOLD_OFFSET_BYTES);\n")
    out[f"post{K}"] = header_words(post, K)
    return out


def census(src, obj_dir):
    from aie.utils import config
    peano = Path(config.peano_install_dir()) / "bin"
    obj = obj_dir / (src.parent.name + ".o")
    cmd = [str(peano / "clang++.exe"), "-O2", "-std=c++20", "--target=aie2-none-unknown-elf", "-nostdlib", "-DNDEBUG",
           "-D__AIE_API_AIE_ADF_HPP__", "-I", config.cxx_header_path(), "-c", str(src), "-o", str(obj)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        return f"compile failed: {r.stderr[-2000:]}"
    size = subprocess.run([str(peano / "llvm-size.exe"), "-A", str(obj)], capture_output=True, text=True).stdout
    text = sum(int(m.group(1)) for m in re.finditer(r"^\.text\S*\s+(\d+)", size, re.M))
    dump = subprocess.check_output([str(peano / "llvm-objdump.exe"), "-d", str(obj)], text=True)
    insns = len(re.findall(r"^\s+[0-9a-f]+:", dump, re.M))
    moves = Counter(f"{op} {reg}" for op, reg in re.findall(r"\b(vst|vlda)\s+([a-z]+)[0-9]+, \[sp", dump))
    spills = sum(v for k, v in moves.items() if k.split()[1].startswith("am"))
    return (f".text {text} B, {insns} instructions, accumulator stack moves {spills}; stack vector moves "
            + (", ".join(f"{k} {v}" for k, v in sorted(moves.items())) or "none"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--lines", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--source", default=str(SRC), help="the engine.cc to patch (default: the repository's)")
    args = ap.parse_args()
    out = Path(args.out)
    base = Path(args.source).read_text(encoding="utf-8")
    if "F_SIGMOID" in base:
        raise SystemExit(f"{args.source} already has the sigmoid epilogue; pass the program from before it "
                         f"(--source, see the docstring)")
    sources = {"engine": base}
    for K in args.lines:
        assert 1 <= K <= 4, "the 32-word header has six free words: lines 2..4"
        sources.update(variants(base, K))
    for name, text in sources.items():
        (out / name).mkdir(parents=True, exist_ok=True)
        (out / name / "engine.cc").write_text(text, encoding="utf-8")
    if args.no_compile:
        print(f"wrote {len(sources)} sources under {out}")
        return
    for name in sources:
        print(f"{name:18s} {census(out / name / 'engine.cc', out)}", flush=True)


if __name__ == "__main__":
    main()

"""Price the compile-time dispatch gate through the mechanism production actually uses.

WHY THIS IS NOT engine_opcode_census.py --without. That tool measures what removing a dispatch
group frees by TEXT-STRIPPING the case out of a copy. Production instead compiles the case out
with `-DENGINE_OP_<NAME>=0` (kernels/aie2/conv_engine/design.py::gate_flags), which is a
different mechanism, so the price does not transfer by assertion. This tool compiles both and
compares the objects BYTE FOR BYTE.

Byte identity between those two says the earlier census's numbers are the gated build's numbers.
It does NOT say the gated object's surviving code matches the UNGATED build's, so it cannot on
its own retire the latency question - and on this kernel it does not: section 3 finds one
surviving stride-2 dual loop at 28 bundles where the ungated build has 27. On AIE2 a
hardware-loop body's bundle count is its cycle count, so that is a real timing change, and the
gate owes an A/B rather than an argument. It was measured at +0.43% of a YOLOv8n dispatch
(docs/BENCHMARKS.md). Section 3 prints which loops survive, which the gate removes, and which
moved, so a future change to this kernel gets the same warning rather than inheriting a claim.

Static: nothing is run on the device. This establishes object sizes, not a latency.

    bash scripts/research-lowlevel.sh --log results/aie/engine_dispatch_gate_census_<date>.log \\
        --checks-only -- bash scripts/research-iron.sh tools/engine_dispatch_gate_census.py \\
        --baseline <a pre-gate engine.cc> \\
        --containers build/gate/yolov8n_gated.ignite build/gate/yolov8n_ctl.ignite --survey

`--baseline` reproduces from git: `git show f7e75ec:kernels/aie2/conv_engine/engine.cc`.
"""
from __future__ import annotations

import argparse
import glob as globmod
import re
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SRC = ROOT / "kernels" / "aie2" / "conv_engine" / "engine.cc"
OUT = ROOT / "scratch" / "dispatch_gate_census"

# Must match kernels/aie2/conv_engine/design.py::GATEABLE_OPS.
OPS = ["FUSED_CONV", "MUL", "SCALE", "POOL"]
CALL_OF = {
    "MUL": "mul_tile(d, apkt, psum, out)",
    "SCALE": "scale_tile(d, apkt, wpkt, psum, out)",
    "POOL": "avgpool_tile(d, apkt, psum, out)",
    "FUSED_CONV": "fused_conv_tile(hdr, apkt, psum, out, core_row)",
}
PROGRAM_MEMORY = 16384
CORE_WRAPPER = 880          # tools/engine_linked_size.py: IRON's core program beside the object
BUDGET = PROGRAM_MEMORY - CORE_WRAPPER


def peano_bin():
    from aie.utils import config
    return Path(config.peano_install_dir()) / "bin"


def compile_one(src: Path, obj: Path, defines):
    from aie.utils import config
    peano = peano_bin()
    obj.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(peano / "clang++.exe"), "-O2", "-std=c++20", "--target=aie2-none-unknown-elf",
           "-nostdlib", "-DNDEBUG", "-D__AIE_API_AIE_ADF_HPP__"]
    cmd += [f"-D{d}" for d in defines]
    cmd += ["-I", config.cxx_header_path(), "-c", str(src), "-o", str(obj)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"compile failed ({obj.name}):\n{r.stderr[-3000:]}")
    size = subprocess.run([str(peano / "llvm-size.exe"), "-A", str(obj)],
                          capture_output=True, text=True).stdout
    text = sum(int(m.group(1)) for m in re.finditer(r"^\.text\S*\s+(\d+)", size, re.M))
    return text, sha256(obj.read_bytes()).hexdigest(), obj


def strip_cases(text: str, eol: str) -> str:
    for op in OPS:
        block = f"    case OP_{op}:{eol}        {CALL_OF[op]};{eol}        break;{eol}"
        if text.count(block) != 1:
            raise SystemExit(f"case OP_{op} not found verbatim - the census anchors have moved")
        text = text.replace(block, "")
    return text


def survey_container(path: Path):
    from ignite_xdna.compiler import engine_emulator as em
    from ignite_xdna.compiler.serializer import IgniteModelReader
    import numpy as np
    with IgniteModelReader(str(path)) as r:
        wp = r.get_blob_bytes("wpackets.bin")
        ge = r.manifest["graph_engine"]
    a = np.frombuffer(wp, dtype=np.uint8)
    n = a.size // em.W_BYTES
    words = a[:n * em.W_BYTES].reshape(n, em.W_BYTES)[:, :em.HDR_BYTES].copy().view(np.int32)
    present = sorted({em.OP_NAMES.get(int(o), f"?{o}") for o in np.unique(words[:, em.H_OP])})
    gateable = [o for o in present if o in OPS]
    return n, present, gateable, ge


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=str(SRC))
    ap.add_argument("--baseline", default=None,
                    help="a pre-gate engine.cc; checks the guards are inert when none is set")
    ap.add_argument("--containers", nargs=2, metavar=("GATED", "UNGATED"), default=None,
                    help="two containers of the same model; checks the gate does not move the schedule")
    # A flag, not a pattern: passed as a string it is expanded by the shell before it ever
    # reaches here, because this runs through two wrapper scripts and one layer of quoting
    # does not survive both.
    ap.add_argument("--survey", action="store_true",
                    help="read the derived gate set from every container under build/")
    ap.add_argument("--survey-glob", default="build/**/*.ignite",
                    help="what --survey scans (default: %(default)s)")
    args = ap.parse_args()

    src = Path(args.source)
    gated_src = src.read_text(encoding="utf-8", newline="")
    if "#if ENGINE_OP_FUSED_CONV" not in gated_src:
        raise SystemExit(f"{src} carries no dispatch gate")
    work = OUT / "src"
    work.mkdir(parents=True, exist_ok=True)
    copy = work / "engine.cc"
    copy.write_text(gated_src, encoding="utf-8", newline="")

    print("=== 1. what the gate frees ===")
    ungated_text, ungated_sha, ungated_obj = compile_one(copy, OUT / "obj" / "ungated.o", [])
    print(f"{'build':<34s} {'.text':>8s} {'freed':>7s} {'headroom':>9s}  object sha256")
    print(f"{'ungated (every case compiled in)':<34s} {ungated_text:8d} {'-':>7s} "
          f"{BUDGET - ungated_text:9d}  {ungated_sha[:16]}")
    for op in OPS:
        t, s, _ = compile_one(copy, OUT / "obj" / f"minus_{op.lower()}.o", [f"ENGINE_OP_{op}=0"])
        print(f"{'  without ' + op:<34s} {t:8d} {ungated_text - t:7d} "
              f"{BUDGET - t:9d}  {s[:16]}")
    all_off = [f"ENGINE_OP_{op}=0" for op in OPS]
    gated_text, gated_sha, gated_obj = compile_one(copy, OUT / "obj" / "gated.o", all_off)
    print(f"{'  without all four':<34s} {gated_text:8d} {ungated_text - gated_text:7d} "
          f"{BUDGET - gated_text:9d}  {gated_sha[:16]}")
    print()
    print(f"ENGINE_DISPATCH_GATE_FREED {ungated_text - gated_text} B "
          f"({ungated_text} -> {gated_text}); headroom {BUDGET - ungated_text} -> "
          f"{BUDGET - gated_text} B of the {BUDGET} B object budget")

    print()
    print("=== 2. the -D gate is the text strip the earlier census priced ===")
    strip_src = OUT / "strip" / "engine.cc"
    strip_src.parent.mkdir(parents=True, exist_ok=True)
    strip_src.write_text(strip_cases(gated_src, "\r\n" if "\r\n" in gated_src else "\n"),
                         encoding="utf-8", newline="")
    strip_text, strip_sha, _ = compile_one(strip_src, OUT / "obj" / "stripped.o", [])
    ok = strip_sha == gated_sha
    print(f"text-stripped (what --without prices)  .text {strip_text:6d}  {strip_sha[:16]}")
    print(f"EQUIVALENCE  the -D gate is byte-identical to the text strip : "
          f"{'PASS' if ok else 'FAIL'}")

    inert = True
    if args.baseline:
        base = Path(args.baseline).read_text(encoding="utf-8", newline="")
        if "ENGINE_OP_" in base:
            raise SystemExit(f"{args.baseline} is already gated; pass the kernel from before it")
        b_src = OUT / "baseline" / "engine.cc"
        b_src.parent.mkdir(parents=True, exist_ok=True)
        b_src.write_text(base, encoding="utf-8", newline="")
        b_text, b_sha, _ = compile_one(b_src, OUT / "obj" / "baseline.o", [])
        inert = b_sha == ungated_sha
        print(f"baseline (pre-gate kernel)             .text {b_text:6d}  {b_sha[:16]}")
        print(f"INERTNESS    the guards change nothing when none is set  : "
              f"{'PASS' if inert else 'FAIL'}")

    print()
    print("=== 3. the object the gate ships: spills, and whether the hot loops moved ===")
    from engine_epilogue_variants import widened          # noqa: E402
    objdump = peano_bin() / "llvm-objdump.exe"
    ung_w, gat_w = widened(ungated_obj, objdump), widened(gated_obj, objdump)
    print(f"ungated  {ung_w}")
    print(f"gated    {gat_w}")

    # On AIE2 a hardware-loop body's bundle count is its cycle count. The loops that only the
    # gated-out cases reached are expected to vanish; a loop that SURVIVES and changes its
    # count is a timing change, and the gate then owes a measurement rather than an argument.
    def loops_of(s):
        m = re.search(r"MAC loops bundles/MACs (.*)$", s)
        return m.group(1).split() if m else []

    ung_l, gat_l = loops_of(ung_w), loops_of(gat_w)
    survived = [x for x in gat_l if x in ung_l]
    vanished = [x for x in ung_l if x not in gat_l]
    changed = [x for x in gat_l if x not in ung_l]
    print(f"  loops that survive unchanged : {' '.join(survived) or 'none'}")
    print(f"  loops the gate removes       : {' '.join(vanished) or 'none'}")
    print(f"  loops that CHANGED           : {' '.join(changed) or 'none'}")
    if changed:
        print("  LOOPS MOVED - the gate is not a pure program-memory change and owes an A/B.")
    else:
        print("  No surviving loop moved: the gate is a program-memory change, not a timing one.")

    schedule_ok = True
    if args.containers:
        print()
        print("=== 4. the gate does not move the schedule ===")
        from ignite_xdna.compiler.serializer import IgniteModelReader
        got = []
        for p in args.containers:
            with IgniteModelReader(str(ROOT / p if not Path(p).is_absolute() else p)) as r:
                got.append({n: r.get_blob_bytes(n)
                            for n in ("engine.xclbin", "insts.bin", "wpackets.bin")})
        for name in ("wpackets.bin", "insts.bin", "engine.xclbin"):
            same = sha256(got[0][name]).hexdigest() == sha256(got[1][name]).hexdigest()
            want_same = name != "engine.xclbin"
            good = same == want_same
            schedule_ok &= good
            delta = len(got[1][name]) - len(got[0][name])
            print(f"  {name:16s} {len(got[0][name]):>9,} vs {len(got[1][name]):>9,} B  "
                  f"{'identical' if same else f'differs by {delta:,} B':22s} "
                  f"({'must match' if want_same else 'must differ'}) "
                  f"{'PASS' if good else 'FAIL'}")

    if args.survey:
        print()
        print("=== 5. which gate set every built container derives ===")
        paths = sorted(Path(p) for p in globmod.glob(str(ROOT / args.survey_glob), recursive=True))
        rows = 0
        family = set()
        for p in paths:
            try:
                n, present, gateable, ge = survey_container(p)
            except Exception as exc:
                print(f"  {p.relative_to(ROOT).as_posix():<46s} unreadable ({type(exc).__name__})")
                continue
            rows += 1
            family.add("+".join(present))
            print(f"  {p.relative_to(ROOT).as_posix():<46s} {n:>6,} pkts  "
                  f"{'+'.join(present):<26s} gateable: {','.join(gateable) or 'NONE'}"
                  f"  obj {(ge.get('kernel_object_sha256') or '-')[:8]}")
        print(f"  -> {rows} containers, {len(family)} distinct opcode set(s): "
              f"{' | '.join(sorted(family))}")

    return 0 if (ok and inert and schedule_ok) else 1


if __name__ == "__main__":
    sys.exit(main())

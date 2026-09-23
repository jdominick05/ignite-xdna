# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reproduce Hello XDNA!'s hand-scheduled XDNA1 bf16 kernel on this machine's NPU.

Steinert and Breuer (tnzr.org/xdna/xdna1_kernel.html) report 398 BF16 GFLOPS for one compute
tile of a Ryzen 7 8700G running their hand-written VLIW assembly kernel
`tensor_kernel_32x32x32_bf16_bf16_fp32` (M32 x K32 x N32, 256 of 288 bundles issue a `vmac.f`).
This tool re-runs that measurement here, with nothing of theirs entering the tracked tree
(their repository carries no licence): `fetch` downloads the pinned sources into `scratch/`.

    fetch   download src/XDNA1.mlir and the .s kernel at a pinned commit into scratch/tnzr-xdna/<sha>/
    build   assemble the .s with Peano, run aiecc (mlir-aie v1.4.2 flags equivalent to their
            Makefile's), and count the kernel's bundles statically with tools/aie_disasm.py
    run     time it on the NPU with their method: opcode-3 dispatch, 3 warm-ups + 10 timed
            dispatches, 10^6 kernel calls per dispatch (the core loop in XDNA1.mlir), wall clock
            around submit+wait, GFLOPS = 2*32^3*10^6 / t

Their method times a whole dispatch, so it includes dispatch overhead (~0.1 ms against a
~165 ms dispatch). The run also reports the implied core cycles per kernel call at the
measured 1.80 GHz (docs/SILICON.md), to set against the static bundle count: if they agree,
the rate is issue-bound at that clock; if not, the gap is clock or stalls, not schedule.

The kernel's buffers are never initialised (their benchmark does not check results either),
so this measures throughput only, never correctness.

    python tools/tnzr_repro.py fetch
    python tools/tnzr_repro.py build
    python tools/tnzr_repro.py run --out results/aie/tnzr_bf16_32x32x32_repro_desktop2_<ts>.log

Before `run`: `xrt-smi examine -r aie-partitions` must show no hardware contexts, and another
session's NPU work must not be in flight (CLAUDE.md, "Three sessions can be on this one NPU").
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

REPO = "scalable-analyses/xdna"
FILES = ("src/XDNA1.mlir", "src/tensor_kernel_32x32x32_bf16_bf16_fp32.s", "Makefile", "src/driver_kernel.cpp")
KERNEL = "tensor_kernel_32x32x32_bf16_bf16_fp32"
IRON = Path.home() / "mlir-aie" / "ironenv"
PEANO = IRON / "Lib" / "site-packages" / "llvm-aie"
AIECC = IRON / "Scripts" / "aiecc.exe"
XRT_SMI = Path(r"C:\Windows\System32\AMD\xrt-smi.exe")
FLOP_PER_CALL = 2 * 32 * 32 * 32
CALLS_PER_DISPATCH = 1_000_000
CLOCK_GHZ = 1.80  # MEASURED, docs/SILICON.md


def head_commit() -> str:
    url = "https://api." + "github.com/repos/" + REPO + "/commits/HEAD"
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)["sha"]


def workdir(commit: str) -> Path:
    return ROOT / "scratch" / "tnzr-xdna" / commit


def cmd_fetch(args) -> int:
    commit = args.commit or head_commit()
    d = workdir(commit)
    d.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        url = f"https://raw.githubusercontent.com/{REPO}/{commit}/{f}"
        with urllib.request.urlopen(url, timeout=60) as r:
            (d / Path(f).name).write_bytes(r.read())
        print(f"fetched {f} -> {(d / Path(f).name).relative_to(ROOT).as_posix()}")
    (ROOT / "scratch" / "tnzr-xdna" / "LATEST").write_text(commit)
    print(f"commit {commit}")
    return 0


def latest() -> str:
    return (ROOT / "scratch" / "tnzr-xdna" / "LATEST").read_text().strip()


def run_cmd(cmd: list[str], cwd: Path) -> None:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stdout[-4000:] + p.stderr[-4000:])
        raise SystemExit(f"FAILED ({p.returncode}): {' '.join(str(c) for c in cmd)}")


def static_count(obj: Path) -> dict:
    """Bundles in the kernel, and the dynamic count once its hardware loop's trip count is applied."""
    import aie_disasm
    objdump = aie_disasm.find_objdump(None)
    secs = aie_disasm.parse(aie_disasm.disassemble(str(obj), objdump))
    bundles = [b for s in secs for b in s.bundles]
    text = "\n".join(";".join(b.fields) for b in bundles)
    lc = re.search(r"movxm\s+lc,\s*#(0x[0-9a-f]+|\d+)", text)
    trip = int(lc.group(1), 0) if lc else 1
    vmac = sum(1 for b in bundles if any(f.startswith("vmac") for f in b.fields))
    # Hand-written assembly names its hardware loop `.l_start`/`.l_end` (the compiler's
    # `.L_LEnd*` convention, which aie_disasm.find_loops keys on, does not apply).
    names = [b.label or "" for b in bundles]
    s_i = next((i for i, n in enumerate(names) if n.endswith("l_start")), None)
    e_i = next((i for i, n in enumerate(names) if n.endswith("l_end")), None)
    body = bundles[s_i:e_i + 1] if s_i is not None and e_i is not None else []
    loop_b = len(body)
    loop_v = sum(1 for b in body if any(f.startswith("vmac") for f in b.fields))
    return {
        "static_bundles": len(bundles),
        "static_vmac_bundles": vmac,
        "hw_loop_bundles": loop_b,
        "hw_loop_trip": trip,
        "dynamic_bundles": len(bundles) + (trip - 1) * loop_b,
        "dynamic_vmac_bundles": vmac + (trip - 1) * loop_v,
    }


def cmd_build(args) -> int:
    commit = args.commit or latest()
    d = workdir(commit)
    b = d / "build"
    b.mkdir(exist_ok=True)
    obj = b / f"{KERNEL}.o"
    run_cmd([str(PEANO / "bin" / "clang++.exe"), "--target=aie2-none-unknown-elf", "-c",
             str(d / f"{KERNEL}.s"), "-o", str(obj)], cwd=b)
    # Their Makefile: aiecc.py --alloc-scheme=basic-sequential --no-compile-host --no-xchesscc
    # --no-xbridge --peano $PEANO --aie-generate-npu-insts --npu-insts-name=insts_XDNA1.bin
    # --aie-generate-xclbin --xclbin-name=final_XDNA1.xclbin XDNA1.mlir. mlir-aie v1.4.2's
    # aiecc uses Peano and no host compile by default, and names its outputs with --get-*.
    run_cmd([str(AIECC), str(d / "XDNA1.mlir"), "--alloc-scheme=basic-sequential", f"--peano={PEANO}",
             "--get-npu-insts", "--npu-insts-name=insts_XDNA1.bin",
             "--get-xclbin", "--xclbin-name=final_XDNA1.xclbin", f"--tmpdir={b / 'prj'}"], cwd=b)
    sc = static_count(obj)
    (b / "static.json").write_text(json.dumps(sc, indent=2))
    print(f"built {(b / 'final_XDNA1.xclbin').relative_to(ROOT).as_posix()}; static: {sc}")
    return 0


def xrt_versions() -> list[str]:
    out = subprocess.run([str(XRT_SMI), "examine"], capture_output=True, text=True).stdout
    keep = [ln.strip() for ln in out.splitlines()
            if re.search(r"^\s*(Version|NPU Driver Version|NPU Firmware Version)\s*:", ln)]
    return keep


def partitions() -> str:
    out = subprocess.run([str(XRT_SMI), "examine", "-r", "aie-partitions"], capture_output=True, text=True).stdout
    return "idle" if "No hardware contexts running" in out else "BUSY:\n" + out


def cmd_run(args) -> int:
    commit = args.commit or latest()
    b = workdir(commit) / "build"
    sc = json.loads((b / "static.json").read_text())
    L = []
    say = lambda s="": (print(s), L.append(s))  # noqa: E731
    say("Hello XDNA! XDNA1 bf16 kernel reproduction (tools/tnzr_repro.py)")
    say(f"date: {dt.datetime.now().isoformat(timespec='seconds')}  host: {os.environ.get('COMPUTERNAME')}")
    say(f"reference: https://tnzr.org/xdna/xdna1_kernel.html (398 GFLOPS, Ryzen 7 8700G, --pmode turbo)")
    say(f"source: github.com/{REPO} @ {commit} (fetched to scratch/, not vendored)")
    say(f"kernel: {KERNEL}.s assembled by Peano {PEANO.name} ({subprocess.run([str(PEANO / 'bin' / 'clang.exe'), '--version'], capture_output=True, text=True).stdout.splitlines()[0]})")
    say(f"aiecc: mlir-aie ironenv v1.4.2 ({AIECC.name})")
    for v in xrt_versions():
        say(f"xrt-smi: {v}")
    say(f"pmode: {args.pmode_note}")
    before = partitions()
    say(f"partitions before: {before}")
    if before != "idle" and not args.force:
        say("REFUSED: device not idle")
        return 2
    say(f"static (aie_disasm on the assembled object): {sc}")

    from ignite_xdna.runtime.driver import XrtSiliconHarness
    times = []
    with XrtSiliconHarness() as h:
        h.load_xclbin(str(b / "final_XDNA1.xclbin"), kernel_name="MLIR_AIE")
        bo_instr, n = h.create_instruction_bo(str(b / "insts_XDNA1.bin"))
        bo_io = h.create_host_bo(4, 3)
        bo_io.sync(h.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        for i in range(args.warmups + args.iterations):
            t0 = time.perf_counter()
            run, state = h.dispatch_kernel(bo_instr, n, bo_io, timeout_ms=args.timeout_ms)
            t1 = time.perf_counter()
            if "COMPLETED" not in str(state):
                say(f"dispatch {i}: {state} -- aborting")
                break
            if i >= args.warmups:
                times.append(t1 - t0)
        del run, bo_io, bo_instr
    after = partitions()
    say(f"partitions after close: {after}")
    if not times:
        say("NO TIMED DISPATCHES")
        return 1
    flop = FLOP_PER_CALL * CALLS_PER_DISPATCH
    say("")
    say("| iteration | time (ms) | GFLOPS |")
    say("|---|---|---|")
    for i, t in enumerate(times):
        say(f"| {i} | {t * 1e3:.3f} | {flop / t / 1e9:.1f} |")
    mean = sum(times) / len(times)
    say("")
    say(f"mean {mean * 1e3:.3f} ms/dispatch -> {flop / mean / 1e9:.1f} GFLOPS "
        f"(min {min(times) * 1e3:.3f} ms -> {flop / min(times) / 1e9:.1f}, max {max(times) * 1e3:.3f} ms -> {flop / max(times) / 1e9:.1f})")
    cyc = mean * CLOCK_GHZ * 1e9 / CALLS_PER_DISPATCH
    say(f"implied cycles per kernel call at {CLOCK_GHZ} GHz: {cyc:.1f} "
        f"(static: {sc['dynamic_bundles']} kernel bundles, {sc['dynamic_vmac_bundles']} issuing vmac.f; "
        f"the difference is the calling loop, call/return and any stall)")
    say(f"per-core peak at {CLOCK_GHZ} GHz: {256 * CLOCK_GHZ:.1f} GFLOPS -> measured is "
        f"{100 * flop / mean / 1e9 / (256 * CLOCK_GHZ):.1f}% of it; reference 398 -> "
        f"{100 * (flop / mean / 1e9) / 398:.1f}% of the reference")
    if args.out:
        Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--commit")
    bld = sub.add_parser("build")
    bld.add_argument("--commit")
    r = sub.add_parser("run")
    r.add_argument("--commit")
    r.add_argument("--warmups", type=int, default=3)
    r.add_argument("--iterations", type=int, default=10)
    r.add_argument("--timeout-ms", type=int, default=10000)
    r.add_argument("--pmode-note", default="default (not changed by this run; docs/SILICON.md measured 1.80 GHz in default)")
    r.add_argument("--force", action="store_true", help="run even if xrt-smi reports contexts")
    r.add_argument("--out")
    args = ap.parse_args(argv)
    return {"fetch": cmd_fetch, "build": cmd_build, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

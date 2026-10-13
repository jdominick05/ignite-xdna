#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Is a graph-engine container's data movement issue-bound or transport-bound, layer by layer? Add inert
instruction ops to every shim task and see which layers slow down.

    python tools/issue_inject_probe.py make --container build/x.ignite --variant bw1     (writes x_inj_bw1.ignite)
    python tools/issue_inject_probe.py classify --container build/x.ignite --onnx models/x.onnx --json cls.json
    python tools/issue_inject_probe.py analyse --cls cls.json --base base.json --variant bw1=v1.json ...

The array has one serial issue path for shim tasks (~1.42 us per task, tools/task_issue_probe.py on one column
against four), and the engine's per-layer NOP times fit two shapes about equally well: ``max(transport, issue)``
and ``transport + issue``. Under the first, extra issue cost slows only the layers whose issue already outruns
their transport; under the second it slows every layer in proportion to its tasks. This changes ONLY the issue
cost: the bytes moved, the packets, the schedule and the core program are untouched.

``make`` inserts, immediately before every shim task's BD write (a BLOCKWRITE to a shim BD register block), one of:
    bw1   one BLOCKWRITE of 8 zero words to shim BD 15 of the same column           (+48 B, +1 op per task)
    bw2   two such BLOCKWRITEs                                                         (+96 B, +2 ops)
    mw1   one MASKWRITE with mask 0 (writes nothing) to BD 15's first word            (+28 B, +1 op)
BD 15 must never be written or pushed by the container (checked; the emitter uses ids 0..11 on YOLOv8s). The
injected ops are never WRITEs, so ``split_instruction_stream``'s count of task pushes - and with it
``tools/engine_layer_timing.py``'s per-layer cut - is unchanged; an injected op before a layer's first task lands
in that layer's piece. Outputs must stay byte-identical (check with ``tools/verify_engine_container.py``).

``classify`` re-lowers the container and labels each layer issue-bound when ``tasks x C > transport`` (C the
1.20 us per task the max fit found; transport = the busiest shim stream's bytes at 6.9 GB/s). ``analyse`` reads
``engine_layer_timing`` JSONs and reports, per variant and arm, the added NOP and real net time per added task on
each class and their ratio (transport-bound over issue-bound). Written on Desktop 2 (Phoenix), 2026-10-13; what
has been run is in its logs.
"""
import argparse
import json
import statistics
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ignite_xdna.compiler.scheduler import (OP_BLOCKWRITE, OP_MASKWRITE, TXN_HEADER_BYTES,  # noqa: E402
                                           parse_transaction_stream)
from ignite_xdna.compiler.serializer import ARCH_XDNA1_PHOENIX, IgniteModelReader, IgniteModelWriter  # noqa: E402

SHIM_BD_BASE = 0x1D000
SHIM_BD_STRIDE = 0x20
SHIM_BD_COUNT = 16
DUMMY_BD = 15
COL_SHIFT = 25
LOCAL_MASK = (1 << COL_SHIFT) - 1
C_ISSUE_MS = 1.20e-3
RATE = 6.9e9
VARIANTS = {"bw1": ("bw",), "bw2": ("bw", "bw"), "mw1": ("mw",)}


def _shim_bd(op):
    """The shim BD id a BLOCKWRITE writes, or None."""
    if op["op"] != "BLOCKWRITE":
        return None
    a = op["addr"] & LOCAL_MASK
    if SHIM_BD_BASE <= a < SHIM_BD_BASE + SHIM_BD_STRIDE * SHIM_BD_COUNT and (a - SHIM_BD_BASE) % SHIM_BD_STRIDE == 0:
        return (a - SHIM_BD_BASE) // SHIM_BD_STRIDE
    return None


def _dummy(kind, col_addr_bits, col_row):
    addr = col_addr_bits | (SHIM_BD_BASE + SHIM_BD_STRIDE * DUMMY_BD)
    if kind == "bw":
        words = [0] * 8
        return struct.pack("<4I", OP_BLOCKWRITE, col_row, addr, 16 + 4 * len(words)) + struct.pack("<8I", *words)
    return struct.pack("<7I", OP_MASKWRITE, col_row, addr, 0, 0, 0, 28)


def make(container: Path, variant: str) -> Path:
    with IgniteModelReader(container) as r:
        manifest = dict(r.manifest)
        blobs = {n: r.get_blob_bytes(n) for n in ("engine.xclbin", "insts.bin", "wpackets.bin")}
    insts = blobs["insts.bin"]
    ops = parse_transaction_stream(insts)
    used = {_shim_bd(o) for o in ops} - {None}
    touched = any((o["addr"] & LOCAL_MASK) - SHIM_BD_BASE in range(SHIM_BD_STRIDE * DUMMY_BD,
                                                                  SHIM_BD_STRIDE * (DUMMY_BD + 1))
                  for o in ops if o["op"] in ("BLOCKWRITE", "MASKWRITE", "WRITE"))
    if DUMMY_BD in used or touched:
        raise SystemExit(f"shim BD {DUMMY_BD} is used by the container; pick another dummy id")
    major, minor, num_ops, _size = struct.unpack("<4I", insts[:TXN_HEADER_BYTES])
    body, added = bytearray(), 0
    for o in ops:
        if _shim_bd(o) is not None:
            for kind in VARIANTS[variant]:
                body += _dummy(kind, o["addr"] & ~LOCAL_MASK, o["col_row"])
                added += 1
        body += insts[o["offset"]:o["offset"] + o["size"]]
    new = struct.pack("<4I", major, minor, num_ops + added, TXN_HEADER_BYTES + len(body)) + bytes(body)
    check = parse_transaction_stream(new)
    assert sum(o["op"] == "WRITE" for o in check) == sum(o["op"] == "WRITE" for o in ops)
    manifest["model_name"] = f"{manifest.get('model_name', container.stem)}_inj_{variant}"
    ge = dict(manifest["graph_engine"])
    ge["issue_injection"] = {"variant": variant, "ops_added": added, "bytes_added": len(new) - len(insts),
                             "dummy_shim_bd": DUMMY_BD}
    ge["insts_bytes"] = len(new)
    manifest["graph_engine"] = ge
    out = container.with_name(f"{container.stem}_inj_{variant}.ignite")
    w = IgniteModelWriter(manifest_meta=manifest, arch_id=ARCH_XDNA1_PHOENIX)
    w.add_blob("engine.xclbin", blobs["engine.xclbin"], content_type="xclbin")
    w.add_blob("insts.bin", new, content_type="npu_instructions")
    w.add_blob("wpackets.bin", blobs["wpackets.bin"], content_type="weight_packets")
    w.write(out)
    print(f"[make] {out.name}: +{added} ops, +{len(new) - len(insts):,} B "
          f"({len(insts):,} -> {len(new):,}); shim BD ids used {sorted(used)}")
    return out


def classify(container: Path, onnx: Path) -> list:
    from engine_layer_timing import relower
    from ignite_xdna.compiler.engine_sequence import program_task_count
    with IgniteModelReader(container) as r:
        manifest = dict(r.manifest)
    _, _, scheds, _ = relower(onnx, manifest)
    rows = []
    for s in scheds:
        tasks = sum(program_task_count(p) for p in s.programs)
        busiest = 0
        for prog in s.programs:
            per = {"a": 0, "w": 0, "o": 0}
            for it in prog:
                k = it[0]
                if k == "A":
                    per["a"] += it[1].nbytes
                elif k == "a":
                    per["a"] += sum(p.nbytes for p in it[1])
                elif k == "w":
                    per["w"] += it[2]
                elif k == "W":
                    per["w"] += it[1].nbytes
                elif k == "o":
                    per["o"] += it[1].nbytes
            busiest = max(busiest, max(per.values()))
        transport_ms = busiest / RATE * 1e3
        rows.append({"name": s.name, "tasks": tasks, "transport_ms": transport_ms,
                     "issue_ms": tasks * C_ISSUE_MS, "issue_bound": tasks * C_ISSUE_MS > transport_ms})
    return rows


def _net(paths: str, arm: str) -> dict:
    """Per-layer net ms of one container, averaged over the '+'-joined engine_layer_timing JSONs (its passes)."""
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths.split("+")]
    names = [L["name"] for L in runs[0]["arms"][arm]["layers"]]
    return {n: statistics.fmean(next(L["net_ms"] for L in r["arms"][arm]["layers"] if L["name"] == n)
                                for r in runs) for n in names}


def sit(onnx: Path, image: Path, containers: list, passes: int, rounds: int, out_dir: Path) -> None:
    """engine_layer_timing.py --nop on every container, one child process each, the order reversed every pass."""
    import subprocess
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in range(1, passes + 1):
        for spec in (containers[::-1] if p % 2 == 0 else containers):
            name, path = spec.split("=", 1)
            js = out_dir / f"{name}_p{p}.json"
            cmd = [sys.executable, str(ROOT / "tools" / "engine_layer_timing.py"), "--onnx", str(onnx),
                   "--container", path, "--image", str(image), "--rounds", str(rounds), "--warmup", "10",
                   "--empty", "5", "--nop", "--json", str(js)]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            tail = [ln[:400] for ln in proc.stdout.splitlines() if ln.startswith("SUMMARY")]
            print(f"[sit] pass {p} {name}: rc {proc.returncode}; " + " | ".join(tail), flush=True)
            if proc.returncode != 0:
                print(proc.stdout[-2000:], proc.stderr[-2000:], sep="\n")
                raise SystemExit(f"[sit] {name} failed in pass {p}; stopping - do not retry in a loop")


def analyse(cls_path: Path, base: Path, variants: list) -> None:
    cls = json.loads(cls_path.read_text(encoding="utf-8"))
    per_task = {"bw1": 1, "bw2": 2, "mw1": 1}
    for arm in ("nop", "real"):
        b = _net(base, arm)
        for spec in variants:
            name, path = spec.split("=", 1)
            v = _net(path, arm)
            out = {}
            for label, want in (("issue-bound", True), ("transport-bound", False)):
                rows = [c for c in cls if c["issue_bound"] == want]
                d = sum(v[c["name"]] - b[c["name"]] for c in rows)
                t = sum(c["tasks"] for c in rows)
                out[label] = (d, t, d / t * 1e3 if t else float("nan"))
            ib, tb = out["issue-bound"], out["transport-bound"]
            ratio = tb[2] / ib[2] if ib[2] else float("nan")
            print(f"[{arm}] {name}: issue-bound {ib[0] * 1e3:+8.1f} us over {ib[1]} tasks = {ib[2]:+.3f} us/task "
                  f"({ib[2] / per_task.get(name, 1):+.3f} us per added op); transport-bound {tb[0] * 1e3:+8.1f} us "
                  f"over {tb[1]} tasks = {tb[2]:+.3f} us/task; ratio TB/IB {ratio:.2f}; total "
                  f"{(ib[0] + tb[0]) * 1e3:+.1f} us")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--container", type=Path, required=True)
    m.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    c = sub.add_parser("classify")
    c.add_argument("--container", type=Path, required=True)
    c.add_argument("--onnx", type=Path, required=True)
    c.add_argument("--json", type=Path, required=True)
    s = sub.add_parser("sit")
    s.add_argument("--onnx", type=Path, required=True)
    s.add_argument("--image", type=Path, required=True)
    s.add_argument("--container", action="append", required=True, metavar="NAME=PATH")
    s.add_argument("--passes", type=int, default=2)
    s.add_argument("--rounds", type=int, default=20)
    s.add_argument("--out-dir", type=Path, required=True)
    a = sub.add_parser("analyse")
    a.add_argument("--cls", type=Path, required=True)
    a.add_argument("--base", required=True, help="engine_layer_timing JSON(s) of the uninjected container, '+'-joined")
    a.add_argument("--variant", action="append", required=True, metavar="NAME=JSON[+JSON]")
    args = ap.parse_args()
    if args.cmd == "make":
        make(args.container, args.variant)
    elif args.cmd == "sit":
        sit(args.onnx, args.image, args.container, args.passes, args.rounds, args.out_dir)
    elif args.cmd == "classify":
        rows = classify(args.container, args.onnx)
        n = sum(r["issue_bound"] for r in rows)
        print(f"[classify] {len(rows)} layers, {n} issue-bound at {C_ISSUE_MS * 1e3:.2f} us/task; tasks "
              f"{sum(r['tasks'] for r in rows if r['issue_bound'])} issue-bound, "
              f"{sum(r['tasks'] for r in rows if not r['issue_bound'])} transport-bound")
        args.json.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    else:
        analyse(args.cls, args.base, args.variant)


if __name__ == "__main__":
    main()

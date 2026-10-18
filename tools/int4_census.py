#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""W4V step 1: the int4 byte census of the vision models. Report-only; it decides nothing.

    python tools/int4_census.py --assets <checkout holding build/, models/, scratch/> [--log results/int4/<name>.log]

No device, no hardware context. The containers and models are git-ignored, so they are read from
``--assets`` and every file read is listed with its size and sha256. The compiler that lowers and
schedules them is this checkout's own src/ (asserted).

Per engine container it rebuilds the schedule the container was compiled with (``lower_yolov8n`` with the
host regions the container's manifest names, ``plan_workspace``, ``schedule_graph``), counts every DMA item
of one frame as ``tools/engine_stream_report.py`` counts it, and walks every weight packet as
``tools/int4_bytes_gate.py`` (gate B, frozen, imported not edited) prices it. It reads out, per the W4V
plan's step 1:

  1. int4 over the trimmed frame: int4_ms / (dispatch - trim_ms); the trim is UNBUILT
  2. the frame with the MemTile activation ring on (6 slots, the width measured on silicon), in bytes
     only; no dispatch is priced with it here
  3. the transport rate the ms are priced at, and what measured it
  4. weights resident on-chip: conv-weight bytes from the ONNX file at int8 and int4, the engine's unique
     packets at int8 and int4, the reachable 3,145,728 B beside them, and the ceiling
     (weight_ms - trim_ms + weight_fills x 4 ops x 145 ns) / (dispatch - trim_ms)
  5. YOLO11n, YOLO26n and YOLO-World v2 beside gate B's five
  6. activation and drain bytes per frame beside the weight bytes
  7. AMD-stack models: weight against activation elements from the ONNX shapes (a DERIVED proxy)

Every ms is DERIVED: bytes at engine_stream_report's 26.8 GB/s, as if the frame were bound by that
transport. Each denominator is a MEASURED NPU dispatch cited per row; for a model with host segments it
excludes them, as the walk (NPU layers only) does.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import engine_stream_report as esr  # noqa: E402
import int4_bytes_gate as ibg  # noqa: E402
from ignite_xdna.compiler import engine_emulator as em  # noqa: E402
from ignite_xdna.compiler import engine_schedule as es  # noqa: E402
from ignite_xdna.compiler.graph_ir import lower_yolov8n  # noqa: E402

RING_SLOTS = 6  # results/aie/notes_memtile_activation_ring.md:1111, :1132 (yolov8n, yolov8s)
REACHABLE = 16 * 65536 + 4 * 524288  # 3,145,728 B, DERIVED (docs/SILICON.md:182, :186)
K_CONV_YOLOV8N = 3_146_160  # docs/SILICON.md:186; results/aie/silicon_weight_storage_desktop2_20260919.log
K_SHA_YOLOV8N = "f02e86ba1bf61ff3fdc159e29b15259567e279df06f68c6d9adeb966fe56c885"
MB = "docs/MODEL_ZOO_BENCHMARKS.md"

# name, ONNX, containers (the first is walked against; any other must match it), MEASURED NPU dispatch (ms), source
ENGINE = [
    ("YOLOv8n", "models/yolov8n_cut_xint8.onnx", ["build/yolov8n_full.ignite"], 7.161,
     f"{MB}:297 (live_ignition dispatch; gate B's denominator)"),
    ("YOLOv8s", "models/yolov8s_cut_xint8.onnx", ["build/yolov8s.ignite"], 16.609,
     f"{MB}:298 (live_ignition dispatch; gate B's denominator)"),
    ("YOLOv8n-pose", "models/yolov8n-pose_cut_xint8.onnx", ["build/yolov8n_pose.ignite"], 7.548,
     f"{MB}:388 (NPU dispatch; gate B's denominator)"),
    ("SESR-M7", "models/sesr_m7_xint8.onnx", ["build/sesr_m7.ignite"], 4.208,
     f"{MB}:186 (dispatch; gate B's denominator)"),
    ("resnet50_head", "build/test_resnet50_head.onnx", ["build/resnet50_head.ignite"], 1.142,
     "docs/BENCHMARKS.md:10394 (NPU dispatch; gate B's denominator, which it cited at :10125 on its commit)"),
    ("YOLO11n", "models/yolo11n_cut_xint8.onnx", ["build/yolo11n.ignite"], 8.362,
     "docs/BENCHMARKS.md:9013 (Ignition dispatch, the smallest found for this container: 8.380 at :9016, 8.372 "
     f"at :8657, 8.452 / 8.442 at {MB}:327-328; host segment excluded)"),
    ("YOLO11n-core", "models/yolo11n_cut_xint8.onnx", ["build/yolo11n_core.ignite"], 8.692,
     "docs/BENCHMARKS.md:9017 (Ignition dispatch, the smallest found for this container: 8.748 at :9014, "
     f"8.784 / 8.708 at {MB}:334; host segment excluded)"),
    ("YOLO26n", "scratch/yolo26/yolo26n_cut_xint8.onnx", ["build/yolo26n_r2.ignite", "build/yolo26n.ignite"], 8.831,
     "results/aie/yolo26n_ignition_sitting_20260921.log:92 (control arm, yolo26n_r2.ignite; the smaller of "
     "8.831 / 8.833 at :92, :132; host segments excluded)"),
    ("YOLO-World v2", "models/yolov8s-worldv2_cut_xint8_gptqcv2.onnx", ["build/yolow.ignite"], 18.720,
     "results/aie/yolow_gptq/verify_gptqcv2.log:73 (dispatch mean over 20 of yolow_gptqcv2.ignite, compiled "
     "from the same ONNX with the same four host regions, docs/MODEL_ZOO_BENCHMARKS.md:457; host excluded, :74)"),
]
PINNED = {"YOLOv8n": 26_303_744, "YOLOv8s": 83_618_816}  # commit 5162671's body, as gate B
GATE_B_LOG = "results/aie/int4_engine_bytes_gate_desktop2_20260923.log"
GATE_B_INT4 = {"YOLOv8n": 5_385_216, "YOLOv8s": 17_367_552, "YOLOv8n-pose": 5_587_968, "SESR-M7": 1_539_072,
               "resnet50_head": 1_048_576}  # its 'int4 (variable-size packets)' rows, :25, :40, :54, :68, :82
NOTES = {
    "YOLO26n": ["build/yolo26n.ignite's source ONNX was lost with a removed worktree; scratch/yolo26/yolo26n_cut_xint8.onnx",
                "is the 2026-09-21 re-derivation, compiled as build/yolo26n_r2.ignite with an identical lowering",
                "(results/aie/yolo26n_requant_vs_amd_20260921.log:11-13, :17-28). The walk is checked against both."],
}

# name, ONNX (the pipeline's own source file), what it is
AMD = [
    ("ResNet50", "models/resnet50_fp32.onnx", "pipelines/resnet50"),
    ("YOLOv6n", "models/yolov6n_cut_xint8.onnx", "pipelines/yolov6n"),
    ("MiDaS small", "models/midas_small_cut.onnx", "pipelines/midas"),
    ("FastDepth", "models/fastdepth_fp32.onnx", "pipelines/fastdepth"),
    ("Real-ESRGAN compact", "models/realesrgan_compact_r64_fp32.onnx", "pipelines/realesrgan"),
    ("BiSeNetV2", "models/bisenetv2_fp32.onnx", "pipelines/bisenetv2"),
    ("YOLOv8n (calibration)", "models/yolov8n_cut_xint8.onnx", "engine row above"),
    ("YOLOv8s (calibration)", "models/yolov8s_cut_xint8.onnx", "engine row above"),
]
PROXY_OPS = ("Conv", "ConvTranspose", "Gemm", "MatMul")

_HOME = str(Path.home())
_USER = str(Path.home().parent).replace("\\", "/") + "/<user>"


def scrub(s: str) -> str:
    return s.replace(_HOME, _USER).replace(_HOME.replace("\\", "/"), _USER)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def graph_engine(path: Path) -> dict:
    return ibg.manifest(path).get("graph_engine", {})


def host_regions(ge: dict) -> tuple:
    return tuple(s["name"] for s in ge.get("segments") or [] if isinstance(s, dict) and s.get("kind") == "host")


def walk(scheds, store) -> dict:
    """One frame: DMA tasks and bytes per kind (engine_stream_report's count) and every weight packet sent
    (gate B's pricing), plus the unique packets the frame touches."""
    tot, unknown = esr.empty(), set()
    n = dict(sent=0, conv_sent=0, bytes=0, layout=0, conv_layout=0, overruns=0, unaligned=0)
    cache = {}
    for s in scheds:
        for prog in s.programs:
            for it in prog:
                if it[0] in esr.RING_ITEMS:
                    continue
                kind = esr.KINDS.get(it[0])
                if kind is None:
                    unknown.add(it[0])
                    continue
                tot[kind]["tasks"] += len(it[1]) if it[0] == "a" else 1
                tot[kind]["bytes"] += it[2] if it[0] == "w" else sum(p.nbytes for p in esr.patterns(it[1:]))
                if kind != "weight":
                    continue
                runs = [(it[1], it[2])] if it[0] == "w" else \
                    [r for p in esr.patterns(it[1:]) for r in ibg.pattern_runs(p)]
                for off, nb in runs:
                    if off % em.W_BYTES or nb % em.W_BYTES:
                        n["unaligned"] += 1
                    for k in range(nb // em.W_BYTES):
                        o = off + k * em.W_BYTES
                        if o not in cache:
                            cache[o] = ibg.packet_cost(store.packet_at(o))
                        c = cache[o]
                        n["sent"] += 1
                        n["bytes"] += em.W_BYTES
                        n["layout"] += c["layout"]
                        n["overruns"] += c["overrun"]
                        if c["conv"]:
                            n["conv_sent"] += 1
                            n["conv_layout"] += c["layout"]
    n.update(totals=tot, unknown=sorted(unknown), w_fills=sum(s.w_fills for s in scheds),
             store_bytes=store.nbytes, unique=len(cache), u_conv=sum(c["conv"] for c in cache.values()),
             u_layout=sum(c["layout"] for c in cache.values()),
             u_conv_layout=sum(c["layout"] for c in cache.values() if c["conv"]))
    return n


def resolve(name: str, producers: dict) -> str:
    p = producers.get(name)
    return p.input[0] if p is not None and p.op_type == "DequantizeLinear" else name


def layer_weights(model: onnx.ModelProto) -> dict:
    """silicon_weight_bytes_probe's method for Conv (input 1, through its DequantizeLinear, to the
    initializer), and the same for any ConvTranspose / Gemm / MatMul operand that is an initializer;
    elements counted (int8 = 1 B each), int4 = ceil(elements / 2) per tensor."""
    inits = {t.name: t for t in model.graph.initializer}
    producers = {o: nd for nd in model.graph.node for o in nd.output}
    conv, other, missing = {}, {}, 0
    for nd in model.graph.node:
        if nd.op_type == "Conv" and len(nd.input) >= 2:
            key = resolve(nd.input[1], producers)
            if key in inits:
                conv[key] = inits[key]
            else:
                missing += 1
        elif nd.op_type in PROXY_OPS:
            for name in nd.input[:2]:
                key = resolve(name, producers)
                if key in inits:
                    other[key] = inits[key]
    numel = lambda d: [int(np.prod(t.dims)) for t in d.values()]  # noqa: E731
    return dict(convs=sum(nd.op_type == "Conv" for nd in model.graph.node), conv_tensors=len(conv), missing=missing,
                conv_elements=sum(numel(conv)), other_tensors=len(other), other_elements=sum(numel(other)),
                int4=sum(-(-e // 2) for e in numel(conv) + numel(other)),
                dtypes=sorted({onnx.helper.tensor_dtype_to_np_dtype(t.data_type).name
                               for t in list(conv.values()) + list(other.values())}))


def operand_proxy(model: onnx.ModelProto) -> dict:
    """Per Conv / ConvTranspose / Gemm / MatMul: weight elements (an initializer operand, through its
    DequantizeLinear) against activation elements (every other operand plus the output), shapes from ONNX
    shape inference. A layer's input and output are both counted, as if each went to DDR once."""
    mi = onnx.shape_inference.infer_shapes(model)
    shapes = {}
    for v in list(mi.graph.input) + list(mi.graph.value_info) + list(mi.graph.output):
        dims = [d.dim_value for d in v.type.tensor_type.shape.dim]
        if dims and all(d > 0 for d in dims):
            shapes[v.name] = int(np.prod(dims))
    inits = {t.name: int(np.prod(t.dims)) for t in model.graph.initializer}
    producers = {o: nd for nd in mi.graph.node for o in nd.output}
    w = a = unresolved = nodes = 0
    for nd in mi.graph.node:
        if nd.op_type not in PROXY_OPS:
            continue
        nodes += 1
        for name in list(nd.input[:2]) + [nd.output[0]]:
            key = resolve(name, producers)
            if key in inits and name not in nd.output:
                w += inits[key]
            elif name in shapes:
                a += shapes[name]
            else:
                unresolved += 1
    return dict(nodes=nodes, weight=w, activation=a, unresolved=unresolved)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--assets", type=Path, default=ROOT,
                    help="checkout that holds the git-ignored build/, models/ and scratch/ (default: this one)")
    ap.add_argument("--log", type=Path, default=None, help="also write the report here (UTF-8, LF)")
    args = ap.parse_args()
    if args.log and args.log.exists():
        raise SystemExit(f"refusing to overwrite existing log {args.log}")

    here = Path(es.__file__).resolve()
    if ROOT / "src" not in here.parents:
        raise SystemExit(f"engine_schedule resolved to {here}, not this checkout's src/ -- refusing")
    assets = args.assets.resolve()
    bw = esr.TRANSPORT_BYTES_PER_S
    ms = lambda b: b / bw * 1e3  # noqa: E731
    issue_s = esr.OPS_PER_TASK_ISSUE * esr.SECONDS_PER_OP

    git = lambda *a: subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    out = ["W4V step 1: the int4 byte census of the vision models (report-only under criterion (b); it decides nothing)",
           f"UTC: {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
           f"MACHINE: {platform.node()}",
           f"COMMIT: {git('rev-parse', '--short', 'HEAD')} ({git('rev-parse', '--abbrev-ref', 'HEAD')})",
           "COMMAND: python tools/int4_census.py --assets " + scrub(assets.as_posix())
           + (f" --log {args.log.as_posix()}" if args.log else ""),
           f"PYTHON: {platform.python_version()}, numpy {np.__version__}, onnx {onnx.__version__}",
           f"COMPILER: {here.relative_to(ROOT).as_posix()} (this checkout's src/, asserted)",
           f"ASSETS: {scrub(assets.as_posix())} -- build/, models/ and scratch/ are git-ignored there; every file",
           "  read from them is listed below with its sha256. No container records its source model's sha256",
           "  (none carries source_model_sha256), so each pairing rests on the manifest checks per row.",
           "",
           "SCOPE: no device, no hardware context. Default schedule (activation ring off) unless a line says ring.",
           "  Tags: every figure below is DERIVED (computed here from the files listed) except each row's dispatch,",
           "  tagged MEASURED with its cite, and what is tagged UNMEASURED.",
           f"TRANSPORT (item 3): every ms below is DERIVED at {bw / 1e9:.1f} GB/s (engine_stream_report.py:37), as if",
           "  the frame were bound by that transport. The rate is MEASURED by a no-compute transport probe: 78.7 MB",
           "  of fills in 2.94 ms as 52 or 772 tasks, fills and drains in parallel (docs/BENCHMARKS.md:8343-8344).",
           "  The engine's weight-fill rate inside its own schedule is UNMEASURED: no log isolates it.",
           f"  Task issue is priced at {esr.OPS_PER_TASK_ISSUE} ops x {esr.SECONDS_PER_OP * 1e9:.0f} ns "
           "(engine_sequence.py:229; engine_stream_report.py:38).",
           "DENOMINATORS: gate B's five keep gate B's dispatch, so those rows reproduce gate B; each new row takes",
           "  the smallest NPU dispatch found for its container. Host segments are excluded, as the walk excludes them.",
           "TRIM: 'trim' sends each weight packet as its declared layout instead of a fixed "
           f"{em.W_BYTES:,} B object. It is UNBUILT.",
           f"RESIDENCY: the reachable array is {REACHABLE:,} B (DERIVED, 16 x 64 KiB + 4 x 512 KiB; docs/SILICON.md:182,",
           "  :186), shared with the workspace; that sharing is not priced here. ONNX weight counts cover the whole",
           "  file, including layers a container carves to the host; engine packet counts cover its NPU layers only.",
           f"RING (item 2): {RING_SLOTS} slots, the width built on silicon, where it MEASURED slower: YOLOv8n 7.392 -> 10.052 ms,",
           "  YOLOv8s 16.828 -> 20.228 ms (results/aie/notes_memtile_activation_ring.md:1134-1135); the default stays 0",
           "  (engine_compile.py:339). Its bytes below price no time, and it cannot be combined with host regions",
           "  (engine_compile.py:369-375), so only the single-dispatch rows carry it.",
           ""]

    files, prov, prov7 = {}, [], []

    def read(rel: str, sink: list = prov) -> Path:
        p = assets / rel
        if rel not in files:
            files[rel] = p
            sink.append(f"  {rel:<52} {p.stat().st_size:>12,} B  sha256 {sha256(p)}")
        return p

    rows = []
    for name, onnx_rel, conts, disp, src in ENGINE:
        onnx_p = read(onnx_rel)
        ges = [(c, graph_engine(read(c))) for c in conts]
        hr = host_regions(ges[0][1])
        ir = lower_yolov8n(onnx_p, host_regions=hr)
        ws = es.plan_workspace(ir)
        r = walk(*es.schedule_graph(ir, ws, activation_ring=0))
        t = r["totals"]
        checks = [f"walk's packet bytes == its item bytes ({t['weight']['bytes']:,} B): "
                  f"{r['bytes'] == t['weight']['bytes']}"]
        good = r["bytes"] == t["weight"]["bytes"] and not r["unaligned"] and not r["overruns"] and not r["unknown"]
        if not hr:
            ref = esr.report(onnx_p, 0)["totals"]
            same = all(ref[k]["bytes"] == t[k]["bytes"] and ref[k]["tasks"] == t[k]["tasks"] for k in ref)
            checks.append(f"== engine_stream_report (weight, activation, drain tasks and bytes): {same}")
            good &= same
        else:
            checks.append(f"host regions from {conts[0]}'s manifest: {list(hr)} (engine_stream_report cannot "
                          "take host regions, so the manifest checks carry this row)")
        if name in PINNED:
            checks.append(f"== 5162671's {PINNED[name]:,} B: {r['bytes'] == PINNED[name]}")
            good &= r["bytes"] == PINNED[name]
        if name in GATE_B_INT4:
            checks.append(f"int4 bytes == gate B's ({GATE_B_INT4[name]:,} B, {GATE_B_LOG}): "
                          f"{r['conv_layout'] // 2 == GATE_B_INT4[name]}")
            good &= r["conv_layout"] // 2 == GATE_B_INT4[name]
        for c, ge in ges:
            mw, mf = ge.get("wpackets_bytes"), ge.get("weight_fills")
            ok = (mw, mf) == (r["store_bytes"], r["w_fills"]) and host_regions(ge) == hr
            checks.append(f"{c}: wpackets_bytes {mw:,} vs {r['store_bytes']:,}, weight_fills {mf} vs {r['w_fills']}, "
                          f"host regions {'same' if host_regions(ge) == hr else 'DIFFER'}" + (" (match)" if ok else " (DIFFERS)"))
            good &= ok
        ring = None
        if not hr:
            try:
                ring = walk(*es.schedule_graph(ir, ws, activation_ring=RING_SLOTS))
                ref6 = esr.report(onnx_p, RING_SLOTS)["totals"]
                ring["same"] = all(ref6[k]["bytes"] == ring["totals"][k]["bytes"] for k in ref6)
            except Exception as e:  # noqa: BLE001 -- a refused ring is a row, not a crash
                ring = {"error": f"{type(e).__name__}: {e}"}
        cw = layer_weights(onnx.load(str(onnx_p)))
        rows.append((name, onnx_rel, conts, disp, src, hr, r, ring, cw, checks, good))

    out.append("FILES READ (relative to ASSETS):")
    out += prov
    out.append("")

    for name, onnx_rel, conts, disp, src, hr, r, ring, cw, checks, good in rows:
        kind = f"{len(hr)} host segment(s)" if hr else "single dispatch"
        out.append(f"{name}  ({onnx_rel} -> {conts[0]}; {kind}; {r['sent']:,} weight packets sent per frame, "
                   f"{r['conv_sent']:,} conv; {r['unique']:,} unique)")
        out.append(f"  dispatch {disp:.3f} ms MEASURED, {src}")
        out += [f"  note: {line}" for line in NOTES.get(name, [])]
        for c in checks:
            out.append(f"  check: {c}")
        out.append(f"  check: unaligned runs {r['unaligned']}, overrunning packets {r['overruns']}, unknown item kinds {r['unknown']}")
        if not good:
            out += ["  CHECKS FAILED -- no figures for this row", ""]
            continue
        t = r["totals"]
        w, a, d = t["weight"]["bytes"], t["activation"]["bytes"], t["drain"]["bytes"]
        trim = r["bytes"] - (r["sent"] * ibg.HDR_BIAS + r["layout"])
        int4 = r["conv_layout"] // 2
        w_ms, trim_ms, int4_ms = ms(w), ms(trim), ms(int4)
        fills_ms = r["w_fills"] * issue_s * 1e3
        out.append(f"  6. weight bytes / frame          {w:>14,} B  {w_ms:7.3f} ms  ({t['weight']['tasks']:,} tasks)")
        out.append(f"  6. activation fill bytes / frame {a:>14,} B  {ms(a):7.3f} ms  ({t['activation']['tasks']:,} tasks)")
        out.append(f"  6. drain bytes / frame           {d:>14,} B  {ms(d):7.3f} ms  ({t['drain']['tasks']:,} tasks)")
        out.append(f"  6. weight share of the frame's DDR bytes  {w / (w + a + d):6.1%}")
        out.append(f"     trim to declared layout (UNBUILT)  {trim:>12,} B  {trim_ms:7.3f} ms")
        out.append(f"     int4, conv layout halved          {int4:>12,} B  {int4_ms:7.3f} ms  {int4_ms / disp:6.1%} of dispatch")
        out.append(f"  1. int4 over the trimmed frame: {int4_ms:.3f} / ({disp:.3f} - {trim_ms:.3f}) = "
                   f"{int4_ms / (disp - trim_ms):6.1%}")
        out.append(f"  4. residency ceiling: ({w_ms:.3f} - {trim_ms:.3f} + {r['w_fills']} fills x {esr.OPS_PER_TASK_ISSUE} x "
                   f"{esr.SECONDS_PER_OP * 1e9:.0f} ns = {fills_ms:.3f}) / ({disp:.3f} - {trim_ms:.3f}) = "
                   f"{(w_ms - trim_ms + fills_ms) / (disp - trim_ms):6.1%}")
        e8 = r["unique"] * ibg.HDR_BIAS + r["u_layout"]
        e4 = e8 - r["u_conv_layout"] // 2
        out.append(f"  4. ONNX weights: Conv {cw['conv_tensors']} tensors of {cw['convs']} Conv ({cw['missing']} unresolved) "
                   f"{cw['conv_elements']:,} B; ConvTranspose / Gemm / MatMul {cw['other_tensors']} tensors "
                   f"{cw['other_elements']:,} B; {cw['dtypes']}")
        out.append(f"     int8 {cw['conv_elements'] + cw['other_elements']:,} B, int4 {cw['int4']:,} B; reachable minus int4 "
                   f"{REACHABLE - cw['int4']:+,} B")
        out.append(f"  4. engine packets: {r['unique']:,} unique ({r['u_conv']:,} conv) of {r['store_bytes'] // em.W_BYTES:,} "
                   f"stored; header+bias+declared layout int8 {e8:,} B, int4 {e4:,} B; reachable minus int4 {REACHABLE - e4:+,} B")
        if ring is None:
            out.append("  2. ring: not built for a container with host segments (engine_compile.py:369-375)")
        elif "error" in ring:
            out.append(f"  2. ring ({RING_SLOTS} slots): schedule refused -- {ring['error']}")
        else:
            rt = ring["totals"]
            rw, ra, rd = rt["weight"]["bytes"], rt["activation"]["bytes"], rt["drain"]["bytes"]
            ri4 = ring["conv_layout"] // 2
            out.append(f"  2. ring ({RING_SLOTS} slots): weight {rw:,} B, activation {ra:,} B, drain {rd:,} B; weight share "
                       f"{rw / (rw + ra + rd):6.1%}; int4 {ri4:,} B = {ri4 / (rw + ra + rd):6.1%} of the frame's DDR bytes "
                       f"(== engine_stream_report at {RING_SLOTS}: {ring['same']}; no dispatch priced)")
        out.append("")

    out.append("RESIDENCY K-CHECK (docs/SILICON.md:186):")
    k = next(cw for n, _, _, _, _, _, _, _, cw, _, _ in rows if n == "YOLOv8n")
    ksha = sha256(files["models/yolov8n_cut_xint8.onnx"])
    out.append(f"  YOLOv8n Conv weights {k['conv_elements']:,} B == {K_CONV_YOLOV8N:,} B: {k['conv_elements'] == K_CONV_YOLOV8N}; "
               f"file sha256 == the probe log's: {ksha == K_SHA_YOLOV8N}")
    out.append("")

    out.append("7. AMD-STACK MODELS: weight against activation elements from the ONNX shapes (DERIVED proxy; int8 = 1 B each)")
    out.append(f"  ops counted: {', '.join(PROXY_OPS)}; every layer's inputs and output counted once, as if each went to DDR")
    for name, rel, what in AMD:
        p = read(rel, prov7)
        m = onnx.load(str(p))
        ins = [f"{i.name} {[dd.dim_value for dd in i.type.tensor_type.shape.dim]}" for i in m.graph.input
               if i.name not in {t.name for t in m.graph.initializer}]
        q = operand_proxy(m)
        wa = q["weight"] + q["activation"]
        out.append(f"  {name:<22} {rel} ({what}); input {', '.join(ins)}")
        out.append(f"  {'':<22} {q['nodes']} layers, weight {q['weight']:,}, activation {q['activation']:,} "
                   f"({q['unresolved']} unresolved); weight share {q['weight'] / wa:6.1%}; int4 saves "
                   f"{q['weight'] // 2:,} = {q['weight'] / 2 / wa:6.1%} of the proxy's bytes")
    out.append("")
    out.append("FILES READ FOR ITEM 7 (relative to ASSETS; the calibration rows' files are listed above):")
    out += prov7
    out.append("")
    out.append("NOT ESTABLISHED: any accuracy (step 2's screen is the gate under (b)); any dispatch with the ring or")
    out.append("the trim built; the engine's own weight-fill rate; whether the workspace leaves room for resident")
    out.append("weights. Every percentage above is a number, not a verdict: this census decides nothing.")

    text = "\n".join(out)
    print(text)
    if args.log:
        args.log.write_text(text + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

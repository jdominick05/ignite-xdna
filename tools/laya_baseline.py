#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Laya's latency on the CPU and the iGPU through ONNX Runtime: the baselines an NPU build of it would have to beat.

Laya (Convai Innovations, Apache 2.0) is a typed-decision model: ModernBERT-large plus a decision head, 421M
parameters, one forward pass per question, no text generated. This runs receptron/laya-onnx's fp32 export (or an int8
copy made by ``quantize``) on synthetic inputs of a fixed shape: token ids drawn with a fixed seed, a full-length
attention mask, four option markers in the last 192 tokens, question type 0 (choice). Latency depends on the shape,
not on the words, so no text is needed; nothing here judges a decision.

    python tools/laya_baseline.py platform --model M [--model M2]   ORT, providers, CPU, GPU, power scheme, file shas
    python tools/laya_baseline.py quantize --model M --out O         int8 copy: dynamic quantization, per-channel
                                                                     QInt8 weights, MatMuls with constant weights only
    python tools/laya_baseline.py dmlcopy --model M                  M_az0.onnx for DirectML: Reshape allowzero
                                                                     cleared, equivalence checked on the CPU
    python tools/laya_baseline.py placement --ep dml --model M       which provider ran each node (ORT profiling)
    python tools/laya_baseline.py check --ep E --model M --ref R     one call against R in fp32 on the CPU
    python tools/laya_baseline.py run --ep cpu|dml --model M --seq L --batch B --seconds S

``run`` prints ``[run] frame N`` after every call, so ``tools/energy_sitting.py`` can take the window and the package
power, and ``[summary]`` lines: latency over the timed calls (mean, median, min, p90, max), calls and tokens per
second, and the multiply-accumulates per call DERIVED from the graph's MatMul shapes. The int8 copy's accuracy is
unchecked: it is ORT's dynamic scheme, not the static one an NPU build would use.
"""
import argparse
import ctypes
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import winreg
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

VOCAB = 50368   # ModernBERT-large's vocabulary
HEAD = 192      # laya_config.json head_max_len: the question and its options sit in the last 192 tokens
OPTIONS = 4


def macs_per_call(model: Path, seq: int, batch: int) -> tuple[int, int]:
    """(weight MatMul MACs, activation x activation MatMul MACs) per call, DERIVED from the graph. A product with a
    constant 2-D [K, N] weight (MatMul, the int8 copy's MatMulInteger, Gemm) costs rows*K*N, rows being every dim but
    the last of its activation input as ONNX shape inference gives it with the inputs fixed at this shape: seq*batch in
    the encoder, the option count or 1 in the decision head. The attention products cost 2*seq*seq*1024 per layer
    (QK^T and AV, 16 heads x 64), one layer per Softmax over the sequence."""
    import onnx
    m = onnx.load(str(model), load_external_data=False)
    del m.graph.value_info[:]   # the export's value_info conflicts with inference (see cmd_quantize)
    fixed = {"batch": batch, "seq": seq, "options": OPTIONS}
    for t in m.graph.input:
        for d in t.type.tensor_type.shape.dim:
            if d.dim_param in fixed:
                d.dim_value = fixed[d.dim_param]
    m = onnx.shape_inference.infer_shapes(m, data_prop=True)
    shapes = {v.name: [d.dim_value if d.HasField("dim_value") else None for d in v.type.tensor_type.shape.dim]
              for v in list(m.graph.value_info) + list(m.graph.input)}
    init = {i.name: list(i.dims) for i in m.graph.initializer}
    prod = {o: n for n in m.graph.node for o in n.output}
    weight = 0
    for n in m.graph.node:
        if n.op_type not in ("MatMul", "MatMulInteger", "Gemm") or n.input[1] not in init or len(init[n.input[1]]) != 2:
            continue
        a = n.input[0]
        if a not in shapes and a in prod and prod[a].op_type == "DynamicQuantizeLinear":
            a = prod[a].input[0]
        lead = shapes.get(a, [None])[:-1]
        if not lead or any(x is None for x in lead):
            raise SystemExit(f"MAC count: {n.name}'s activation rows are not resolved by shape inference")
        weight += int(np.prod(lead)) * init[n.input[1]][0] * init[n.input[1]][1]
    layers = sum(n.op_type == "Softmax" for n in m.graph.node) - 2   # two Softmaxes belong to the head
    return weight, batch * layers * 2 * seq * seq * 1024


def inputs(seq: int, batch: int) -> dict:
    if seq < 2 * HEAD:
        raise SystemExit(f"--seq {seq}: the option markers sit in the last {HEAD} tokens, so seq must be >= {2 * HEAD}")
    rng = np.random.default_rng(0)
    ids = rng.integers(5, VOCAB, size=(batch, seq), dtype=np.int64)
    pos = np.array([seq - HEAD + 8 + 16 * k for k in range(OPTIONS)], dtype=np.int64)
    return {"input_ids": ids, "attention_mask": np.ones((batch, seq), dtype=np.int64),
            "marker_pos": np.tile(pos, (batch, 1)), "marker_mask": np.ones((batch, OPTIONS), dtype=bool),
            "qtype": np.zeros(batch, dtype=np.int64)}


def session(model: Path, ep: str, threads: int, profile: str | None = None):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if profile:
        so.enable_profiling = True
        so.profile_file_prefix = profile
    if ep == "dml":
        so.enable_mem_pattern = False   # DirectML's documented requirement, with sequential execution
        providers = [("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    else:
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(model), sess_options=so, providers=providers)


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
        (n, ctypes.c_ulonglong) for n in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                                          "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]


def avail_gb() -> float:
    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2**30


class PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
        (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                       "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                       "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]


def peak_ws_gb() -> float:
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p   # a pseudo-handle: truncated to 32 bits, the call fails
    k32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
    c = PMC()
    c.cb = ctypes.sizeof(c)
    if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        return float("nan")
    return c.PeakWorkingSetSize / 2**30


def cmd_run(a) -> int:
    model = Path(a.model)
    weight, attn = macs_per_call(model, a.seq, a.batch)
    t0 = time.perf_counter()
    sess = session(model, a.ep, a.threads)
    feed = inputs(a.seq, a.batch)
    print(f"[summary] {a.ep} {model.name} seq {a.seq} batch {a.batch}: providers {sess.get_providers()}, "
          f"threads {a.threads if a.ep == 'cpu' else '-'}, session load {time.perf_counter() - t0:.1f} s", flush=True)
    for _ in range(a.warmup):
        sess.run(None, feed)
    times, start, n = [], time.perf_counter(), 0
    low_mem = avail_gb()
    while n < a.min_calls or time.perf_counter() - start < a.seconds:
        t = time.perf_counter()
        out = sess.run(None, feed)
        times.append((time.perf_counter() - t) * 1000.0)
        n += 1
        print(f"[run] frame {n}", flush=True)
        if n % 10 == 0:
            low_mem = min(low_mem, avail_gb())
    ms = sorted(times)
    mean = statistics.fmean(ms)
    print(f"[summary] latency ms over {n} calls after {a.warmup} warm-up: mean {mean:.2f}, median "
          f"{statistics.median(ms):.2f}, min {ms[0]:.2f}, p90 {ms[int(0.9 * (n - 1))]:.2f}, max {ms[-1]:.2f}; "
          f"{1000.0 * a.batch / mean:.2f} sequences/s, {1000.0 * a.batch * a.seq / mean:.0f} tokens/s", flush=True)
    print(f"[summary] MACs per call (DERIVED from the graph): weight MatMuls {weight / 1e9:.1f} G, attention "
          f"{attn / 1e9:.1f} G; {2 * (weight + attn) / (mean / 1000.0) / 1e12:.3f} TOPS effective at the mean", flush=True)
    print(f"[summary] memory: process peak working set {peak_ws_gb():.2f} GB, lowest available {low_mem:.1f} GB; "
          f"outputs {[o.shape for o in out]}, logits finite {bool(np.isfinite(out[0]).all())}", flush=True)
    return 0


def cmd_placement(a) -> int:
    sess = session(Path(a.model), a.ep, a.threads, profile=str(Path(a.profile_dir) / f"laya_{a.ep}"))
    feed = inputs(a.seq, a.batch)
    for _ in range(3):
        sess.run(None, feed)
    prof = Path(sess.end_profiling())
    events = json.loads(prof.read_text(encoding="utf-8"))
    nodes, dur = defaultdict(set), Counter()
    for e in events:
        if e.get("cat") == "Node" and "provider" in e.get("args", {}):
            p = e["args"]["provider"]
            nodes[p].add(e["name"].removesuffix("_kernel_time").removesuffix("_fence_before").removesuffix(
                "_fence_after"))
            if e["name"].endswith("_kernel_time"):
                dur[p] += e.get("dur", 0)
    total = sum(dur.values()) or 1
    for p in sorted(nodes):
        print(f"PLACEMENT {a.ep} seq {a.seq}: {p} ran {len(nodes[p])} nodes, {100.0 * dur[p] / total:.1f}% of kernel "
              f"time over 3 profiled calls")
    prof.unlink(missing_ok=True)
    return 0


def cmd_dmlcopy(a) -> int:
    """<model>_az0.onnx: the graph with allowzero cleared on every Reshape (it reads the same external weights). This
    ORT's DirectML rejects allowzero=1 (0x80070057 at the first such Reshape). Clearing it is equivalent only when no
    Reshape target holds a literal 0 at run time, so every run-time target is evaluated on the CPU at each --check-seq,
    and the two files' CPU outputs are compared exactly; any 0 or difference refuses."""
    import onnx
    import onnxruntime as ort
    src = Path(a.model)
    m = onnx.load(str(src), load_external_data=False)
    cleared = 0
    for n in m.graph.node:
        for at in n.attribute:
            if n.op_type == "Reshape" and at.name == "allowzero" and at.i == 1:
                at.i, cleared = 0, cleared + 1
    dst = src.with_name(src.stem + "_az0.onnx")
    onnx.save(m, str(dst))
    probe = onnx.load(str(src), load_external_data=False)
    init = {i.name for i in probe.graph.initializer}
    targets = sorted({n.input[1] for n in probe.graph.node if n.op_type == "Reshape"} - init)
    for t in targets:
        probe.graph.output.append(onnx.helper.make_tensor_value_info(t, onnx.TensorProto.INT64, None))
    tmp = src.with_name(src.stem + "_shapes.onnx")
    onnx.save(probe, str(tmp))
    zeros = []
    try:
        for seq in a.check_seq:
            s = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
            got = dict(zip([o.name for o in s.get_outputs()], s.run(None, inputs(seq, 1))))
            zeros += [(seq, t) for t in targets if (np.asarray(got[t]) == 0).any()]
    finally:
        tmp.unlink(missing_ok=True)
    feed = inputs(a.check_seq[0], 1)
    x = session(src, "cpu", a.threads).run(None, feed)
    y = session(dst, "cpu", a.threads).run(None, feed)
    same = all(np.array_equal(p, q) for p, q in zip(x, y))
    print(f"DMLCOPY {dst.name}: allowzero cleared on {cleared} Reshape nodes; {len(targets)} run-time targets, "
          f"{len(zeros)} holding a 0 at seq {a.check_seq}; CPU outputs identical to {src.name}: {same}; "
          f"sha256 {sha256(dst)}")
    if zeros or not same:
        dst.unlink(missing_ok=True)
        raise SystemExit("not equivalent: the copy is removed")
    return 0


def cmd_check(a) -> int:
    """One call of --model on --ep against --ref on the CPU in fp32, same input: the largest logit and act_probs
    differences. A numerics sanity line for the log, not an accuracy verdict (that needs labelled decisions)."""
    feed = inputs(a.seq, a.batch)
    ref = session(Path(a.ref), "cpu", a.threads).run(None, feed)
    got = session(Path(a.model), a.ep, a.threads).run(None, feed)
    print(f"CHECK {a.ep} {Path(a.model).name} vs cpu {Path(a.ref).name}, seq {a.seq} batch {a.batch}: logits max |diff| "
          f"{float(np.abs(got[0] - ref[0]).max()):.3e}, act_probs max |diff| {float(np.abs(got[1] - ref[1]).max()):.3e}, "
          f"same top option {bool((got[0].argmax(-1) == ref[0].argmax(-1)).all())} (synthetic input; not an accuracy "
          f"verdict)")
    return 0


def cmd_quantize(a) -> int:
    """The export's value_info records one head tensor as [256, ...] where shape inference derives [1028, ...], and the
    quantizer's shape inference refuses the conflict. value_info is optional metadata, so the quantizer reads a copy of
    the graph without it (beside the original, so the same external weights resolve)."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    src, out = Path(a.model), Path(a.out)
    for p in (out, out.with_name(out.name + ".data")):
        if p.exists():   # an existing external-data file is appended to, not replaced: the result would differ
            raise SystemExit(f"{p} exists; this tool never writes over a quantized model. Delete it by hand to re-run.")
    g = onnx.load(str(src), load_external_data=False)
    n_vi = len(g.graph.value_info)
    del g.graph.value_info[:]
    bare = src.with_name(src.stem + "_novi.onnx")
    onnx.save(g, str(bare))
    out.parent.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    try:
        quantize_dynamic(str(bare), str(out), per_channel=True, weight_type=QuantType.QInt8,
                         op_types_to_quantize=["MatMul"], use_external_data_format=True,
                         extra_options={"MatMulConstBOnly": True})
    finally:
        bare.unlink(missing_ok=True)
    print(f"QUANTIZED {out.name} in {time.perf_counter() - t:.0f} s from {src.name} without its {n_vi} value_info "
          f"entries: dynamic, per-channel QInt8 weights, MatMuls with constant weights only; accuracy unchecked")
    return 0


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_platform(a) -> int:
    import onnxruntime as ort
    print(f"PLATFORM onnxruntime {ort.__version__}, providers {ort.get_available_providers()}, python "
          f"{sys.version.split()[0]}, numpy {np.__version__}")
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
        print(f"PLATFORM CPU {winreg.QueryValueEx(k, 'ProcessorNameString')[0].strip()}, {os.cpu_count()} logical")
    gpu = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + ', driver ' + "
                          "$_.DriverVersion }"], capture_output=True, text=True)
    for line in gpu.stdout.splitlines():
        if line.strip():
            print(f"PLATFORM GPU {line.strip()}")
    scheme = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True).stdout.strip()
    print(f"PLATFORM power scheme (read-only) {scheme}")
    for m in a.model:
        for p in sorted(Path(m).parent.glob(Path(m).name + "*")):
            print(f"PLATFORM file {p.name} {p.stat().st_size} bytes sha256 {sha256(p)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=("platform", "quantize", "dmlcopy", "placement", "check", "run"))
    ap.add_argument("--check-seq", type=int, nargs="+", default=[512, 1024], help="dmlcopy: sequence lengths checked")
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--out")
    ap.add_argument("--ref", help="check: the fp32 model run on the CPU as the reference")
    ap.add_argument("--ep", choices=("cpu", "dml"), default="cpu")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8, help="CPU intra-op threads (the repo's CPU bar: 8 Zen4 cores)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--min-calls", type=int, default=20)
    ap.add_argument("--profile-dir", default=".")
    a = ap.parse_args()
    if a.mode != "platform":
        a.model = a.model[0]
    return {"platform": cmd_platform, "quantize": cmd_quantize, "dmlcopy": cmd_dmlcopy, "placement": cmd_placement,
            "check": cmd_check, "run": cmd_run}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())

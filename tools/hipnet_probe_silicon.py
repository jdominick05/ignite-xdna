"""hipnet NPU probe on silicon: layer exactness, then one interleaved latency sitting against ONNX Runtime's CPU.

    bash scripts/research-iron.sh tools/hipnet_probe_silicon.py --work <dir> --src <dir with the fp32 exports> \
        --calib <calib_inputs.npy> --log <file> --json <file> [--samples 16] [--iters 1000] [--passes 3]

Phase 1 (exactness), per model: the reuse-free container (<stem>_noreuse.ignite) is dispatched on each of
--samples calibration frames; every layer's tensor is read back and compared byte for byte with
graph_reference.run_direct, and the output map is also compared with ONNX Runtime CPU's output on the same QDQ
model. The default container (<stem>.ignite, workspace reuse on) gets the verifier's check: each readable layer
exact, each reused slot holding its planned final tenant, and the output exact.

Phase 2 (sitting), --passes times over all models, interleaved per model:
  npu dispatch   sess.dispatch() alone, default container
  npu call       float32 [1,32,5,20] in -> quantize -> stage_tensor -> dispatch -> read_tensor -> dequantize ->
                 float32 [1,32,5,20] out: what a caller pays per frame
  cpu fp32       ONNX Runtime CPU, the original FP32 export, 1 intra-op and 1 inter-op thread
  cpu qdq        ONNX Runtime CPU, the XINT8 QDQ model, 1 thread
each --warmup calls then --iters timed calls, inputs cycling through the calibration frames. ONNX Runtime
InferenceSession.run is counted during every NPU block; the containers declare no host segment, so it must be 0.
Refuses to start if xrt-smi shows a hardware context.
"""
import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

import ignite_xdna
from ignite_xdna.compiler import engine_schedule as es
from ignite_xdna.compiler import graph_ir, graph_reference as gr
from ignite_xdna.compiler.serializer import IgniteModelReader
from ignite_xdna.pipelines import npu_power
from ignite_xdna.runtime.graph_session import EngineSession

MODELS = ("hipnet_v0", "hipnet_v0_k5", "hipnet_v0_relu_after_add")
LOG = None


def log(msg=""):
    print(msg, flush=True)
    if LOG is not None:
        LOG.write(msg + "\n")
        LOG.flush()


ORT_RUNS = {"n": 0}
_orig_run = ort.InferenceSession.run


def _counting_run(self, *a, **k):
    ORT_RUNS["n"] += 1
    return _orig_run(self, *a, **k)


ort.InferenceSession.run = _counting_run


def stats(ms):
    a = np.asarray(ms, dtype=np.float64)
    return {"n": int(a.size), "median": float(np.median(a)), "p99": float(np.percentile(a, 99)),
            "mean": float(a.mean()), "min": float(a.min()), "max": float(a.max())}


def fmt(s):
    return f"median {s['median']:.4f} ms, p99 {s['p99']:.4f}, mean {s['mean']:.4f}, min {s['min']:.4f}, max {s['max']:.4f} (n={s['n']})"


def cpu_session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def lowered(model_path, reuse):
    ir = graph_ir.lower_yolov8n(str(model_path))
    ir.task = "segment"
    ws = es.plan_workspace(ir, reuse=reuse)
    return ir, ws


def check_plan(ge, ir, ws, path):
    if ws.nbytes != int(ge["workspace_bytes"]) or ir.input != ge["input_tensor"]:
        raise SystemExit(f"{path}: container plan differs from the lowering (workspace {ge['workspace_bytes']} "
                         f"vs {ws.nbytes})")


def manifest_summary(path):
    with IgniteModelReader(path) as r:
        m = dict(r.manifest)
    ge = m["graph_engine"]
    return {"task": m.get("task"), "layers": len(ge["layers"]),
            "host_segments": sum(1 for s in ge.get("segments", []) if s["kind"] == "host"),
            "segments": ge.get("segments", []), "rounds": ge["rounds"],
            "activation_packets": ge["activation_packets"], "weight_fills": ge["weight_fills"],
            "workspace_bytes": ge["workspace_bytes"], "insts_bytes": ge["insts_bytes"],
            "wpackets_bytes": ge["wpackets_bytes"], "workspace_reuse": ge.get("workspace_reuse", True),
            "egress_bytes": m.get("egress_bytes"),
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}, ge


def exactness(stem, work, calib, samples, qdq_path):
    rec = {}
    s_q = cpu_session(qdq_path)
    in_name = s_q.get_inputs()[0].name
    for kind, fname, reuse in (("noreuse", f"{stem}_noreuse.ignite", False), ("reuse", f"{stem}.ignite", True)):
        path = work / fname
        summ, ge = manifest_summary(path)
        ir, ws = lowered(qdq_path, reuse)
        check_plan(ge, ir, ws, path)
        final_of = {ws.placements[L.output].base: L.output for L in ir.layers}
        out_t = ir.outputs[0][1]
        t_in, t_out = ir.tensors[ir.input], ir.tensors[out_t]
        per_layer = {L.name: {"mismatches": 0, "readable": final_of[ws.placements[L.output].base] == L.output}
                     for L in ir.layers}
        out_bad_ref = out_bad_ort = 0
        ort_before = ORT_RUNS["n"]
        sess = EngineSession(path)
        try:
            ort_in_npu = 0
            for i in range(samples):
                x = calib[i:i + 1].astype(np.float32)
                q_in = gr.quantize_input(x[0], t_in.scale, t_in.zero_point)
                direct = gr.run_direct(ir, q_in)
                n0 = ORT_RUNS["n"]
                sess.stage_tensor(ir.input, q_in)
                sess.dispatch()
                got_all = {L.output: sess.read_tensor(L.output)[:ir.tensors[L.output].channels] for L in ir.layers}
                ort_in_npu += ORT_RUNS["n"] - n0
                for L in ir.layers:
                    owner = final_of[ws.placements[L.output].base]
                    c = min(ir.tensors[L.output].channels, ir.tensors[owner].channels)
                    per_layer[L.name]["mismatches"] += int(np.sum(got_all[L.output][:c] != direct[owner][:c]))
                y = got_all[out_t]
                out_bad_ref += int(np.sum(y != direct[out_t]))
                ref = s_q.run(None, {in_name: x})[0][0]
                y_f = (y.astype(np.float32) - t_out.zero_point) * np.float32(t_out.scale)
                out_bad_ort += int(np.sum(y_f != ref))
        finally:
            sess.close()
        exact = sum(1 for v in per_layer.values() if v["readable"] and v["mismatches"] == 0)
        readable = sum(1 for v in per_layer.values() if v["readable"])
        slot_ok = sum(1 for v in per_layer.values() if not v["readable"] and v["mismatches"] == 0)
        reused = len(per_layer) - readable
        ok = exact == readable and slot_ok == reused and out_bad_ref == 0 and out_bad_ort == 0 and ort_in_npu == 0
        rec[kind] = {"container": summ, "samples": samples, "layers_exact": exact, "layers_readable": readable,
                     "slots_reused": reused, "slots_reused_ok": slot_ok, "output_mismatches_vs_reference": out_bad_ref,
                     "output_mismatches_vs_ort_cpu": out_bad_ort, "ort_runs_during_npu": ort_in_npu,
                     "per_layer": per_layer, "ok": ok}
        log(f"  {fname}: {summ['layers']} layers, {summ['host_segments']} host segments, {summ['rounds']} rounds, "
            f"{summ['activation_packets']} activation packets, workspace {summ['workspace_bytes']:,} B")
        log(f"    {exact}/{readable} readable layers exact"
            + (f", {slot_ok}/{reused} reused slots hold their final tenant" if reused else "")
            + f" | output vs run_direct {out_bad_ref} mismatches, vs ORT CPU (QDQ) {out_bad_ort} mismatches"
            + f" | ORT runs during NPU {ort_in_npu} | over {samples} samples -> {'PASS' if ok else 'FAIL'}")
        for L in ir.layers:
            v = per_layer[L.name]
            if v["mismatches"]:
                log(f"      L{L.index} {L.name}: {v['mismatches']} mismatches ({'readable' if v['readable'] else 'slot'})")
    return rec


def npu_block(sess, ir, calib, warmup, iters, full):
    t_in = ir.tensors[ir.input]
    out_t = ir.outputs[0][1]
    t_out = ir.tensors[out_t]
    s_in, zp_in = np.float64(t_in.scale), t_in.zero_point
    s_out = np.float32(t_out.scale)
    ms = []
    n0 = ORT_RUNS["n"]
    cpu0 = None
    for i in range(warmup + iters):
        if i == warmup:
            cpu0 = time.process_time()
        x = calib[i % len(calib)][None]
        if full:
            t0 = time.perf_counter_ns()
            q = np.clip(np.round(x[0].astype(np.float64) / s_in) + zp_in, 0, 255).astype(np.uint8)
            sess.stage_tensor(ir.input, q)
            sess.dispatch()
            y = (sess.read_tensor(out_t).astype(np.float32) - np.float32(t_out.zero_point)) * s_out
            y = y[None]
            dt = (time.perf_counter_ns() - t0) / 1e6
        else:
            dt = sess.dispatch()
        if i >= warmup:
            ms.append(dt)
    cpu_ms = (time.process_time() - cpu0) * 1e3 / iters
    return ms, ORT_RUNS["n"] - n0, cpu_ms


def cpu_block(s, calib, warmup, iters):
    name = s.get_inputs()[0].name
    ms = []
    cpu0 = None
    for i in range(warmup + iters):
        if i == warmup:
            cpu0 = time.process_time()
        x = calib[i % len(calib)][None]
        t0 = time.perf_counter_ns()
        s.run(None, {name: x})
        dt = (time.perf_counter_ns() - t0) / 1e6
        if i >= warmup:
            ms.append(dt)
    return ms, (time.process_time() - cpu0) * 1e3 / iters


def main() -> int:
    global LOG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--commit", default="")
    args = ap.parse_args()
    LOG = open(args.log, "w", encoding="utf-8")
    work, src = Path(args.work), Path(args.src)
    calib = np.load(args.calib).astype(np.float32)
    models = args.models.split(",")

    rc, smi = npu_power.run_xrt_smi(["examine", "-r", "aie-partitions"])
    pmode = npu_power.read_mode()
    env = {"hostname": socket.gethostname(), "processor": platform.processor(), "python": sys.version.split()[0],
           "numpy": np.__version__, "onnxruntime": ort.__version__, "ignite_xdna": ignite_xdna.__file__,
           "ignite_xdna_commit": args.commit, "npu_power_mode": pmode, "cpu_count": os.cpu_count(),
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    log(f"[env] {json.dumps(env)}")
    log(f"[xrt-smi aie-partitions rc={rc}]")
    log(smi.strip())
    if "No hardware contexts running" not in smi:
        log("[abort] the NPU is not idle; a number measured now would be contention, not a finding")
        return 3
    result = {"env": env, "xrt_smi_before": smi, "exactness": {}, "sitting": [], "args": vars(args)}

    log("\n=== Phase 1: exactness ===")
    all_ok = True
    for stem in models:
        log(f"{stem}")
        rec = exactness(stem, work, calib, args.samples, work / f"{stem}_xint8.onnx")
        result["exactness"][stem] = rec
        all_ok &= all(r["ok"] for r in rec.values())

    log("\n=== Phase 2: sitting ===")
    irs = {stem: lowered(work / f"{stem}_xint8.onnx", True)[0] for stem in models}
    cpu_fp = {stem: cpu_session(src / f"{stem}.onnx") for stem in models}
    cpu_q = {stem: cpu_session(work / f"{stem}_xint8.onnx") for stem in models}
    for p in range(args.passes):
        for stem in models:
            row = {"pass": p, "model": stem}
            sess = EngineSession(work / f"{stem}.ignite")
            try:
                d_ms, d_ort, d_cpu = npu_block(sess, irs[stem], calib, args.warmup, args.iters, full=False)
                f_ms, f_ort, f_cpu = npu_block(sess, irs[stem], calib, args.warmup, args.iters, full=True)
            finally:
                sess.close()
            row["npu_dispatch"], row["npu_call"] = stats(d_ms), stats(f_ms)
            row["npu_dispatch"]["process_cpu_ms_per_call"] = d_cpu
            row["npu_call"]["process_cpu_ms_per_call"] = f_cpu
            row["ort_runs_during_npu"] = d_ort + f_ort
            c_ms, c_cpu = cpu_block(cpu_fp[stem], calib, args.warmup, args.iters)
            q_ms, q_cpu = cpu_block(cpu_q[stem], calib, args.warmup, args.iters)
            row["cpu_fp32"], row["cpu_qdq"] = stats(c_ms), stats(q_ms)
            row["cpu_fp32"]["process_cpu_ms_per_call"] = c_cpu
            row["cpu_qdq"]["process_cpu_ms_per_call"] = q_cpu
            result["sitting"].append(row)
            log(f"[pass {p}] {stem}")
            log(f"    npu dispatch  {fmt(row['npu_dispatch'])} | process CPU {d_cpu:.4f} ms/call")
            log(f"    npu call      {fmt(row['npu_call'])} | process CPU {f_cpu:.4f} ms/call")
            log(f"    cpu fp32 1t   {fmt(row['cpu_fp32'])} | process CPU {c_cpu:.4f} ms/call")
            log(f"    cpu qdq 1t    {fmt(row['cpu_qdq'])} | process CPU {q_cpu:.4f} ms/call")
            log(f"    ORT runs during NPU blocks: {row['ort_runs_during_npu']}")
            all_ok &= row["ort_runs_during_npu"] == 0

    rc2, smi2 = npu_power.run_xrt_smi(["examine", "-r", "aie-partitions"])
    result["xrt_smi_after"] = smi2
    result["npu_power_mode_after"] = npu_power.read_mode()
    result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    log(f"\n[xrt-smi aie-partitions after, rc={rc2}]")
    log(smi2.strip())
    log(f"[power mode after] {result['npu_power_mode_after']}")
    Path(args.json).write_text(json.dumps(result, indent=1), encoding="utf-8")
    log("PROBE PASS" if all_ok else "PROBE FAIL")
    LOG.close()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

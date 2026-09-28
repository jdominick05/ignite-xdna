"""Offline gate for the hipnet NPU probe: lowering, workspace plan, schedule and packet emulation.

    bash scripts/research-iron.sh tools/hipnet_probe_offline.py --work <dir> --calib <calib_inputs.npy> [--samples 16]

For each <model>_xint8.onnx in --work:
  * lower_yolov8n, plan_workspace (reuse off and on), schedule_graph;
  * replay every layer's DMA patterns and packets through the NumPy core emulator (emulate_layer) on a
    halo-filled workspace, per calibration sample, and compare every layer byte for byte with
    graph_reference.run_direct;
  * compare run_direct at EVERY layer with ONNX Runtime CPU's uint8 intermediates on the same QDQ model
    (optimizations off), so the reference itself is checked against the vendor-independent runtime.
Prints one line per model and writes offline_<model>.json. Exit 0 only if every comparison is exact.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

import ignite_xdna
from ignite_xdna.compiler import engine_schedule as es
from ignite_xdna.compiler import graph_ir, graph_reference as gr

MODELS = ("hipnet_v0", "hipnet_v0_k5", "hipnet_v0_relu_after_add")


def ort_intermediates(model_path: Path, names, x):
    m = onnx.load(str(model_path))
    have = {o.name for o in m.graph.output}
    for n in names:
        if n not in have:
            m.graph.output.append(onnx.helper.make_empty_tensor_value_info(n))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.intra_op_num_threads = 1
    s = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    in_name = s.get_inputs()[0].name
    outs = s.run(names, {in_name: x})
    return dict(zip(names, outs))


def check(model_path: Path, calib: np.ndarray, samples: int, reuse: bool):
    ir = graph_ir.lower_yolov8n(str(model_path))
    ir.task = "segment"
    ws = es.plan_workspace(ir, reuse=reuse)
    scheds, store = es.schedule_graph(ir, ws)
    t_in = ir.tensors[ir.input]
    layer_bad = {L.name: 0 for L in ir.layers}
    ort_bad = {L.name: 0 for L in ir.layers}
    names = [L.output for L in ir.layers]
    for i in range(samples):
        x = calib[i:i + 1].astype(np.float32)
        q_in = gr.quantize_input(x[0], t_in.scale, t_in.zero_point)
        direct = gr.run_direct(ir, q_in)
        ws_arr = ws.halo_fill()
        ws.write_tensor(ws_arr, ir.input, q_in)
        # Read each layer right after it runs: with workspace reuse a later tenant overwrites the slot, so an
        # after-the-frame read of an early tensor would compare its successor (the verifier's SLOT case).
        # A tenant that clobbered a still-live tensor shows up as a mismatch in the layer that reads it.
        for s in scheds:
            es.emulate_layer(s, store, ws_arr)
            L = ir.layers[s.layer_index]
            got = ws.read_tensor(ws_arr, L.output)[:ir.tensors[L.output].channels]
            layer_bad[L.name] += int(np.sum(got != direct[L.output]))
        if reuse:
            continue
        ref = ort_intermediates(model_path, names, x)
        for L in ir.layers:
            o = ref[L.output]
            o = o[0] if o.ndim == 4 else o
            ort_bad[L.name] += int(np.sum(o.astype(np.uint8) != direct[L.output]))
    return ir, ws, scheds, store, layer_bad, ort_bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--models", default=",".join(MODELS))
    args = ap.parse_args()
    print(f"ignite_xdna from {ignite_xdna.__file__}")
    calib = np.load(args.calib)
    work = Path(args.work)
    ok_all = True
    for stem in args.models.split(","):
        mp = work / f"{stem}_xint8.onnx"
        rec = {"model": mp.name}
        try:
            for reuse in (False, True):
                ir, ws, scheds, store, layer_bad, ort_bad = check(mp, calib, args.samples, reuse)
                key = "reuse" if reuse else "no_reuse"
                rec[key] = {"workspace_bytes": ws.nbytes, "layers": len(ir.layers),
                            "emulator_mismatches": layer_bad,
                            "rounds": sum(s.rounds for s in scheds),
                            "activation_packets": sum(s.packets for s in scheds)}
                if not reuse:
                    rec["ort_vs_reference_mismatches"] = ort_bad
                    rec["layers_desc"] = [
                        {"index": L.index, "name": L.name, "type": type(L).__name__,
                         "k": getattr(L, "k", None), "act": getattr(L, "act", None),
                         "residual": getattr(L, "residual", None) is not None,
                         "post_act": "relu" if getattr(L, "post_hswish", None) is not None
                         and getattr(L, "post_act_scale", None) is not None else None,
                         "out": [ir.tensors[L.output].channels, ir.tensors[L.output].height,
                                 ir.tensors[L.output].width]} for L in ir.layers]
                    rec["placements"] = {n: {"base": p.base, "halo": p.halo, "planes": p.planes,
                                             "replicas": p.replicas, "nbytes": p.nbytes}
                                         for n, p in ws.placements.items()}
                emu_ok = all(v == 0 for v in layer_bad.values())
                ort_ok = reuse or all(v == 0 for v in ort_bad.values())
                ok_all &= emu_ok and ort_ok
                print(f"{stem} [{key}]: {len(ir.layers)} layers, ws {ws.nbytes} B, emulator "
                      f"{'EXACT' if emu_ok else 'MISMATCH ' + str({k: v for k, v in layer_bad.items() if v})}"
                      + ("" if reuse else f"; run_direct vs ORT CPU "
                         f"{'EXACT' if ort_ok else 'MISMATCH ' + str({k: v for k, v in ort_bad.items() if v})}")
                      + f" over {args.samples} samples", flush=True)
        except Exception as ex:  # noqa: BLE001 - report the refusal verbatim and continue
            rec["error"] = f"{type(ex).__name__}: {ex}"
            ok_all = False
            print(f"{stem}: REFUSED {rec['error']}", flush=True)
        (work / f"offline_{stem}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print("OFFLINE PASS" if ok_all else "OFFLINE FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

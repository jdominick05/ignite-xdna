#!/usr/bin/env python
"""Does AMD's Vitis AI EP place any FLOATING-POINT model on the XDNA1 NPU?

The engine's case for a half-precision path rests on a claim about the vendor stack: that XDNA1's
AIE2 cores have bf16 hardware, and that AMD's Ryzen AI 1.7.1 toolchain does not expose it, taking
only INT8 QDQ. That claim is worth exactly as much as its evidence, so this measures it instead of
asserting it.

For each dtype it builds the same one-convolution graph, asks the Vitis AI EP to run it, and reads
the EP's OWN `vitisai_ep_report.json` for where the nodes went. The report is the evidence for "did
it reach the NPU"; a session that builds and returns an answer proves nothing, because the EP
silently leaves what it cannot take on the CPU.

Outcomes worth distinguishing, and all three are reported:
  * the session refuses to build at all (ORT or the EP rejects the dtype);
  * it builds and places 0 nodes on the NPU (runs, but entirely on the CPU - "does not run on the NPU");
  * it builds and places nodes on the NPU (the claim would be false).

Run under resnet_env17 with RYZEN_AI_INSTALLATION_PATH set, which
`scripts/research-lowlevel.sh` does:

    ./scripts/research-lowlevel.sh --log results/aie/amd_float_probe_<date>.log --checks-only --npu \
        -- python tools/amd_float_precision_probe.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

import numpy as np
import onnx
from onnx import TensorProto, helper as h, numpy_helper as nh

# ONNX elem_type -> (numpy dtype for initializers, label). bfloat16 has no numpy dtype, so its
# initializer is written from raw bits.
DTYPES = {
    "float32": TensorProto.FLOAT,
    "float16": TensorProto.FLOAT16,
    "bfloat16": TensorProto.BFLOAT16,
}


def build(elem_type: int, path: Path, channels: int = 32, size: int = 64) -> None:
    """One 3x3 convolution, the simplest thing the DPU is supposed to be good at."""
    rng = np.random.default_rng(0)
    w = rng.normal(scale=0.05, size=(channels, channels, 3, 3)).astype(np.float32)
    if elem_type == TensorProto.FLOAT:
        init = nh.from_array(w, "w")
    elif elem_type == TensorProto.FLOAT16:
        init = nh.from_array(w.astype(np.float16), "w")
    else:
        # bfloat16 is the top 16 bits of the float32 bit pattern, round-to-nearest-even omitted:
        # this graph is a placement probe, not an accuracy test.
        bits = (w.view(np.uint32) >> 16).astype(np.uint16)
        init = h.make_tensor("w", TensorProto.BFLOAT16, list(w.shape), bits.ravel().tolist(), raw=False)
    node = h.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    graph = h.make_graph(
        [node], "probe",
        [h.make_tensor_value_info("x", elem_type, [1, channels, size, size])],
        [h.make_tensor_value_info("y", elem_type, [1, channels, size, size])],
        [init])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, str(path))


def placement(report: Path):
    """(total, on NPU, on CPU) from the EP's own report, or None when it wrote none."""
    if not report.exists():
        return None
    data = json.loads(report.read_text())
    out = {}
    for entry in data if isinstance(data, list) else data.get("deviceStat", []):
        name = entry.get("name")
        if name:
            out[name] = int(entry.get("nodeNum", 0))
    return out or None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scratch", type=Path, default=ROOT / "scratch/float_probe")
    ap.add_argument("--dtypes", nargs="+", default=sorted(DTYPES))
    args = ap.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)

    from npu.session import build_session

    for label in args.dtypes:
        path = args.scratch / f"probe_{label}.onnx"
        build(DTYPES[label], path)
        key = f"float_probe_{label}"
        cache = ROOT / key
        if cache.exists():
            shutil.rmtree(cache, ignore_errors=True)
        record = {"dtype": label, "model": path.relative_to(ROOT).as_posix()}
        try:
            sess = build_session(path, "npu", cache_key=key)
            x = np.zeros([1, 32, 64, 64], np.float16 if label == "float16" else np.float32)
            if label == "bfloat16":
                record["ran"] = "not attempted: numpy has no bfloat16 to feed it"
            else:
                sess.run(None, {sess.get_inputs()[0].name: x})
                record["ran"] = True
            record["placement"] = placement(cache / "vitisai_ep_report.json")
        except Exception as ex:                                  # noqa: BLE001 - the refusal IS the result
            record["ran"] = False
            record["error_type"] = type(ex).__name__
            record["error"] = str(ex).strip().splitlines()[0][:400]
            record["placement"] = placement(cache / "vitisai_ep_report.json")
            record["traceback_tail"] = traceback.format_exc().strip().splitlines()[-1][:200]
        npu_nodes = (record.get("placement") or {}).get("NPU")
        record["reached_npu"] = bool(npu_nodes)
        print("FLOAT_PROBE", json.dumps(record, sort_keys=True), flush=True)

    print("NOTE", json.dumps({
        "reading": "reached_npu false means the Vitis AI EP put none of this graph on the NPU - it "
                   "either refused the model or ran it on the CPU. The EP writes its report on every "
                   "session build, so an absent report means the session never built."}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

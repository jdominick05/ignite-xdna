"""
YOLO-World v2 step 3c: requantize the four C2fAttn output convolutions of the XINT8 model with GPTQ-rounded int8
weights and an int32 bias, so the whole backbone runs as XINT8 on the graph engine.

Why:
  /model.{12,15,18,21}/cv2/conv/Conv output a small difference of large terms: the part reading Concat(a, b, c)
  and the part reading the text attention output cancel (RMS 22.4 and 22.3 against 1.34 for their sum at
  /model.12, docs/BENCHMARKS.md). Nearest rounding at one power-of-two weight scale, and XINT8's int8 bias at a
  scale of 1 or 2, are each large against that sum, and the model collapses. GPTQ rounding at the same scale
  compensates each rounding error in the later columns using the input statistics, and an int32 bias at the
  product scale s_x * s_w is exact enough; both are ordinary QDQ tensors that ONNX Runtime and the graph
  engine's 32-bit accumulator take as they are. Splitting each convolution in two halves does not help
  (3a_split_attn_conv.py, kept as the record of that negative result).

For each block in order (statistics come from the model with the earlier blocks already replaced):
  1. H = sum of x x^T over every pixel of --images calibration images (npu.yolow.letterbox), x the dequantized
     Concat the convolution reads;
  2. the FP32 weight W (from the FP32 cut model) gets its MSE-best power-of-two scale and GPTQ rounding at that
     scale (damping --damp times the mean diagonal of H, columns in order);
  3. the bias becomes int32 at scale s_x * s_w.

    conda activate mlir-aie-iron          (onnx, onnxruntime, opencv; Quark is not needed)
    python pipelines/yolow/3c_gptq_cv2.py

Writes models/yolov8s-worldv2_cut_xint8_gptqcv2.onnx. Compile it with the attention cores on the host:
    ignite-compile compile --model models/yolov8s-worldv2_cut_xint8_gptqcv2.onnx --output build/yolow.ignite
        --host-region /model.12/attn/ --host-region /model.15/attn/ --host-region /model.18/attn/
        --host-region /model.21/attn/          (Git Bash: MSYS_NO_PATHCONV=1)
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on sys.path

from npu.paths import DATA, MODELS
from npu.yolow import letterbox

BLOCKS = ("12", "15", "18", "21")


def best_pow2(w: np.ndarray) -> float:
    errs = [(float(((np.clip(np.round(w / 2.0 ** e), -128, 127) * 2.0 ** e - w) ** 2).sum()), 2.0 ** e)
            for e in range(-16, 6)]
    return min(errs)[1]


def gptq_int8(w: np.ndarray, H: np.ndarray, s: float, damp: float) -> np.ndarray:
    """int8 integers Q for W [rows][cols] at scale s minimising the output error on inputs with Gram matrix H."""
    W = w.copy()
    Hd = H + np.eye(H.shape[0]) * damp * np.mean(np.diag(H))
    Hinv = np.linalg.cholesky(np.linalg.inv(Hd)).T   # upper Cholesky factor of H^-1
    Q = np.zeros_like(W)
    for j in range(W.shape[1]):
        qj = np.clip(np.round(W[:, j] / s), -128, 127)
        e = (W[:, j] - qj * s) / Hinv[j, j]
        W[:, j + 1:] -= np.outer(e, Hinv[j, j + 1:])
        Q[:, j] = qj
    return Q


def relative_error_db(W, Q, H):
    D = W - Q
    return 10 * np.log10(np.trace(D @ H @ D.T) / np.trace(W @ H @ W.T))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", default=str(MODELS / "yolov8s-worldv2_cut.onnx"))
    ap.add_argument("--xint8", default=str(MODELS / "yolov8s-worldv2_cut_xint8.onnx"))
    ap.add_argument("--out", default=str(MODELS / "yolov8s-worldv2_cut_xint8_gptqcv2.onnx"))
    ap.add_argument("--calib-dir", default=str(DATA / "coco_calib"))
    ap.add_argument("--images", type=int, default=64)
    ap.add_argument("--damp", type=float, default=0.01)
    args = ap.parse_args()

    fp = onnx.load(args.fp32)
    fp_init = {t.name: numpy_helper.to_array(t).astype(np.float64) for t in fp.graph.initializer}
    fp_conv = {n.name: n for n in fp.graph.node if n.op_type == "Conv"}
    m = onnx.load(args.xint8)
    files = sorted(glob.glob(os.path.join(args.calib_dir, "*.jpg")))[:args.images]
    if not files:
        raise SystemExit(f"no jpgs in {args.calib_dir}")
    images = [letterbox(cv2.imread(f), 640)[0] for f in files]

    def scalar(name):
        t = next(t for t in m.graph.initializer if t.name == name)
        return float(np.asarray(numpy_helper.to_array(t)).flatten()[0])

    def add_init(name, arr):
        m.graph.initializer.append(numpy_helper.from_array(arr, name))
        return name

    for b in BLOCKS:
        name = f"/model.{b}/cv2/conv/Conv"
        conv = next(n for n in m.graph.node if n.name == name)
        dq = {n.output[0]: n for n in m.graph.node if n.op_type == "DequantizeLinear"}
        x_dq, w_dq, b_dq = dq[conv.input[0]], dq[conv.input[1]], dq[conv.input[2]]
        s_x = scalar(x_dq.input[1])
        probe = onnx.ModelProto()
        probe.CopyFrom(m)
        probe.graph.output.append(onnx.helper.make_tensor_value_info(conv.input[0], onnx.TensorProto.FLOAT, None))
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        so.log_severity_level = 3
        sess = ort.InferenceSession(probe.SerializeToString(), so, providers=["CPUExecutionProvider"])
        H = None
        for img in images:
            x = sess.run([conv.input[0]], {sess.get_inputs()[0].name: img})[0][0]
            X = x.reshape(x.shape[0], -1).T.astype(np.float64)
            H = X.T @ X if H is None else H + X.T @ X
        fc = fp_conv[name]
        W, B = fp_init[fc.input[1]][:, :, 0, 0], fp_init[fc.input[2]]
        s_w = best_pow2(W)
        Q = gptq_int8(W, H, s_w, args.damp)
        bias32 = np.round(B / (s_x * s_w)).astype(np.int64)
        if np.abs(bias32).max() >= 1 << 31:
            raise SystemExit(f"{name}: bias does not fit int32")
        nearest = np.clip(np.round(W / s_w), -128, 127)
        print(f"{name}: {W.shape[1]} -> {W.shape[0]} channels, s_x {s_x}, s_w {s_w}; in-sample output error "
              f"nearest {relative_error_db(W, nearest * s_w, H):.1f} dB, GPTQ {relative_error_db(W, Q * s_w, H):.1f} dB; "
              f"int32 bias {bias32.min()}..{bias32.max()}", flush=True)
        tag = f"/model.{b}/cv2/conv/gptq"
        for node, parts in ((w_dq, [add_init(f"{tag}.weight_quantized", Q.astype(np.int8)[:, :, None, None]),
                                    add_init(f"{tag}.weight_scale", np.array(s_w, dtype=np.float32)),
                                    add_init(f"{tag}.weight_zero_point", np.array(0, dtype=np.int8))]),
                            (b_dq, [add_init(f"{tag}.bias_quantized", bias32.astype(np.int32)),
                                    add_init(f"{tag}.bias_scale", np.array(s_x * s_w, dtype=np.float32)),
                                    add_init(f"{tag}.bias_zero_point", np.array(0, dtype=np.int32))])):
            if any(a.name == "axis" for a in node.attribute):
                raise SystemExit(f"{node.name}: per-axis DequantizeLinear")
            del node.input[:]
            node.input.extend(parts)
        used = {i for n in m.graph.node for i in n.input}
        keep = [t for t in m.graph.initializer if t.name in used]
        del m.graph.initializer[:]
        m.graph.initializer.extend(keep)

    onnx.save(m, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

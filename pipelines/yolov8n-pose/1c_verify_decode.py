"""
YOLO-pose step 1c: verify the numpy decode (npu.yolo_pose_decode) against the
ONNX graph's own decode tail, on a real image.

Compares output0 from the uncut float graph to decode_heads() run on the same
graph's HEAD_OUTS tensors (added as extra outputs via onnx.utils.extract_model,
not the cut model -- this checks the decode math in isolation from the cut).
Run again after any change to npu/yolo_pose_decode.py or npu/yolo_pose.py's
head geometry.

    conda activate resnet_env
    python pipelines/yolov8n-pose/1c_verify_decode.py

Writes results/pose_decode_verify.txt (see CONTRIBUTING.md: numeric claims in
README.md need a logged run under results/, not just an inline assertion).
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on sys.path

from npu.paths import MODELS, RESULTS
from npu.yolo_pose import HEAD_OUTS, INPUT_SIZE, letterbox
from npu.yolo_pose_decode import decode_heads

SRC = MODELS / "yolov8n-pose.onnx"
IMG = "assets/test_image.jpg"


def main():
    if not SRC.is_file():
        raise SystemExit(f"{SRC} missing - run 1_export.py first")
    if not os.path.exists(IMG):
        raise SystemExit(f"{IMG} missing")

    m = onnx.load(str(SRC))
    produced = {o for n in m.graph.node for o in n.output}
    for h in HEAD_OUTS:
        m.graph.output.append(onnx.ValueInfoProto(name=h))
    tmp = str(MODELS / "_pose_decode_verify_tmp.onnx")
    onnx.save(m, tmp)
    try:
        sess = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
        img = cv2.imread(IMG)
        x, pad, scale = letterbox(img, INPUT_SIZE)
        outs = sess.run(None, {sess.get_inputs()[0].name: x})
        names = [o.name for o in sess.get_outputs()]
        ref = outs[names.index("output0")]
        heads = [outs[names.index(h)] for h in HEAD_OUTS]
        mine = decode_heads(heads, imgsz=INPUT_SIZE)

        max_diff = float(np.abs(ref - mine).max())
        mean_diff = float(np.abs(ref - mine).mean())
        lines = [
            f"model: {SRC}",
            f"image: {IMG}",
            f"ref shape: {ref.shape}  decoded shape: {mine.shape}",
            f"max abs diff:  {max_diff:.6g}",
            f"mean abs diff: {mean_diff:.6g}",
        ]
        for l in lines:
            print(l)

        RESULTS.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS / "pose_decode_verify.txt"
        out_path.write_text("\n".join(lines) + "\n")
        print(f"\nwrote {out_path}")

        if max_diff > 1e-2:
            raise SystemExit(f"decode diverges from the graph's own output0 "
                             f"(max abs diff {max_diff:.4g}) - check "
                             "npu/yolo_pose_decode.py against the pose head's math")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    main()

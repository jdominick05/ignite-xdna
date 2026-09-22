"""The bar a bf16 NPU path actually has to clear: ONNX Runtime fp32 on this machine's CPU.

WHY THIS IS THE BAR AND NOT AMD. `tools/amd_placement_probe.py` measured the Vitis AI EP placing
ZERO nodes on the NPU at fp32 and at bfloat16, against 502 of 507 for the same model quantized. So
at float width there is no AMD arm to beat, and every bf16 claim is either "a first", or against our
own int8, or against this. This repo already carries two results where the NPU lost to the same Zen4
CPU - MobileNetV2 and resnetv2_50x3 - so the CPU is not a soft target.

WHAT IT TIMES. `session.run` only: the network, on an input already in memory, with no preprocessing
and no postprocessing on either side of it. That is the span comparable to a container's dispatch
plus transfer plus host layers, and it is deliberately the narrowest honest comparison - a whole-frame
number would fold in ingress costs the two stacks pay differently.

THE INPUT IS SYNTHETIC, seeded, and uniform in [0, 1). Content does not change the work an image
model does, and a fixed seed keeps two runs comparable; zeros are avoided because denormals can stall
a float kernel and would flatter the CPU. Any model whose input is not float is filled with zeros and
that is recorded.

    bash scripts/research-lowlevel.sh --log results/aie/<name>.log -- \\
        python tools/ort_cpu_baseline.py --model models/sesr_m7_fp32.onnx \\
            --model models/realesrgan_rrdb_r64_fp32.onnx --warmup 20 --frames 200
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import onnxruntime as ort


def emit(tag, **payload) -> None:
    print(tag, json.dumps(payload, sort_keys=True), flush=True)


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


ORT_TO_NP = {"tensor(float)": np.float32, "tensor(float16)": np.float16,
             "tensor(uint8)": np.uint8, "tensor(int8)": np.int8, "tensor(int64)": np.int64}


def make_input(meta, rng):
    """A tensor of the declared shape; a symbolic or missing dimension becomes 1."""
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in meta.shape]
    dtype = ORT_TO_NP.get(meta.type, np.float32)
    if dtype in (np.float32, np.float16):
        return rng.random(size=shape, dtype=np.float32).astype(dtype), shape, False
    return np.zeros(shape, dtype=dtype), shape, True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="repeatable; any ONNX")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--threads", type=int, default=0,
                    help="intra-op threads; 0 leaves ONNX Runtime's own default, which is recorded")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    emit("ORT_CPU_BASELINE_ENV", ort=ort.__version__, numpy=np.__version__,
         providers=ort.get_available_providers(), threads_requested=args.threads or "ort default",
         warmup=args.warmup, frames=args.frames, seed=args.seed)

    for spec in args.model:
        path = Path(spec) if Path(spec).is_absolute() else ROOT / spec
        if not path.exists():
            raise SystemExit(f"no such model: {path}")
        so = ort.SessionOptions()
        so.log_severity_level = 3
        if args.threads:
            so.intra_op_num_threads = args.threads
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        meta = sess.get_inputs()[0]
        x, shape, zero_filled = make_input(meta, np.random.default_rng(args.seed))
        name = meta.name
        outs = [o.name for o in sess.get_outputs()]

        for _ in range(args.warmup):
            sess.run(None, {name: x})
        samples = []
        for _ in range(args.frames):
            t0 = time.perf_counter()
            y = sess.run(None, {name: x})
            samples.append((time.perf_counter() - t0) * 1e3)

        emit("ORT_CPU_BASELINE",
             model=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
             model_sha256=sha(path), input_name=name, input_shape=shape, input_dtype=meta.type,
             input_zero_filled=zero_filled, outputs={o: list(np.asarray(v).shape)
                                                     for o, v in zip(outs, y)},
             mean_ms=float(np.mean(samples)), p50_ms=float(np.percentile(samples, 50)),
             p95_ms=float(np.percentile(samples, 95)), min_ms=float(np.min(samples)),
             frames=args.frames)
        del sess
    return 0


if __name__ == "__main__":
    sys.exit(main())

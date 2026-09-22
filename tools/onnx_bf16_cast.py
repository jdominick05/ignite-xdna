"""Round a float ONNX model's convolution arithmetic to bfloat16, so it can be scored on the CPU.

WHY. Before building a bf16 engine for a model it is worth knowing whether bf16 actually recovers
what int8 costs that model. That is a question about numerics, not about silicon, so it should be
answered on the CPU where it is cheap - and answered BEFORE the engine exists, because a "no" makes
the engine pointless.

WHAT IT MODELS. kernels/bf16_conv/conv_bf16.cc holds bf16 activations, bf16 weights and a bf16 bias,
multiplies them in the native mmul, accumulates in fp32, and stores the result back as bf16. This
reproduces exactly that, and nothing else:

  * weight and bias initializers are rounded to bf16 once, here, because the kernel reads them from
    memory already rounded;
  * each convolution's activation input and its output get a Cast(BFLOAT16) -> Cast(FLOAT) pair,
    which is the store-and-reload rounding the core performs;
  * the accumulation between them stays float32.

It does NOT model the halo, the packet geometry or the tile shape - those change no arithmetic.

WHAT THE ACCUMULATOR ACTUALLY DOES, and why this is still a fair model. This file used to justify
the float32 accumulation by saying the core's accumulator is fp32. It is not: the accumulate probe
measured `mmul<4,8,4>::mac` aligning its nine operands to the largest exponent among them and
rounding each separately to a 24-bit grid, ties to even, which is not IEEE addition
(results/aie/engine_bf16_mac_model_probe_npu_20260921.log). That made every number this tool
produces an upper bound on silicon rather than a prediction of it, until the gap was measured.
It was, offline, against the emulator's own `mac()`: over the tap counts real layers have - up to
144, k3 over 128 channels, and the post-ReLU case where errors accumulate instead of cancelling -
the aligned model tracks an exact sum to 2e-7 relative, and NO lane differs after the bf16 store
(`tools/bf16_accum_fidelity.py`, results/aie/bf16_accum_fidelity_desktop2_20260921.log). The
reason is margin: the grid keeps 24 bits below an instruction's largest operand and a bf16 store
keeps 8, so the difference sits 16 bits under what a stored activation can carry into the next
layer. The boundary is a cancellation ratio of 2**24 within ONE instruction - measured, not
assumed, and the same log brackets it. Below that the two are indistinguishable; above it the core
returns zero where an exact sum keeps the value.

So the remaining CAVEAT is narrower than the old one: this is a faithful model of PRECISION for any
layer that does not cancel by more than 2**24 inside a single multiply-accumulate, and a bit-exact
model of the kernel for nothing. A model with that much cancellation - this repo has met the
milder form in YOLO-World's four C2fAttn cv2 convs - must be re-checked with `--cancel` before its
CPU score is quoted as a silicon prediction.

Usage:

    python tools/onnx_bf16_cast.py --in models/foo_fp32.onnx --out scratch/foo_bf16.onnx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def round_bf16(a: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even from float32 to bfloat16, kept in a float32 container.

    Done on the bits rather than via ml_dtypes so the tool has no dependency beyond onnx/numpy, and
    so the tie-breaking is visible: add half an ULP plus the low bit of the surviving mantissa.
    """
    f = np.ascontiguousarray(a, dtype=np.float32)
    u = f.view(np.uint32)
    # NaN must stay NaN: adding the rounding bias to a NaN payload can clear it to infinity.
    nan = np.isnan(f)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).view(np.float32)
    return np.where(nan, f, r).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", type=Path, required=True)
    ap.add_argument("--out", dest="dst", type=Path, required=True)
    ap.add_argument("--ops", default="Conv",
                    help="comma-separated op types to round; anything the engine would run in bf16")
    args = ap.parse_args()

    ops = {s.strip() for s in args.ops.split(",") if s.strip()}
    model = onnx.load(str(args.src))
    graph = model.graph

    inits = {i.name: i for i in graph.initializer}
    targets = [n for n in graph.node if n.op_type in ops]
    if not targets:
        raise SystemExit(f"no {sorted(ops)} nodes in {args.src}")

    # 1. Weights and biases, rounded in place. A shared initializer is rounded once; rounding is
    #    idempotent, so a second visit would be harmless anyway.
    rounded = []
    for n in targets:
        for name in n.input[1:]:
            if name in inits and name not in rounded:
                t = inits[name]
                if t.data_type != TensorProto.FLOAT:
                    continue
                a = numpy_helper.to_array(t)
                t.CopyFrom(numpy_helper.from_array(round_bf16(a), name))
                rounded.append(name)

    # 2. Activation inputs and outputs, through a real Cast pair. Emitted in place so the node list
    #    stays topologically sorted - appending at the end would not load.
    nodes, chains, n_in, n_out = [], {}, 0, 0
    tag = 0

    def cast_pair(source: str, sink: str) -> None:
        """source -> Cast(BFLOAT16) -> Cast(FLOAT) -> sink, appended to `nodes`."""
        nonlocal tag
        mid = f"{source}__bf16_{tag}"
        tag += 1
        nodes.append(helper.make_node("Cast", [source], [mid], to=TensorProto.BFLOAT16,
                                      name=f"bf16_down_{tag}"))
        nodes.append(helper.make_node("Cast", [mid], [sink], to=TensorProto.FLOAT,
                                      name=f"bf16_up_{tag}"))

    for n in graph.node:
        if n.op_type not in ops:
            nodes.append(n)
            continue

        # Input side: one chain per distinct tensor, reused by every later consumer of it.
        act = n.input[0]
        if act not in inits:
            if act not in chains:
                chains[act] = f"{act}__bf16_in"
                cast_pair(act, chains[act])
                n_in += 1
            n.input[0] = chains[act]

        # A non-initializer weight is an activation too, and the engine would round it the same way.
        for k in range(1, len(n.input)):
            name = n.input[k]
            if name and name not in inits:
                if name not in chains:
                    chains[name] = f"{name}__bf16_in"
                    cast_pair(name, chains[name])
                    n_in += 1
                n.input[k] = chains[name]

        # Output side: the node produces a private tensor, the Cast pair republishes the original
        # name, so every consumer - including a graph output - sees the rounded value.
        out = n.output[0]
        pre = f"{out}__pre_bf16"
        n.output[0] = pre
        nodes.append(n)
        cast_pair(pre, out)
        n_out += 1

    del graph.node[:]
    graph.node.extend(nodes)

    # Cast to BFLOAT16 needs opset 13 or later; below that the cast is not defined for the type.
    for o in model.opset_import:
        if o.domain in ("", "ai.onnx") and o.version < 13:
            raise SystemExit(f"opset {o.version} predates bfloat16 Cast; re-export at 13 or later")

    onnx.checker.check_model(model)
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.dst))

    print("BF16_CAST " + json.dumps({
        "source": str(args.src), "output": str(args.dst), "ops": sorted(ops),
        "nodes_wrapped": len(targets), "initializers_rounded": len(rounded),
        "input_cast_chains": n_in, "output_cast_chains": n_out,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

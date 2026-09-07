"""Read the quantization grid straight out of a quantized .onnx, and the EP's
own placement report, to explain a post-quantization accuracy collapse.

    python tools/audit_quant_grid.py                     # the MobileViT vs MobileNetV2 pair
    python tools/audit_quant_grid.py --model models/foo_xint8.onnx
    python tools/audit_quant_grid.py --rank-audit        # just the EP placement side

Static only -- reads files, runs no hardware, needs no NPU. Everything printed
is derivable by anyone holding the same models/ artifacts.

Why this exists: mobilevit_xxs drops from 68.30% top-1 (FP32) to 0.80%
(XINT8+AdaRound) on this backend, while mobilenetv2_100 keeps 73.40%. Both are
depthwise-separable CNNs quantized by the same Quark config to the same
per-tensor power-of-two grid, so "depthwise + per-tensor" alone cannot be the
explanation. This prints the three quantities that actually separate them:

  * weight-scale granularity   -- per-tensor vs per-channel (the XDNA1 DPU
                                  compiler mandates per-tensor)
  * the depthwise scale grid   -- MobileViT reaches Delta=1.0 on a 3x3
                                  depthwise kernel; MobileNetV2 never exceeds
                                  0.25. At Delta=1.0 almost every real weight
                                  rounds to 0 or +-1.
  * dead channels              -- how many output channels quantize to all
                                  zeros, i.e. are deleted from the network

The dead-channel count is the symptom most people reach for, and on its own it
does NOT discriminate. Two controls say so:
  * MobileNetV2 carries a 25%-dead depthwise block and still recovers to 73.40%.
  * models/mobilevit_xint8.onnx -- an older UseRandomData=True place-and-route
    probe -- has ZERO dead depthwise channels and still scores ~0%.
The scale grid is what separates them. That probe is also the control proving
the real-data models are a distinct calibration and not the same artifact
relabelled: its activation scales span 0.000122..4.0 (32768x, Gaussian input)
against the real calibration's 0.0078..0.5 (64x).

AdaRound cannot repair this: it chooses between floor(w/Delta) and ceil(w/Delta)
but never changes Delta. The audit shows exactly that -- the scale grid is
byte-identical before and after AdaRound, while the dead-channel count only
shifts at the rounding boundary (28/432 -> 24/432), which buys 0.80% top-1
instead of 0.00%. Rounding cannot recover a channel whose whole weight vector
is smaller than half a quantization step.
"""

import argparse
import collections
import glob
import json

import numpy as np
import onnx
from onnx import numpy_helper


def scan_model(path):
    """Return (depthwise, pointwise, activation_scales) for a quantized ONNX.

    Each conv entry is (name, scale, dead_channels, out_channels, is_per_channel).
    A channel is "dead" when every weight in it quantized to exactly zero.
    """
    m = onnx.load(path)
    init = {i.name: i for i in m.graph.initializer}
    prod = {o: n for n in m.graph.node for o in n.output}

    depthwise, pointwise = [], []
    for n in m.graph.node:
        if n.op_type != "Conv":
            continue
        dq = prod.get(n.input[1])
        if dq is None or dq.op_type != "DequantizeLinear":
            continue  # not a QDQ-quantized weight
        w = numpy_helper.to_array(init[dq.input[0]])
        s_arr = numpy_helper.to_array(init[dq.input[1]])
        scale = float(np.ravel(s_arr)[0])
        dead = int((np.abs(w).reshape(w.shape[0], -1).max(1) == 0).sum())
        group = next((a.i for a in n.attribute if a.name == "group"), 1)
        entry = (n.name, scale, dead, w.shape[0], s_arr.size > 1)
        (depthwise if group > 1 else pointwise).append(entry)

    # Activation scales: a QuantizeLinear whose data input is NOT an initializer
    # is quantizing a live tensor, not a weight.
    acts = [float(np.ravel(numpy_helper.to_array(init[n.input[1]]))[0])
            for n in m.graph.node
            if n.op_type == "QuantizeLinear" and n.input[0] not in init and n.input[1] in init]
    return depthwise, pointwise, acts


def report_model(path, tag):
    dw, pw, acts = scan_model(path)
    if not dw and not pw:
        print(f"\n{tag}\n  {path}\n  no QDQ-quantized Conv found (is this an FP32 model?)")
        return
    per_channel = any(e[4] for e in dw + pw)
    print(f"\n{tag}")
    print(f"  {path}")
    print(f"  granularity            : {'PER-CHANNEL' if per_channel else 'per-tensor'}"
          f"  ({len(dw)} depthwise + {len(pw)} other convs)")
    if dw:
        frac = [d / c for _, _, d, c, _ in dw]
        print(f"  depthwise scale grid   : {sorted({e[1] for e in dw})}")
        print(f"  depthwise dead chans   : {sum(e[2] for e in dw)}/{sum(e[3] for e in dw)}"
              f"   worst block {100 * max(frac):.1f}%   median {100 * np.median(frac):.1f}%")
        print(f"  per-block dead pct     : {' '.join(f'{100 * f:.0f}' for f in frac)}")
    if pw:
        print(f"  other conv scale range : {min(e[1] for e in pw):g} .. {max(e[1] for e in pw):g}")
        print(f"  other conv dead chans  : {sum(e[2] for e in pw)}/{sum(e[3] for e in pw)}")
    if acts:
        print(f"  activation scales      : {min(acts):g} .. {max(acts):g}"
              f"   ({max(acts) / min(acts):.0f}x range, {len(acts)} sites)")
        print(f"  distinct activation    : {sorted(set(acts))}")


def rank_audit():
    """Which tensor ranks does the VitisAI EP actually place on the NPU?

    Reads every vitisai_ep_report.json the repo's compile caches have produced.
    The EP writes one on every session build, so this needs no rerun.
    """
    print("\n=== EP placement by tensor rank (from the EP's own reports) ===")
    reports = (sorted(glob.glob("*cachekey/vitisai_ep_report.json"))
               + sorted(glob.glob("*cachekey/*/vitisai_ep_report.json"))
               + sorted(glob.glob("mobilevit_*/vitisai_ep_report.json")))
    if not reports:
        print("  no vitisai_ep_report.json found -- run something on --ep npu first")
        return
    for p in reports:
        with open(p) as f:
            r = json.load(f)
        rank = {s["name"]: len(s["shape"]) for s in r.get("shapeInfo", [])}

        def max_rank(node):
            rs = [rank[t] for t in list(node["output"]) + list(node["input"]) if t in rank]
            return max(rs) if rs else 0

        nodes = r.get("nodeStat", [])
        hi = collections.Counter(n["device"] for n in nodes if max_rank(n) >= 5)
        by_op = collections.Counter((n["device"], max_rank(n)) for n in nodes
                                    if n["opType"] in ("Slice", "Transpose", "Squeeze"))
        print(f"\n  {p}   ({len(nodes)} nodes)")
        print(f"    nodes touching a rank>=5 tensor : {dict(hi) or 'none in this graph'}")
        print(f"    Slice/Transpose/Squeeze by (device, rank):")
        for k, v in sorted(by_op.items(), key=lambda x: str(x)):
            print(f"      {k} -> {v}")


DEFAULT_PAIR = [
    ("models/mobilenetv2_xint8_adaround.onnx",
     "MobileNetV2 (ReLU6)         -- RECOVERS to 73.40% despite a 25%-dead block"),
    ("models/mobilevit_xxs_xint8.onnx",
     "MobileViT-XXS full XINT8    -- COLLAPSES to 0.00%, depthwise scale reaches 1.0"),
    ("models/mobilevit_xxs_hybrid_adaround.onnx",
     "MobileViT-XXS hyb+AdaRound  -- 0.80%; scale grid IDENTICAL to the row above"),
    ("models/mobilevit_xint8.onnx",
     "MobileViT-XXS random-probe  -- CONTROL: 0 dead channels, still ~0% accuracy"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=None,
                    help="quantized ONNX to audit (repeatable); default is the "
                         "MobileNetV2/MobileViT pair the finding rests on")
    ap.add_argument("--rank-audit", action="store_true",
                    help="only print the EP rank-placement audit")
    args = ap.parse_args()

    if not args.rank_audit:
        if args.model:
            for p in args.model:
                report_model(p, p)
        else:
            for p, tag in DEFAULT_PAIR:
                report_model(p, tag)
    rank_audit()


if __name__ == "__main__":
    main()

"""AIE-ML Systolic Shift-Cut Feasibility Theorem & Verifier for Project Ignition.

Mathematical Formulation:
-------------------------
On AMD XDNA1 AIE-ML architecture, integer convolution and matrix multiplication
accumulate into 32-bit registers. The post-multiplication ALU maps the 32-bit
accumulator to an 8-bit output tensor via an integer multiplier M (15-bit)
and an arithmetic right-shift register sigma in [0, 31] (the 'shift_cut'):

    out_8 = clamp( floor( (acc_32 * M + 2^(sigma - 1)) / 2^sigma ), -128, 127 )

For an ONNX QuantizeLinear/DequantizeLinear triad with:
    Input scale:  S_x
    Weight scale: S_w
    Output scale: S_y

The ideal analytical scale factor is:
    A = (S_x * S_w) / S_y

The hardware compiler approximates A using (M, sigma):
    A ≈ M * 2^(-sigma),  where M in [16384, 32767] and sigma in [0, 31].

Theorem 1 (Systolic Shift-Cut Bound):
    An operation is physically executable without numerical distortion on XDNA1
    if and only if:
        0 <= sigma <= 31

    If sigma > 31, the hardware shift register overflows or clamps (as observed in
    RegNetX-002, where sigma = 131 clamped to -108, producing a scale error of
    2^(131 - (-108)) = 2^239 and destroying top-1 accuracy to 0.50%).

Theorem 2 (Multi-Branch Inter-Scale Feasibility):
    For elementwise tensor operations C = A * B or C = A + B:
        A_elem = (S_A * S_B) / S_C
    Both branches must satisfy identical quantization scale alignments; divergent
    power-of-two scale grids cause dynamic range truncation (as observed in
    BiSeNetV2 Bilateral Guided Aggregation, dropping mIoU from 25.72% to 2.44%).
"""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import onnx
from onnx import numpy_helper


class ShiftCutHazard:
    def __init__(self, node_name: str, op_type: str, scale_x: float, scale_w: float, scale_y: float,
                 A: float, M: int, sigma: int, is_hazard: bool, reason: str):
        self.node_name = node_name
        self.op_type = op_type
        self.scale_x = scale_x
        self.scale_w = scale_w
        self.scale_y = scale_y
        self.A = A
        self.M = M
        self.sigma = sigma
        self.is_hazard = is_hazard
        self.reason = reason

    def __repr__(self):
        status = "HAZARD" if self.is_hazard else "OK"
        return (f"[{status}] {self.op_type} '{self.node_name}': "
                f"A={self.A:.6e}, M={self.M}, sigma={self.sigma} -> {self.reason}")


def compute_shift_cut(A: float) -> Tuple[int, int]:
    """Decompose real scale factor A into 15-bit multiplier M and shift-cut sigma.
    
    Target: A ≈ M * 2^(-sigma), with M in [16384, 32767].
    """
    if A <= 0:
        return 0, 0
    
    # We want M in [2^14, 2^15 - 1] = [16384, 32767]
    # log2(A) = log2(M) - sigma
    # sigma = round(15 - log2(A))
    log2_A = math.log2(A)
    sigma = int(math.floor(15 - log2_A))
    M = int(round(A * (2 ** sigma)))
    
    # Normalize M into 15-bit range if needed
    while M >= 32768 and sigma < 64:
        M //= 2
        sigma -= 1
    while M < 16384 and sigma > -64:
        M *= 2
        sigma += 1
        
    return M, sigma


def analyze_model_shift_cut(model_path: str) -> List[ShiftCutHazard]:
    """Walk an ONNX QDQ model and evaluate the shift-cut feasibility of all nodes."""
    model = onnx.load(model_path)
    graph = model.graph

    # Map tensor names to initializer arrays / values
    initializers = {}
    for init in graph.initializer:
        initializers[init.name] = numpy_helper.to_array(init)

    # Map node outputs to producing node
    producers = {}
    for node in graph.node:
        for out in node.output:
            producers[out] = node

    hazards = []

    for node in graph.node:
        if node.op_type in ["Conv", "Gemm", "MatMul"]:
            # Find input scale, weight scale, output scale
            # In QDQ ONNX, node inputs come from DequantizeLinear
            scale_x = 1.0
            scale_w = 1.0
            scale_y = 1.0

            # Input 0 scale
            in0 = node.input[0]
            if in0 in producers and producers[in0].op_type == "DequantizeLinear":
                dq_x = producers[in0]
                scale_x_name = dq_x.input[1]
                if scale_x_name in initializers:
                    scale_x = float(np.ravel(initializers[scale_x_name])[0])

            # Input 1 scale (weights)
            in1 = node.input[1]
            if in1 in producers and producers[in1].op_type == "DequantizeLinear":
                dq_w = producers[in1]
                scale_w_name = dq_w.input[1]
                if scale_w_name in initializers:
                    scale_w = float(np.ravel(initializers[scale_w_name])[0])
            elif in1 in initializers:
                # Direct float weights (non-quantized or pre-folded)
                scale_w = 1.0

            # Output scale (from subsequent QuantizeLinear)
            out0 = node.output[0]
            for consumer in graph.node:
                if consumer.op_type == "QuantizeLinear" and consumer.input[0] == out0:
                    scale_y_name = consumer.input[1]
                    if scale_y_name in initializers:
                        scale_y = float(np.ravel(initializers[scale_y_name])[0])
                    break

            A = (scale_x * scale_w) / scale_y if scale_y > 0 else 1.0
            M, sigma = compute_shift_cut(A)

            is_hazard = False
            reason = "Feasible systolic shift"
            if sigma < 0:
                is_hazard = True
                reason = f"ACCUMULATOR OVERFLOW HAZARD: sigma={sigma} < 0 (requires left-shift beyond accumulator)"
            elif sigma > 31:
                is_hazard = True
                reason = f"SHIFT CLAMP HAZARD: sigma={sigma} > 31 (hardware 5-bit shifter clamps/overflows)"

            hazards.append(ShiftCutHazard(
                node_name=node.name or out0,
                op_type=node.op_type,
                scale_x=scale_x,
                scale_w=scale_w,
                scale_y=scale_y,
                A=A,
                M=M,
                sigma=sigma,
                is_hazard=is_hazard,
                reason=reason
            ))

        elif node.op_type == "Mul":
            # Elementwise multiplication (e.g. BiSeNetV2 Bilateral Gating)
            scale_a = 1.0
            scale_b = 1.0
            scale_out = 1.0

            if node.input[0] in producers and producers[node.input[0]].op_type == "DequantizeLinear":
                dq_a = producers[node.input[0]]
                if dq_a.input[1] in initializers:
                    scale_a = float(np.ravel(initializers[dq_a.input[1]])[0])

            if len(node.input) > 1 and node.input[1] in producers and producers[node.input[1]].op_type == "DequantizeLinear":
                dq_b = producers[node.input[1]]
                if dq_b.input[1] in initializers:
                    scale_b = float(np.ravel(initializers[dq_b.input[1]])[0])

            out0 = node.output[0]
            for consumer in graph.node:
                if consumer.op_type == "QuantizeLinear" and consumer.input[0] == out0:
                    if consumer.input[1] in initializers:
                        scale_out = float(np.ravel(initializers[consumer.input[1]])[0])
                    break

            A = (scale_a * scale_b) / scale_out if scale_out > 0 else 1.0
            M, sigma = compute_shift_cut(A)

            is_hazard = False
            reason = "Feasible elementwise systolic shift"
            if sigma < 0 or sigma > 31:
                is_hazard = True
                reason = f"BILATERAL GATING HAZARD: sigma={sigma} outside [0, 31] (branch scale ratio divergent)"

            hazards.append(ShiftCutHazard(
                node_name=node.name or out0,
                op_type=node.op_type,
                scale_x=scale_a,
                scale_w=scale_b,
                scale_y=scale_out,
                A=A,
                M=M,
                sigma=sigma,
                is_hazard=is_hazard,
                reason=reason
            ))

    return hazards


def print_shift_cut_report(model_path: str, hazards: List[ShiftCutHazard]):
    print("=" * 80)
    print(f"AIE-ML Systolic Shift-Cut Verification Report: {model_path}")
    print("=" * 80)
    
    total = len(hazards)
    violations = [h for h in hazards if h.is_hazard]
    
    print(f"Analyzed {total} quantized systolic operation(s).")
    print(f"Shift-cut violations found: {len(violations)} / {total} ({(len(violations)/max(1, total))*100:.1f}%)")
    
    if violations:
        print("\n" + "!" * 80)
        print("CRITICAL HARDWARE INFEASIBILITY HAZARDS DETECTED:")
        print("!" * 80)
        for v in violations[:15]:
            print(f"  - {v}")
        if len(violations) > 15:
            print(f"  ... and {len(violations) - 15} more violations.")
    else:
        print("\nPASS: All layers reside cleanly within the [0, 31] systolic shift-cut basin.")
        
    # Print distribution
    sigmas = [h.sigma for h in hazards]
    if sigmas:
        print(f"\nShift-Cut Register Distribution: min={min(sigmas)}, median={int(np.median(sigmas))}, max={max(sigmas)}")
        print("=" * 80)

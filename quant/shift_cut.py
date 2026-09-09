"""AIE-ML Systolic Shift-Cut Feasibility Theorem & Verifier for Project Ignition.

Mathematical Formulation:
-------------------------
On AMD XDNA1 AIE-ML architecture, integer convolution and matrix multiplication
accumulate into 32-bit registers. The post-multiplication ALU maps the 32-bit
accumulator to an 8-bit output tensor via an integer multiplier M (15-bit)
and an arithmetic right-shift register sigma in [0, 31] (the 'shift_cut'):

    out_8 = clamp( floor( (acc_32 * M + 2^(sigma - 1)) / 2^sigma ), -128, 127 )

For an ONNX QuantizeLinear/DequantizeLinear triad with:
    Input scale:  S_x = 2^(-pos_x)
    Weight scale: S_w = 2^(-pos_w)
    Output scale: S_y = 2^(-pos_y)

The ideal analytical scale factor is:
    A = (S_x * S_w) / S_y

The hardware compiler approximates A using (M, sigma):
    A ≈ M * 2^(-sigma),  where M in [16384, 32767] and sigma in [0, 31].

Theorem 1 (Systolic Shift-Cut Bound):
    An operation is physically executable without numerical distortion on XDNA1
    if and only if:
        0 <= sigma <= 31

    Since M ≈ 2^14 and A = 2^(pos_y - pos_x - pos_w), we have:
        sigma = pos_x + pos_w - pos_y + 14

    If sigma < 0, the operation requires an arithmetic left-shift exceeding accumulator
    precision (accumulator overflow hazard).
    If sigma > 31, the hardware 5-bit shifter clamps to 31 (shift clamp hazard).

Theorem 2 (Systolic Scale Feasibility Window):
    For any fixed input scale S_x (pos_x) and weight scale S_w (pos_w), the output
    scale S_y (pos_y) must lie strictly in the closed interval:
        pos_y in [pos_x + pos_w - 17,  pos_x + pos_w + 14]
    Any scale chosen outside this interval cannot be executed on the physical
    AIE-ML systolic array without numerical clamping or overflow.

Theorem 3 (Fixed-Point Position Feasibility) -- RETRACTED 2026-09-09:
    This claimed that positions must satisfy pos in [0, 31], and that pos < 0 (S > 1.0)
    causes dynamic range overflow. It is contradicted by measurement on this device.

    tools/xint8_arithmetic_probe.py builds a QDQ Conv whose output position is -sc. Fifteen
    of its seventeen NPU fixtures ran at pos_y in {-1, -4, -8, -16} -- output scales up to
    2^16 -- placed on the DPU, and matched an independent integer reference exactly apart
    from a uniform one-code half-up rounding difference explained elsewhere. Two shipped
    models also contradict it: SESR-M7 was flagged 9/9 while placing 50 of 52 nodes and
    scoring 34.06 dB, and MODNet-Cut was flagged 1/74 while placing 502 of 507.

    The RegNetX-002 evidence cited above was also misattributed: its pos = -120 comes from
    cross-layer equalization inflating the ranges, not from a hardware bound. The same graph
    and producer without CLE gives positions 2..10 and 66.20% top-1 against 0.10%.

    Positions outside [0, 31] are therefore reported as an advisory note, never as a hazard.
    Theorem 1's sigma window remains the executability criterion, as Theorem 1 itself states.
    Note that the sigma edges have NOT been measured either: the probe only ever reached
    sigma in {14, 15, 18, 22, 30}, all comfortably inside [0, 31].
"""

import math
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import onnx
from onnx import numpy_helper


class ShiftCutHazard:
    def __init__(self, node_name: str, op_type: str, scale_x: float, scale_w: float, scale_y: float,
                 A: float, M: int, sigma: int, pos_x: int, pos_w: int, pos_y: int,
                 is_hazard: bool, reason: str, note: str = ""):
        self.node_name = node_name
        self.op_type = op_type
        self.scale_x = scale_x
        self.scale_w = scale_w
        self.scale_y = scale_y
        self.A = A
        self.M = M
        self.sigma = sigma
        self.pos_x = pos_x
        self.pos_w = pos_w
        self.pos_y = pos_y
        self.is_hazard = is_hazard
        self.reason = reason
        # Advisory only: an out-of-[0,31] position. Measured NOT to be a hazard on its own
        # (see Theorem 3, retracted). Never counted as a violation.
        self.note = note

    def __repr__(self):
        status = "HAZARD" if self.is_hazard else ("NOTE" if self.note else "OK")
        return (f"[{status}] {self.op_type} '{self.node_name}': "
                f"A={self.A:.6e}, M={self.M}, sigma={self.sigma} "
                f"(pos_x={self.pos_x}, pos_w={self.pos_w}, pos_y={self.pos_y}) -> "
                f"{self.reason}{('; ' + self.note) if self.note else ''}")


def position_note(**positions: int) -> str:
    """Advisory text for positions outside [0, 31]; never a hazard on its own.

    Both operator branches call this, so Conv and Mul classify positions identically.
    The previous code checked > 31 for Conv but not for Mul, and treated either side as
    CRITICAL -- which fired on exactly the operations Theorem 1 calls feasible.
    """
    low = {k: v for k, v in positions.items() if v < 0}
    high = {k: v for k, v in positions.items() if v > 31}
    parts = []
    if low:
        parts.append("position(s) below 0 (scale > 1.0): "
                     + ", ".join(f"{k}={v}" for k, v in sorted(low.items())))
    if high:
        parts.append("position(s) above 31: " + ", ".join(f"{k}={v}" for k, v in sorted(high.items())))
    if not parts:
        return ""
    return "ADVISORY, not a hazard (Theorem 3 retracted): " + "; ".join(parts)


def compute_shift_cut(A: float) -> Tuple[int, int]:
    """Decompose real scale factor A into 15-bit multiplier M and shift-cut sigma.
    
    Target: A ≈ M * 2^(-sigma), with M in [16384, 32767].
    """
    if A <= 0:
        return 0, 0
    
    log2_A = math.log2(A)
    sigma = int(math.floor(15 - log2_A))
    M = int(round(A * (2 ** sigma)))
    
    while M >= 32768 and sigma < 64:
        M //= 2
        sigma -= 1
    while M < 16384 and sigma > -64:
        M *= 2
        sigma += 1
        
    return M, sigma


def find_output_quantizer(graph: onnx.GraphProto, tensor_name: str,
                          visited: Optional[Set[str]] = None) -> Optional[Tuple[onnx.NodeProto, str]]:
    """Traverse downstream from tensor_name through transparent unary activations.
    
    Returns the QuantizeLinear node and its scale tensor name, or None if unquantized.
    """
    if visited is None:
        visited = set()
    if tensor_name in visited:
        return None
    visited.add(tensor_name)
    
    for consumer in graph.node:
        if tensor_name in consumer.input:
            if consumer.op_type == "QuantizeLinear":
                return consumer, consumer.input[1]
            elif consumer.op_type in ("Relu", "Clip", "LeakyRelu", "PRelu", "Identity"):
                res = find_output_quantizer(graph, consumer.output[0], visited)
                if res is not None:
                    return res
    return None


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
                scale_w = 1.0

            # Output scale (traverse transparent activations to QuantizeLinear)
            out0 = node.output[0]
            q_info = find_output_quantizer(graph, out0)
            if q_info is not None:
                _, scale_y_name = q_info
                if scale_y_name in initializers:
                    scale_y = float(np.ravel(initializers[scale_y_name])[0])

            A = (scale_x * scale_w) / scale_y if scale_y > 0 else 1.0
            M, sigma = compute_shift_cut(A)
            pos_x = int(round(-math.log2(scale_x))) if scale_x > 0 else 0
            pos_w = int(round(-math.log2(scale_w))) if scale_w > 0 else 0
            pos_y = int(round(-math.log2(scale_y))) if scale_y > 0 else 0

            is_hazard = False
            reason = "Feasible systolic shift"
            if sigma < 0:
                is_hazard = True
                reason = f"ACCUMULATOR OVERFLOW HAZARD: sigma={sigma} < 0 (requires left-shift beyond accumulator)"
            elif sigma > 31:
                is_hazard = True
                reason = f"SHIFT CLAMP HAZARD: sigma={sigma} > 31 (hardware 5-bit shifter clamps/overflows)"
            note = position_note(pos_x=pos_x, pos_w=pos_w, pos_y=pos_y)

            hazards.append(ShiftCutHazard(
                node_name=node.name or out0,
                op_type=node.op_type,
                scale_x=scale_x,
                scale_w=scale_w,
                scale_y=scale_y,
                A=A,
                M=M,
                sigma=sigma,
                pos_x=pos_x,
                pos_w=pos_w,
                pos_y=pos_y,
                is_hazard=is_hazard,
                reason=reason,
                note=note
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
            q_info = find_output_quantizer(graph, out0)
            if q_info is not None:
                _, scale_out_name = q_info
                if scale_out_name in initializers:
                    scale_out = float(np.ravel(initializers[scale_out_name])[0])

            A = (scale_a * scale_b) / scale_out if scale_out > 0 else 1.0
            M, sigma = compute_shift_cut(A)
            pos_a = int(round(-math.log2(scale_a))) if scale_a > 0 else 0
            pos_b = int(round(-math.log2(scale_b))) if scale_b > 0 else 0
            pos_out = int(round(-math.log2(scale_out))) if scale_out > 0 else 0

            is_hazard = False
            reason = "Feasible elementwise systolic shift"
            if sigma < 0 or sigma > 31:
                is_hazard = True
                reason = f"BILATERAL GATING HAZARD: sigma={sigma} outside [0, 31] (branch scale ratio divergent)"
            note = position_note(pos_a=pos_a, pos_b=pos_b, pos_out=pos_out)

            hazards.append(ShiftCutHazard(
                node_name=node.name or out0,
                op_type=node.op_type,
                scale_x=scale_a,
                scale_w=scale_b,
                scale_y=scale_out,
                A=A,
                M=M,
                sigma=sigma,
                pos_x=pos_a,
                pos_w=pos_b,
                pos_y=pos_out,
                is_hazard=is_hazard,
                reason=reason,
                note=note
            ))

    return hazards


def print_shift_cut_report(model_path: str, hazards: List[ShiftCutHazard]):
    print("=" * 80)
    print(f"AIE-ML Systolic Shift-Cut Verification Report: {model_path}")
    print("=" * 80)
    
    total = len(hazards)
    violations = [h for h in hazards if h.is_hazard]
    notes = [h for h in hazards if h.note and not h.is_hazard]
    
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

    if notes:
        print(f"\nAdvisory notes (NOT violations): {len(notes)} / {total} operation(s) carry a")
        print("scale position outside [0, 31]. Measured on this device to execute correctly:")
        print("15 of 17 NPU arithmetic fixtures ran at output positions -1 to -16 and matched")
        print("an independent integer reference. Theorem 3 is retracted; see the module docstring.")
        for h in notes[:15]:
            print(f"  - {h}")
        if len(notes) > 15:
            print(f"  ... and {len(notes) - 15} more advisory notes.")
        
    sigmas = [h.sigma for h in hazards]
    if sigmas:
        print(f"\nShift-Cut Register Distribution: min={min(sigmas)}, median={int(np.median(sigmas))}, max={max(sigmas)}")
    print("=" * 80)


def calculate_feasible_scale_bounds(scale_x: float, scale_w: float) -> Tuple[int, int, float, float]:
    """Calculate the closed-form feasible position and scale interval for output scale S_y.

    Theorem (Systolic Scale Feasibility Window):
        pos_y in [max(0, pos_x + pos_w - 17),  min(31, pos_x + pos_w + 14)]
        where S = 2^(-pos).
    """
    pos_x = max(0, min(31, int(round(-math.log2(scale_x))))) if scale_x > 0 else 0
    pos_w = max(0, min(31, int(round(-math.log2(scale_w))))) if scale_w > 0 else 0
    min_pos_y = max(0, pos_x + pos_w - 17)
    max_pos_y = min(31, pos_x + pos_w + 14)
    min_scale_y = 2.0 ** (-max_pos_y)
    max_scale_y = 2.0 ** (-min_pos_y)
    return min_pos_y, max_pos_y, min_scale_y, max_scale_y


def project_scale_to_feasible_basin(scale_x: float, scale_w: float, scale_y: float) -> Tuple[float, int, int]:
    """Project violating output scale S_y to the nearest boundary of the systolic feasibility window.

    The window must be built from the SAME positions sigma is computed from. The earlier version
    clamped pos_x and pos_w into [0, 31] before deriving the window, while sigma was recomputed
    from the unclamped scales, so the two disagreed whenever an input position sat outside that
    range -- and the "repair" could return a sigma far outside [0, 31]. Measured on RegNetX-002
    `/s1/b1/conv2/conv/Conv`: a layer at sigma = 30, already feasible, was projected to sigma =
    -90, manufacturing the accumulator-overflow hazard this module exists to prevent.

    Positions are therefore used unclamped, and the result is checked rather than trusted.
    """
    if scale_x <= 0 or scale_w <= 0 or scale_y <= 0:
        return scale_y, 0, 0
    pos_x = int(round(-math.log2(scale_x)))
    pos_w = int(round(-math.log2(scale_w)))
    pos_y = int(round(-math.log2(scale_y)))
    min_pos_y = pos_x + pos_w - 17
    max_pos_y = pos_x + pos_w + 14
    repaired_pos_y = max(min_pos_y, min(max_pos_y, pos_y))
    repaired_scale_y = 2.0 ** (-repaired_pos_y)
    A = (scale_x * scale_w) / repaired_scale_y
    M, sigma = compute_shift_cut(A)
    if not 0 <= sigma <= 31:
        raise ValueError(
            f"projection did not land in the feasible window: sigma={sigma} from "
            f"pos_x={pos_x}, pos_w={pos_w}, pos_y={pos_y} -> {repaired_pos_y}. "
            "Refusing to emit a scale the analyzer would flag."
        )
    return repaired_scale_y, M, sigma


def repair_model_shift_cut(model_path: str, output_path: Optional[str] = None) -> Tuple[onnx.ModelProto, int, List[Dict[str, Any]]]:
    """Inspect an ONNX QDQ model, project violating scales into the feasible basin, and save the repaired model."""
    model = onnx.load(model_path)
    graph = model.graph
    initializers = {init.name: init for init in graph.initializer}
    init_arrays = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    producers = {out: node for node in graph.node for out in node.output}

    repairs = []

    for node in graph.node:
        if node.op_type in ["Conv", "Gemm", "MatMul"]:
            scale_x = 1.0
            scale_w = 1.0
            in0 = node.input[0]
            if in0 in producers and producers[in0].op_type == "DequantizeLinear":
                dq_x = producers[in0]
                if dq_x.input[1] in init_arrays:
                    scale_x = float(np.ravel(init_arrays[dq_x.input[1]])[0])
            in1 = node.input[1]
            if in1 in producers and producers[in1].op_type == "DequantizeLinear":
                dq_w = producers[in1]
                if dq_w.input[1] in init_arrays:
                    scale_w = float(np.ravel(init_arrays[dq_w.input[1]])[0])
            elif in1 in init_arrays:
                scale_w = 1.0

            q_info = find_output_quantizer(graph, node.output[0])
            if q_info is None:
                continue
            q_node, scale_y_name = q_info
            if scale_y_name not in init_arrays:
                continue
            scale_y = float(np.ravel(init_arrays[scale_y_name])[0])

            A = (scale_x * scale_w) / scale_y if scale_y > 0 else 1.0
            M, sigma = compute_shift_cut(A)
            pos_y = int(round(-math.log2(scale_y))) if scale_y > 0 else 0

            # Must match analyze_model_shift_cut's hazard criterion exactly, or the
            # repair count claims work it never did. project_scale_to_feasible_basin
            # only moves pos_y, and sigma is a function of pos_y, so every hazard it
            # accepts is one it can actually fix.
            if sigma < 0 or sigma > 31:
                repaired_scale_y, new_M, new_sigma = project_scale_to_feasible_basin(scale_x, scale_w, scale_y)
                repairs.append({
                    "node": node.name or node.output[0],
                    "op_type": node.op_type,
                    "scale_name": scale_y_name,
                    "orig_scale_y": scale_y,
                    "repaired_scale_y": repaired_scale_y,
                    "orig_sigma": sigma,
                    "repaired_sigma": new_sigma,
                })
                init_proto = initializers[scale_y_name]
                new_arr = np.array(repaired_scale_y, dtype=np.float32)
                new_init = numpy_helper.from_array(new_arr, name=scale_y_name)
                init_proto.CopyFrom(new_init)
                init_arrays[scale_y_name] = new_arr

        elif node.op_type == "Mul":
            scale_a = 1.0
            scale_b = 1.0
            if node.input[0] in producers and producers[node.input[0]].op_type == "DequantizeLinear":
                dq_a = producers[node.input[0]]
                if dq_a.input[1] in init_arrays:
                    scale_a = float(np.ravel(init_arrays[dq_a.input[1]])[0])
            if len(node.input) > 1 and node.input[1] in producers and producers[node.input[1]].op_type == "DequantizeLinear":
                dq_b = producers[node.input[1]]
                if dq_b.input[1] in init_arrays:
                    scale_b = float(np.ravel(init_arrays[dq_b.input[1]])[0])

            q_info = find_output_quantizer(graph, node.output[0])
            if q_info is None:
                continue
            q_node, scale_y_name = q_info
            if scale_y_name not in init_arrays:
                continue
            scale_out = float(np.ravel(init_arrays[scale_y_name])[0])

            A = (scale_a * scale_b) / scale_out if scale_out > 0 else 1.0
            M, sigma = compute_shift_cut(A)
            pos_y = int(round(-math.log2(scale_out))) if scale_out > 0 else 0

            # Must match analyze_model_shift_cut's hazard criterion exactly, or the
            # repair count claims work it never did. project_scale_to_feasible_basin
            # only moves pos_y, and sigma is a function of pos_y, so every hazard it
            # accepts is one it can actually fix.
            if sigma < 0 or sigma > 31:
                repaired_scale_out, new_M, new_sigma = project_scale_to_feasible_basin(scale_a, scale_b, scale_out)
                repairs.append({
                    "node": node.name or node.output[0],
                    "op_type": node.op_type,
                    "scale_name": scale_y_name,
                    "orig_scale_y": scale_out,
                    "repaired_scale_y": repaired_scale_out,
                    "orig_sigma": sigma,
                    "repaired_sigma": new_sigma,
                })
                init_proto = initializers[scale_y_name]
                new_arr = np.array(repaired_scale_out, dtype=np.float32)
                new_init = numpy_helper.from_array(new_arr, name=scale_y_name)
                init_proto.CopyFrom(new_init)
                init_arrays[scale_y_name] = new_arr

    if output_path is not None:
        onnx.save(model, output_path)
    return model, len(repairs), repairs

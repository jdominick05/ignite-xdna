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

Theorem 1 (Systolic Shift-Cut Bound) -- STATED, NOT VALIDATED ON THIS DEVICE:
    The criterion this module applies is:
        0 <= sigma <= 31

    Since M ≈ 2^14 and A = 2^(pos_y - pos_x - pos_w), we have:
        sigma = pos_x + pos_w - pos_y + 14

    If sigma < 0, the operation would require an arithmetic left-shift exceeding
    accumulator precision (accumulator overflow hazard).
    If sigma > 31, the hardware 5-bit shifter would clamp to 31 (shift clamp hazard).

    Read the "if and only if" as a hypothesis, not a result. Both register widths --
    the 15-bit multiplier and the 5-bit shifter -- are asserted here, not measured, and
    no AIE-ML or DPU ISA document in this repo states either. AMD's own AI Engine
    documentation describes the accumulator-to-vector path as SRS (shift-round-saturate)
    without giving the shift field's width, so the bound is not externally corroborated
    either.

    Neither edge has been observed on hardware. The highest sigma ever executed on this
    DPU is 30; measured placements cover sigma in {14, 15, 17, 18, 22, 30}, all from
    fixtures whose scales sit inside the producer's own [0, 16] shift-cut contract.
    Fixtures built to reach sigma 0, 31 and 32 were rejected by the VitisAI EP -- the
    out-of-contract scales make it refuse the Conv, so those runs record
    "tested_conv_on_npu": false and ran on the CPU EP. A divergence measured there is a
    software result and says nothing about the DPU shifter or the accumulator. Any claim
    that an edge is "measured" needs an NPU:Conv in the EP report, not a fallback.

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

The producer contract, and why [0, 31] cannot fire on a file this repo produced
-------------------------------------------------------------------------------
Quark's own refinement step is the binding constraint, and it is much tighter than
Theorem 1. ``adjust_shift_cut`` (transcribed at quant/refine.py::shift_cut, sourced in
results/quant/notes_xint8_dialect.log:37 and :1216-1253) defines

    shift_cut = wpos + ipos - opos,    and clamps it into [0, 16]

for Conv and Gemm. This module's sigma is pos_x + pos_w - pos_y + 14 over the same three
positions, so the two differ by a constant:

    sigma == shift_cut + 14

and the producer's contract is therefore exactly ``sigma in [14, 30]``, which is a strict
subset of Theorem 1's [0, 31]. A Conv or Gemm in a file Quark or Ignition emitted cannot
land outside [0, 31] without the producer's clamp having failed first.

That makes the "0 violations across 11 models" result a tautology rather than evidence:
the [0, 31] test is structurally incapable of firing on an in-contract file. It is kept
because a hand-edited, CLE-inflated or foreign-producer file is not in-contract -- but a
PASS from it says only "this file came from the producer", not "this file is safe".
MEASURED, this sitting, on ten quantized models: every resolved Conv/Gemm sigma lies in
[14, 30], with RegNetX-002 and ResNeXt-50 touching both edges exactly
(results/quant/shift_cut_contract_20260909_desktop2.log).

Unresolved scales were being reported as feasible
--------------------------------------------------
The mirror of the false-positive problem this module was corrected for in 311a672. The
analyzer initialised scale_x/scale_w/scale_y to 1.0 and overwrote each only where a
DequantizeLinear producer with an initializer scale existed; where none existed the op was
still scored, still got a sigma, and still counted as a passing operation. On
yolov8n_cut_xint8 that was 57 of 177 "analyzed" operations -- the ``<out>_Scale`` /
``<out>_Mul`` pairs that passes.py::_insert_mul writes for DPU simulation, whose inputs are
a Constant and a HardSigmoid rather than a QDQ pair. Their reported sigma of 7 was computed
from two invented 1.0 scales, and it is those vacuous rows, not any Conv, that produced the
sub-14 minima in the published distributions.

Such operations are now classified UNRESOLVED: excluded from the violation denominator and
from the sigma distribution, and counted in their own line so coverage is visible.
"""

import math
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import onnx
from onnx import numpy_helper


# Theorem 1's stated hardware window. Neither edge has been executed on this device.
STATED_SIGMA = (0, 31)
# Quark's own adjust_shift_cut contract, transcribed at quant/refine.py::shift_cut.
# sigma = shift_cut + 14, so [0, 16] on shift_cut is [14, 30] on sigma.
CONTRACT_SHIFT_CUT = (0, 16)
CONTRACT_SIGMA = (CONTRACT_SHIFT_CUT[0] + 14, CONTRACT_SHIFT_CUT[1] + 14)


class ShiftCutHazard:
    def __init__(self, node_name: str, op_type: str, scale_x: float, scale_w: float, scale_y: float,
                 A: float, M: int, sigma: int, pos_x: int, pos_w: int, pos_y: int,
                 is_hazard: bool, reason: str, note: str = "",
                 resolved: bool = True, unresolved_reason: str = ""):
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
        # False when any of the three scales could not be read off a DequantizeLinear with
        # an initializer scale. Such a row's sigma was computed from invented 1.0 defaults,
        # so it is neither a pass nor a hazard -- it is not an analysis at all.
        self.resolved = resolved
        self.unresolved_reason = unresolved_reason

    @property
    def clamp_bound(self) -> bool:
        """Is this operation the one adjust_shift_cut actually clamps?

        Quark's shift_cut rule applies to Conv and Gemm ONLY (quant/refine.py::shift_cut,
        `if node.op_type not in ("Conv", "Gemm"): continue`; sourced at
        results/quant/notes_xint8_dialect.log:37). MatMul is not refined by it at all, and a
        Mul goes through shift_write_mul (clamp 0..32) and shift_swish (clamp 0..15) --
        different rules over different quantities. Judging either against [14, 30] would
        report "this file was not emitted by Quark/Ignition" about a file that was.
        """
        return self.op_type in ("Conv", "Gemm")

    @property
    def in_contract(self) -> bool:
        """Inside Quark's own adjust_shift_cut window.

        Vacuously True for an operation the clamp does not bind, so that a caller filtering
        on `not in_contract` never collects one.
        """
        if not self.clamp_bound:
            return True
        return CONTRACT_SIGMA[0] <= self.sigma <= CONTRACT_SIGMA[1]

    @property
    def at_contract_edge(self) -> Optional[str]:
        """"low"/"high" when the producer's clamp had this op hard against an edge."""
        if not self.resolved or not self.clamp_bound:
            return None
        if self.sigma == CONTRACT_SIGMA[0]:
            return "low"
        if self.sigma == CONTRACT_SIGMA[1]:
            return "high"
        return None

    @property
    def shift_cut(self) -> int:
        """The vendor's own quantity: sigma - 14, i.e. wpos + ipos - opos."""
        return self.sigma - 14

    def __repr__(self):
        if not self.resolved:
            return (f"[UNRESOLVED] {self.op_type} '{self.node_name}': {self.unresolved_reason} "
                    f"-- not analyzed (a sigma here would rest on invented 1.0 scales)")
        status = "HAZARD" if self.is_hazard else ("NOTE" if self.note else "OK")
        return (f"[{status}] {self.op_type} '{self.node_name}': "
                f"A={self.A:.6e}, M={self.M}, sigma={self.sigma} (shift_cut={self.shift_cut}) "
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


def _resolve_scale(producers: Dict[str, onnx.NodeProto], initializers: Dict[str, Any],
                   tensor: str, label: str) -> Tuple[float, str]:
    """Read a quantization scale off the DequantizeLinear that produces ``tensor``.

    Returns (scale, reason). ``reason`` is "" when the scale was genuinely read; otherwise
    the scale is the 1.0 the old code used silently and the reason says why it is not real.
    A per-channel scale is refused rather than approximated by its first channel: the XINT8
    dialect is per-tensor, so an array here is a model this analyzer has never been checked
    against, and reading channel 0 would report one channel's feasibility as the layer's.
    """
    node = producers.get(tensor)
    if node is None:
        return 1.0, f"{label}: no producer for {tensor!r} (graph input or initializer)"
    if node.op_type != "DequantizeLinear":
        return 1.0, f"{label}: producer of {tensor!r} is {node.op_type}, not DequantizeLinear"
    name = node.input[1]
    if name not in initializers:
        return 1.0, f"{label}: scale {name!r} is not an initializer"
    array = np.ravel(initializers[name])
    if array.size != 1:
        return 1.0, f"{label}: per-channel scale ({array.size} channels) is not transcribed"
    return float(array[0]), ""


def _position(scale: float) -> int:
    return int(round(-math.log2(scale))) if scale > 0 else 0


def _classify(sigma: int, kind: str, op_type: str) -> Tuple[bool, str]:
    """Hazard verdict for a resolved operation, plus its reason line.

    Only Theorem 1's [0, 31] is treated as a hazard, and the docstring records that an
    in-contract file cannot reach it. Sitting outside the producer's own [14, 30] is
    reported, but as an observation: it means the file did not come from this producer,
    which is a provenance fact rather than a measured hardware failure -- and it is only
    asked of Conv and Gemm, the only operations adjust_shift_cut clamps.
    """
    low, high = STATED_SIGMA
    if sigma < low:
        return True, (f"ACCUMULATOR OVERFLOW HAZARD: sigma={sigma} < {low} "
                      f"(requires left-shift beyond accumulator; UNVALIDATED bound)")
    if sigma > high:
        return True, (f"SHIFT CLAMP HAZARD: sigma={sigma} > {high} "
                      f"(5-bit shifter would clamp; UNVALIDATED bound)")
    if op_type not in ("Conv", "Gemm"):
        return False, (f"Feasible {kind} shift. No producer contract is asserted: "
                       f"adjust_shift_cut clamps Conv/Gemm only, not {op_type}")
    if not (CONTRACT_SIGMA[0] <= sigma <= CONTRACT_SIGMA[1]):
        return False, (f"Outside the producer contract sigma in {list(CONTRACT_SIGMA)} "
                       f"(shift_cut={sigma - 14} outside {list(CONTRACT_SHIFT_CUT)}): this file "
                       f"was not emitted by Quark/Ignition, or its clamp did not run")
    return False, f"Feasible {kind} shift, inside the producer contract"


def _branch_divergence(graph: onnx.GraphProto, producers: Dict[str, onnx.NodeProto],
                       initializers: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Inter-branch scale spread at every quantized multi-input Add/Concat.

    ADVISORY, and deliberately not a hazard. A DPU must bring two branches to a common
    output scale before it can add or concatenate them, so a large spread is where that
    alignment costs the most precision -- but no divergence has been measured to change an
    output on this device, and BiSeNetV2's bilateral aggregation reads 0 violations under
    every criterion in this module. Reported so the quantity is visible, never gated on.
    """
    rows = []
    for node in graph.node:
        if node.op_type not in ("Add", "Concat") or node.domain:
            continue
        positions, unresolved = [], []
        for i, tensor in enumerate(node.input):
            scale, reason = _resolve_scale(producers, initializers, tensor, f"in{i}")
            if reason:
                unresolved.append(reason)
            else:
                positions.append(_position(scale))
        if len(positions) < 2:
            continue
        spread = max(positions) - min(positions)
        rows.append({"node": node.name or node.output[0], "op_type": node.op_type,
                     "positions": positions, "spread": spread,
                     "unresolved": len(unresolved)})
    return rows


def analyze_model_shift_cut(model_path: str) -> List[ShiftCutHazard]:
    """Walk an ONNX QDQ model and evaluate the shift-cut feasibility of all nodes.

    An operation whose scales cannot all be read is returned with ``resolved`` False and is
    never counted as a pass; see the module docstring for the 57-of-177 case that made this
    necessary.
    """
    model = onnx.load(model_path)
    graph = model.graph

    initializers = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    producers = {out: node for node in graph.node for out in node.output}

    hazards: List[ShiftCutHazard] = []

    for node in graph.node:
        if node.op_type in ("Conv", "Gemm", "MatMul"):
            kind, labels = "systolic", ("input", "weight")
            operands = (node.input[0], node.input[1])
        elif node.op_type == "Mul":
            kind, labels = "elementwise", ("a", "b")
            operands = (node.input[0], node.input[1] if len(node.input) > 1 else None)
        else:
            continue

        reasons = []
        scale_a, reason = _resolve_scale(producers, initializers, operands[0], labels[0])
        if reason:
            reasons.append(reason)
        if operands[1] is None:
            scale_b, reason = 1.0, f"{labels[1]}: {node.op_type} has a single input"
            reasons.append(reason)
        else:
            scale_b, reason = _resolve_scale(producers, initializers, operands[1], labels[1])
            if reason:
                reasons.append(reason)

        scale_y, q_info = 1.0, find_output_quantizer(graph, node.output[0])
        if q_info is None:
            reasons.append("output: no downstream QuantizeLinear")
        elif q_info[1] not in initializers:
            reasons.append(f"output: scale {q_info[1]!r} is not an initializer")
        else:
            array = np.ravel(initializers[q_info[1]])
            if array.size != 1:
                reasons.append(f"output: per-channel scale ({array.size} channels) is not transcribed")
            else:
                scale_y = float(array[0])

        A = (scale_a * scale_b) / scale_y if scale_y > 0 else 1.0
        M, sigma = compute_shift_cut(A)
        pos_a, pos_b, pos_y = _position(scale_a), _position(scale_b), _position(scale_y)

        if reasons:
            hazards.append(ShiftCutHazard(
                node_name=node.name or node.output[0], op_type=node.op_type,
                scale_x=scale_a, scale_w=scale_b, scale_y=scale_y, A=A, M=M, sigma=sigma,
                pos_x=pos_a, pos_w=pos_b, pos_y=pos_y, is_hazard=False, reason="not analyzed",
                note="", resolved=False, unresolved_reason="; ".join(reasons)))
            continue

        is_hazard, reason_text = _classify(sigma, kind, node.op_type)
        note_kwargs = ({"pos_x": pos_a, "pos_w": pos_b, "pos_y": pos_y} if kind == "systolic"
                       else {"pos_a": pos_a, "pos_b": pos_b, "pos_out": pos_y})
        hazards.append(ShiftCutHazard(
            node_name=node.name or node.output[0], op_type=node.op_type,
            scale_x=scale_a, scale_w=scale_b, scale_y=scale_y, A=A, M=M, sigma=sigma,
            pos_x=pos_a, pos_w=pos_b, pos_y=pos_y, is_hazard=is_hazard, reason=reason_text,
            note=position_note(**note_kwargs), resolved=True))

    return hazards


def analyze_branch_divergence(model_path: str) -> List[Dict[str, Any]]:
    """Inter-branch scale spread at quantized Add/Concat. Advisory; see _branch_divergence."""
    model = onnx.load(model_path)
    graph = model.graph
    initializers = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    producers = {out: node for node in graph.node for out in node.output}
    return _branch_divergence(graph, producers, initializers)


def print_shift_cut_report(model_path: str, hazards: List[ShiftCutHazard],
                           divergence: Optional[List[Dict[str, Any]]] = None):
    print("=" * 80)
    print(f"AIE-ML Systolic Shift-Cut Verification Report: {model_path}")
    print("=" * 80)

    total = len(hazards)
    resolved = [h for h in hazards if h.resolved]
    unresolved = [h for h in hazards if not h.resolved]
    violations = [h for h in resolved if h.is_hazard]
    out_of_contract = [h for h in resolved if not h.is_hazard and not h.in_contract]
    notes = [h for h in resolved if h.note and not h.is_hazard]

    print(f"Candidate operations: {total}  (Conv/Gemm/MatMul and Mul)")
    print(f"Analyzed             : {len(resolved)}  -- all three scales read from QDQ initializers")
    print(f"Not analyzable       : {len(unresolved)}  -- excluded from every figure below")
    print(f"Shift-cut violations found: {len(violations)} / {len(resolved)} "
          f"({(len(violations) / max(1, len(resolved))) * 100:.1f}% of analyzed)")

    if violations:
        print("\n" + "!" * 80)
        print("HAZARDS AGAINST THE STATED (UNVALIDATED) BOUND sigma in [0, 31]:")
        print("!" * 80)
        for v in violations[:15]:
            print(f"  - {v}")
        if len(violations) > 15:
            print(f"  ... and {len(violations) - 15} more violations.")
    else:
        print(f"\nPASS against sigma in {list(STATED_SIGMA)}. Read this narrowly: Quark's own")
        print(f"adjust_shift_cut clamps shift_cut into {list(CONTRACT_SHIFT_CUT)}, i.e. sigma into")
        print(f"{list(CONTRACT_SIGMA)}, so a file this producer emitted CANNOT reach the [0, 31]")
        print("edges. A pass here means the file is in-contract, not that the hardware is safe.")

    if unresolved:
        print(f"\nNOT ANALYZABLE: {len(unresolved)} / {total} operation(s). Their scales could not")
        print("be read from a DequantizeLinear, so any sigma would rest on invented 1.0 defaults.")
        print("These were silently counted as passing before 2026-09-09; they are excluded now.")
        for h in unresolved[:10]:
            print(f"  - {h}")
        if len(unresolved) > 10:
            print(f"  ... and {len(unresolved) - 10} more.")

    if out_of_contract:
        print(f"\nOUTSIDE THE PRODUCER CONTRACT (observation, not a hazard): {len(out_of_contract)}")
        print(f"operation(s) with sigma outside {list(CONTRACT_SIGMA)}. This file was not emitted")
        print("by Quark/Ignition, or its adjust_shift_cut did not run.")
        for h in out_of_contract[:10]:
            print(f"  - {h}")
        if len(out_of_contract) > 10:
            print(f"  ... and {len(out_of_contract) - 10} more.")

    if notes:
        print(f"\nAdvisory notes (NOT violations): {len(notes)} / {len(resolved)} operation(s) carry a")
        print("scale position outside [0, 31]. Measured on this device to execute correctly:")
        print("15 of 17 NPU arithmetic fixtures ran at output positions -1 to -16 and matched")
        print("an independent integer reference. Theorem 3 is retracted; see the module docstring.")
        for h in notes[:15]:
            print(f"  - {h}")
        if len(notes) > 15:
            print(f"  ... and {len(notes) - 15} more advisory notes.")

    matmul = [h for h in resolved if h.op_type == "MatMul"]
    if matmul:
        sigmas = [h.sigma for h in matmul]
        print(f"\nMatMul distribution over {len(matmul)} analyzed op(s): "
              f"sigma min={min(sigmas)}, median={int(np.median(sigmas))}, max={max(sigmas)}")
        print("  Reported without a contract verdict: adjust_shift_cut clamps Conv/Gemm only,")
        print("  so a MatMul's sigma is not bound by [14, 30] even when it happens to land there.")

    systolic = [h for h in resolved if h.clamp_bound]
    if systolic:
        sigmas = [h.sigma for h in systolic]
        low = [h for h in systolic if h.at_contract_edge == "low"]
        high = [h for h in systolic if h.at_contract_edge == "high"]
        print(f"\nConv/Gemm shift-cut distribution over {len(systolic)} analyzed op(s) "
              f"(the operations adjust_shift_cut actually clamps):")
        print(f"  sigma      min={min(sigmas)}, median={int(np.median(sigmas))}, max={max(sigmas)}")
        print(f"  shift_cut  min={min(sigmas) - 14}, median={int(np.median(sigmas)) - 14}, max={max(sigmas) - 14}"
              f"   (the vendor's own quantity, contract {list(CONTRACT_SHIFT_CUT)})")
        print(f"  at contract edges: {len(low)} at sigma={CONTRACT_SIGMA[0]}, {len(high)} at sigma={CONTRACT_SIGMA[1]}")
        if low and high:
            pct = 100.0 * (len(low) + len(high)) / len(systolic)
            print(f"  TWO-SIDED SATURATION: {pct:.1f}% of layers pinned to an edge. The producer's")
            print("  clamp had to act at both ends, which is what an inflated per-channel range")
            print("  looks like from here. MEASURED to co-occur with catastrophic accuracy loss on")
            print("  RegNetX-002 and ResNeXt-50 (0.10% top-1, both recovering with CLE off), but")
            print("  n=2 and both are grouped-convolution models, so equalization is NOT separated")
            print("  from architecture by this observation alone. See the module docstring.")

    mul = [h for h in resolved if h.op_type == "Mul"]
    if mul:
        sigmas = [h.sigma for h in mul]
        print(f"\nMul (elementwise) distribution over {len(mul)} analyzed op(s): "
              f"min={min(sigmas)}, median={int(np.median(sigmas))}, max={max(sigmas)}")

    if divergence:
        spreads = [d["spread"] for d in divergence]
        worst = sorted(divergence, key=lambda d: -d["spread"])[:5]
        print(f"\nMulti-branch inter-scale divergence (ADVISORY, never a hazard): "
              f"{len(divergence)} Add/Concat")
        print(f"  branch position spread: min={min(spreads)}, median={int(np.median(spreads))}, "
              f"max={max(spreads)} bits")
        print("  A DPU aligns branches to one output scale before adding or concatenating, so a")
        print("  wide spread is where that alignment costs most. No divergence has been measured")
        print("  to change an output on this device; this is reported, not gated on.")
        for d in worst:
            if d["spread"]:
                print(f"  - {d['op_type']} '{d['node']}': positions {d['positions']} spread={d['spread']}")
    print("=" * 80)


def calculate_feasible_scale_bounds(scale_x: float, scale_w: float) -> Tuple[int, int, float, float]:
    """Calculate the closed-form feasible position and scale interval for output scale S_y.

    Theorem 2 (Systolic Scale Feasibility Window):
        pos_y in [pos_x + pos_w - 17,  pos_x + pos_w + 14]
        where S = 2^(-pos).

    Positions are used UNCLAMPED, for the same reason project_scale_to_feasible_basin
    does: clamping pos_x and pos_w into [0, 31] here while sigma is computed from the
    unclamped scales makes the two disagree whenever an input position sits outside that
    range, which is what projected an already-feasible RegNetX-002 layer from sigma = 30
    to sigma = -90. An earlier version of this function clamped both, and the window it
    returned could therefore exclude the position the analyzer would accept.

    NOTE: this helper currently has no callers. It is kept because it states the window
    in closed form, and it is corrected so that wiring it up cannot reintroduce that bug.
    """
    if scale_x <= 0 or scale_w <= 0:
        return 0, 0, 0.0, 0.0
    pos_x = int(round(-math.log2(scale_x)))
    pos_w = int(round(-math.log2(scale_w)))
    min_pos_y = pos_x + pos_w - 17
    max_pos_y = pos_x + pos_w + 14
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
            # Resolve exactly as the analyzer does. An operation whose scales are not all
            # readable is skipped, never repaired: projecting from an invented 1.0 is how
            # a feasible RegNetX-002 layer was driven to sigma = -90 (see
            # project_scale_to_feasible_basin), and the repair must not outrun the analysis.
            scale_x, why_x = _resolve_scale(producers, init_arrays, node.input[0], "input")
            scale_w, why_w = _resolve_scale(producers, init_arrays, node.input[1], "weight")
            if why_x or why_w:
                continue

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
            # Same rule as the Conv branch: no repair on an operation we could not analyze.
            # This is the branch that covers passes.py::_insert_mul's DPU-simulation Muls,
            # whose Constant/HardSigmoid inputs are not QDQ pairs at all.
            if len(node.input) < 2:
                continue
            scale_a, why_a = _resolve_scale(producers, init_arrays, node.input[0], "a")
            scale_b, why_b = _resolve_scale(producers, init_arrays, node.input[1], "b")
            if why_a or why_b:
                continue

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

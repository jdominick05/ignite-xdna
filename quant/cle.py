"""Cross-layer equalization, transcribed from Quark 0.11rc1 algorithm/cle/equalization.py.

Scope is the Conv->Conv pair path that folded ResNet exercises. The matcher walks the
source's single-consumer chain links (Relu, ReduceMean, Pad, LeakyRelu) and reproduces
its pair ordering, including the insert-before-last sort that lists the first pair
twice. Pairs are equalized in that order on the live initializers, so a pair's head is
the previous pair's already-scaled tail. Depthwise pairs and triples, Gemm-in-pair
transposes and Clip replacement have no instance on the measured graph and raise
rather than shipping untested. Float32 operation order follows the source so the
equalized weights are bit-identical. Defaults are the audited ones (DESIGN.md 2.6).
"""
from dataclasses import asdict, dataclass, field

import numpy as np
from onnx import helper

from .graph import Graph

CHAIN_LINKS = ("Relu", "ReduceMean", "Pad", "LeakyRelu")
TARGETS = ("Conv", "Gemm")


@dataclass
class ClePair:
    head: str
    tail: str
    chain: list[str]
    size: int = 3  # source tuple length; a Conv -> depthwise -> pointwise triple is 4
    middle: str | None = None  # the depthwise Conv, set only when size == 4


@dataclass
class CleReport:
    pattern_count: int
    unique_pairs: int
    pairs: list[dict]
    steps: int
    last_diff: float
    weight_threshold: float
    append_bias: bool
    use_threshold: bool
    scaled: dict = field(default_factory=dict)
    # None when the guard is off, which is the default and the parity path.
    max_scale_log2: float | None = None
    skipped_unstable: list = field(default_factory=list)


def _group(node) -> int | None:
    """None when the attribute is absent: the source then treats the Conv as unsupported."""
    for attr in node.attribute:
        if attr.name == "group":
            return helper.get_attribute_value(attr)
    return None


def _supported(g: Graph, node) -> tuple[bool, int]:
    """check_conv_layers_group: group 1 or depthwise Conv, or Gemm."""
    if node.op_type == "Conv":
        group = _group(node)
        if group is None:
            return False, 0
        if group == 1:
            return True, 1
        weight = g.initializer(node.input[1])
        if weight is not None and weight.shape[1] == 1 and group == weight.shape[0]:
            return True, group
        return False, group
    if node.op_type == "Gemm":
        return True, 1
    return False, 0


def _matcher_support(g: Graph, nodes) -> bool:
    """check_conv_layers_support, including the source's early-break bug.

    The source sets a single `conv_support` flag per node and, when a grouped Conv fails
    the depthwise test, breaks out of the *attribute* loop rather than the node loop. The
    flag is then overwritten by the next node, so the returned verdict is whichever the
    **last** node produced -- a grouped head followed by a group-1 tail is accepted. Only a
    node that is neither Conv nor Gemm breaks the outer loop and sticks.

    Reproducing it matters: on RegNetX-002 it is the difference between 13 matched pairs
    and none, and a stricter reading silently drops thirteen of the vendor's 27 patterns.
    A missing group attribute leaves the flag untouched, which is why it counts as
    supported here.
    """
    supported = True
    for node in nodes:
        if node.op_type == "Conv":
            group = _group(node)
            if group is not None:
                if group == 1:
                    supported = True
                else:
                    weight = g.initializer(node.input[1])
                    supported = bool(weight is not None and weight.shape[1] == 1
                                     and group == weight.shape[0])
        elif node.op_type == "Gemm":
            supported = True
        else:
            return False
    return supported


def find_pairs(g: Graph) -> list[ClePair]:
    """get_cle_pattern_pair, including its ordering quirk."""
    nodes = g.nodes()
    consumers = {}
    for node in nodes:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    def outputs_of(node):
        found, names = [], set()
        for output in node.output:
            for consumer in consumers.get(output, []):
                if consumer.name not in names:
                    names.add(consumer.name)
                    found.append(consumer)
        return found

    pairs = []
    for node in nodes:
        if node.op_type not in TARGETS:
            continue
        chain = [node.output[0]]
        following = outputs_of(node)
        while following and len(following) == 1:
            nxt = following[0]
            if nxt.op_type in CHAIN_LINKS:
                chain.append(nxt.output[0])
                following = outputs_of(nxt)
            elif nxt.op_type in TARGETS:
                if _matcher_support(g, [node, nxt]):
                    chain.append(nxt.output[0])
                    pairs.append(ClePair(node.name, nxt.name, chain))
                break
            else:
                break
    # Depthwise triple pass (Conv -> depthwise Conv -> Conv through Relu only).
    for node in nodes:
        if node.op_type != "Conv":
            continue
        found = [node]
        chain = [node.output[0]]
        following = outputs_of(node)
        while following and len(following) == 1:
            nxt = following[0]
            if nxt.op_type == "Relu":
                chain.append(nxt.output[0])
                following = outputs_of(nxt)
            elif nxt.op_type == "Conv":
                found.append(nxt)
                chain.append(nxt.output[0])
                following = outputs_of(nxt)
                if len(found) == 3:
                    first, middle, last = found
                    weight = g.initializer(middle.input[1])
                    # The source requires the group attribute to be present on all three:
                    # a missing attribute leaves its flag False and the triple is not matched.
                    depthwise = (_group(middle) or 0) > 1 and weight is not None and \
                        weight.shape[0] == weight.shape[1] * _group(middle)
                    if _group(first) == 1 and depthwise and _group(last) == 1:
                        pairs.append(ClePair(first.name, last.name, chain, 4, middle.name))
                    break
            else:
                break
    ordered: list[ClePair] = []
    for node in nodes:
        for pair in pairs:
            if pair.head == node.name:
                if not ordered:
                    ordered.append(pair)
                if ordered[-1].size > pair.size:
                    ordered.append(pair)
                else:
                    ordered.insert(-1, pair)
    return ordered


def _combine_weight_and_bias(weights_ihw: np.ndarray, bias: np.ndarray | None) -> np.ndarray:
    if bias is None:
        return weights_ihw.astype(np.float32)
    bias_clamp = bias.copy().reshape(-1, 1)
    if np.count_nonzero(weights_ihw) != weights_ihw.size:
        weight_ihw_clamp = weights_ihw.copy()
        for channel in range(weight_ihw_clamp.shape[0]):
            if np.count_nonzero(weight_ihw_clamp[channel]) == 0:
                bias_clamp[channel] = 0.0
                weight_ihw_clamp[channel] = 1e-7
            elif np.count_nonzero(weight_ihw_clamp[channel]) != weight_ihw_clamp[channel].size:
                minval = np.min(np.fabs(np.ma.masked_where(weight_ihw_clamp[channel] == 0.0,
                                                           weight_ihw_clamp[channel])))
                weight_ihw_clamp[channel] = np.where(weight_ihw_clamp[channel] == 0.0, -minval,
                                                     weight_ihw_clamp[channel])
        weight_ihw_clamp = np.where(np.fabs(weight_ihw_clamp) < 1e-7, 1e-7, weight_ihw_clamp)
    else:
        weight_ihw_clamp = weights_ihw.copy()
        weight_ihw_clamp = np.where(np.fabs(weight_ihw_clamp) < 1e-7, 1e-7, weight_ihw_clamp)
    factor = np.fabs(bias_clamp) / np.fabs(weight_ihw_clamp)
    if (np.fabs(bias).max() < 10) and (np.fabs(bias).max() / np.fabs(weight_ihw_clamp).max() < 20):
        shrink_factor = 5 if (np.median(factor) > 100 or factor.mean() > 1000) else 2
    elif np.median(factor) > 30 or factor.mean() > 500:
        shrink_factor = 20
    elif np.median(factor) > 15 or factor.mean() > 100:
        shrink_factor = 10
    else:
        shrink_factor = 5
    return np.concatenate((weights_ihw, bias_clamp / shrink_factor), axis=1).astype(np.float32)


def _calc_scale(head_weights: np.ndarray, tail_weights: np.ndarray, weight_threshold: float,
                use_threshold: bool) -> np.ndarray:
    """The source's only balance method is "max"."""
    range_0 = np.max(np.fabs(head_weights), axis=1)
    range_1 = np.max(np.fabs(tail_weights), axis=1)
    sqrt_of_ranges = np.sqrt(range_0 * range_1)
    scale = np.ones_like(range_1)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(sqrt_of_ranges != 0, range_1 / sqrt_of_ranges, scale)
    if use_threshold:
        scale = np.where((range_0 + range_1) < weight_threshold, np.float32(1), scale)
    return scale


def _scale_span_log2(scale) -> float:
    """Widest power-of-two excursion in the per-channel scale, in bits.

    CLE multiplies the head's weights by this vector and the tail's by its reciprocal, so
    the weight ranges stay tidy however extreme it gets -- measured, CLE *narrows* the
    weight positions on RegNetX-002 (5..10 into 6..8) and ResNeXt-50 (4..9 into 5..8) even
    as it destroys them. What it does not rescale is the activation between the two layers,
    which is multiplied by this vector and then calibrated. That is the quantity worth
    bounding, and it separates cleanly: ResNet50, where CLE is worth +10.8 top-1, spans
    6.1 bits; RegNetX-002 spans 66.3 and ResNeXt-50 119.4, and both score 0.10%.
    """
    live = np.asarray(scale, dtype=np.float64)
    live = live[np.isfinite(live) & (live > 0)]
    if live.size == 0:
        return 0.0
    return float(np.abs(np.log2(live)).max())


def equalize_pair(g: Graph, head, tail, weight_threshold: float, append_bias: bool,
                  use_threshold: bool, max_scale_log2: float | None = None) -> dict:
    """_cross_layer_equalize for a Conv->Conv pair with group 1 on both sides."""
    if head.op_type != "Conv" or tail.op_type != "Conv":
        raise NotImplementedError("CLE with a Gemm in the pair is not implemented")
    head_ok, head_group = _supported(g, head)
    if not head_ok:
        return {"skipped": "head unsupported"}
    tail_ok, tail_group = _supported(g, tail)
    if not tail_ok:
        return {"skipped": "tail unsupported"}
    if head_group != 1 or tail_group != 1:
        raise NotImplementedError("Depthwise CLE pairs are not implemented")
    head_w = g.initializer(head.input[1])
    head_b = g.initializer(head.input[2]) if len(head.input) > 2 else None
    tail_w = g.initializer(tail.input[1])
    if head_w is None or tail_w is None or (len(head.input) > 2 and head_b is None):
        raise ValueError(f"CLE needs float initializers at {head.name} and {tail.name}")
    if any(v.dtype != np.float32 for v in (head_w, tail_w) + ((head_b,) if head_b is not None else ())):
        raise ValueError("CLE operates on float32 initializers")
    oc = head_w.shape[0]
    head_weights = head_w.reshape(oc, -1)
    if append_bias:
        head_weights = _combine_weight_and_bias(head_weights, head_b)
    ic = tail_w.shape[1]
    tail_weights = tail_w.transpose(1, 0, 2, 3).reshape(ic, -1)
    if oc != ic:
        raise ValueError(f"CLE pair channel mismatch {head.name}({oc}) -> {tail.name}({ic})")
    scale = _calc_scale(head_weights, tail_weights, weight_threshold, use_threshold)
    if scale.dtype != np.float32:
        raise ValueError("CLE scale must stay float32")
    span = _scale_span_log2(scale)
    if max_scale_log2 is not None and span > max_scale_log2:
        # Leave the pair untouched. Equalising it would multiply the activation between
        # the two layers by this scale, and nothing downstream rescales it back.
        return {"skipped": "unstable scale", "channels": int(oc), "scale_span_log2": span,
                "scale_min": float(scale.min()), "scale_max": float(scale.max())}
    g.set_initializer(head.input[1], head_w * scale.reshape(-1, 1, 1, 1))
    if head_b is not None:
        g.set_initializer(head.input[2], head_b * scale)
    g.set_initializer(tail.input[1], tail_w * (np.float32(1) / scale.reshape(1, -1, 1, 1)))
    return {"channels": int(oc), "scale_min": float(scale.min()), "scale_max": float(scale.max()),
            "unit_scales": int(np.count_nonzero(scale == 1)), "scale_span_log2": span}


def equalize_triple(g: Graph, conv, conv_dw, conv_pw,
                    max_scale_log2: float | None = None) -> dict:
    """_cle_set_with_depthwise_layers for a Conv -> depthwise Conv -> pointwise Conv triple.

    The source passes this path none of the pair path's options -- no balance method, no
    weight threshold, no bias flag, no threshold flag -- so none are accepted here. It
    equalizes three layers at once against the geometric mean of their per-channel maxima,
    and it scales the first two biases but never the third layer's.
    """
    w0, w1, w2 = (g.initializer(n.input[1]) for n in (conv, conv_dw, conv_pw))
    if w0 is None or w1 is None or w2 is None:
        raise ValueError(f"CLE needs float initializers across {conv.name} -> {conv_pw.name}")
    if any(v.dtype != np.float32 for v in (w0, w1, w2)):
        raise ValueError("CLE operates on float32 initializers")
    b0 = g.initializer(conv.input[2]) if len(conv.input) > 2 else None
    b1 = g.initializer(conv_dw.input[2]) if len(conv_dw.input) > 2 else None

    max_0 = np.max(np.fabs(w0), axis=(1, 2, 3))
    max_1 = np.max(np.fabs(w1), axis=(1, 2, 3))
    max_2 = np.max(np.fabs(w2), axis=(0, 2, 3))
    geometric = np.power(max_0 * max_1 * max_2, 1.0 / 3)
    scale_12 = max_0 / geometric
    scale_23 = geometric / max_2
    # The source nan_to_num's nan and posinf only, then maps exact zeros to one.
    scale_12 = np.nan_to_num(scale_12, nan=1.0, posinf=1.0)
    scale_23 = np.nan_to_num(scale_23, nan=1.0, posinf=1.0)
    scale_12[scale_12 == 0.0] = 1.0
    scale_23[scale_23 == 0.0] = 1.0

    span = max(_scale_span_log2(scale_12), _scale_span_log2(scale_23))
    # Measured: skipping every triple over 2 bits recovers RegNetX-002 to 66.20% and
    # ResNeXt-50 to 68.90%, which is exactly what no CLE at all gives on either. Letting
    # the milder ones through is worse than both: at 4 bits RegNetX keeps 5 of its 14 and
    # reads 25.60%.
    if max_scale_log2 is not None and span > max_scale_log2:
        return {"skipped": "unstable scale", "channels": int(max_0.shape[0]),
                "scale_span_log2": span,
                "scale_12": [float(scale_12.min()), float(scale_12.max())],
                "scale_23": [float(scale_23.min()), float(scale_23.max())]}

    g.set_initializer(conv.input[1], w0 * (1.0 / scale_12.reshape(-1, 1, 1, 1)))
    g.set_initializer(conv_dw.input[1],
                      w1 * scale_12.reshape(-1, 1, 1, 1) * (1.0 / scale_23.reshape(-1, 1, 1, 1)))
    g.set_initializer(conv_pw.input[1], w2 * scale_23.reshape(1, -1, 1, 1))
    if b0 is not None:
        g.set_initializer(conv.input[2], b0 * (1.0 / scale_12))
    if b1 is not None:
        g.set_initializer(conv_dw.input[2], b1 * (1.0 / scale_23))
    return {"channels": int(max_0.shape[0]), "scale_span_log2": span,
            "scale_12": [float(scale_12.min()), float(scale_12.max())],
            "scale_23": [float(scale_23.min()), float(scale_23.max())]}


def cross_layer_equalize(g: Graph, *, steps: int = 1, balance_method: str = "max",
                         weight_threshold: float = 0.5, append_bias: bool = True,
                         use_threshold: bool = True, diff_threshold: float = 2e-7,
                         max_scale_log2: float | None = None) -> CleReport:
    """cle_transforms: match, then process_cle_transforms with the source's step loop."""
    if balance_method != "max":
        raise ValueError("The source implements only the max balance method")
    pairs = find_pairs(g)
    by_name = {n.name: n for n in g.nodes()}
    targets = [n for n in g.nodes() if n.op_type in TARGETS]
    scaled = {}
    skipped_unstable = []
    diff, count, converge_count, step_count = 10.0, 0, 20, 0
    while diff > diff_threshold and count < converge_count:
        if steps >= 0 and step_count >= steps:
            break
        previous = {n.input[1]: g.initializer(n.input[1]) for n in targets}
        for index, pair in enumerate(pairs):
            if pair.size == 4:
                key = f"{step_count}:{index}:{pair.head}->{pair.middle}->{pair.tail}"
                scaled[key] = equalize_triple(
                    g, by_name[pair.head], by_name[pair.middle], by_name[pair.tail],
                    max_scale_log2)
            else:
                # Pairs are never guarded. The threshold cannot separate benefit from harm
                # across both populations: ResNet50's beneficial pairs top out at 2.36 bits
                # while ten of ResNeXt-50's destructive triples sit between 2.2 and 3.4, so
                # any cut that disarms the triples would also disarm ResNet50's pairs.
                key = f"{step_count}:{index}:{pair.head}->{pair.tail}"
                scaled[key] = equalize_pair(
                    g, by_name[pair.head], by_name[pair.tail], weight_threshold, append_bias,
                    use_threshold, None)
            if scaled[key].get("skipped") == "unstable scale":
                skipped_unstable.append({"pair": key, **{k: v for k, v in scaled[key].items()
                                                         if k != "skipped"}})
        diff_tmp = 0.0
        for node in targets:
            diff_tmp += float(np.mean(np.abs(np.float64(previous[node.input[1]] - g.initializer(node.input[1])))))
        if abs(diff - diff_tmp) > 1e-9:
            count = 0
            diff = diff_tmp
        else:
            count += 1
        step_count += 1
    return CleReport(pattern_count=len(pairs), unique_pairs=len({(p.head, p.tail) for p in pairs}),
                     pairs=[asdict(p) for p in pairs], steps=step_count, last_diff=diff,
                     weight_threshold=weight_threshold, append_bias=append_bias,
                     use_threshold=use_threshold, scaled=scaled,
                     max_scale_log2=max_scale_log2, skipped_unstable=skipped_unstable)

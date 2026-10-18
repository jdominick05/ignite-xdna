# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""C5b, the independent bias check (v2.5 section C): is an int32-bias arm's emitted bias the section 3.2 BC step?

results/int4/w4v_audit/w4v_bc_audit.py (sha256 dbc4b7bb..., the screen1 STOP audit the gate accepted) with the arm
and its base as parameters. Its static (b)-(d), SEQ, ANCHORED and self-check code stay line for line; the pins come
from tools/w4v_verdict.py's pins(), and the tensors mode is dropped. It never imports tools/w4_recover.py or quant/;
the IMPORTS line of every mode says so, and a recompute ends with C5B <model> <arm> PASS|FAIL: PASS iff its HELD
line reads GENUINE, the BASE check holds (none on VC) and no such import was loaded.

    w4v_bias_audit.py recompute <model> <arm>         (a)-(d) on one arm (bc, vc, arbc or gptqbc, any form)
    w4v_bias_audit.py static <model> <arm>...         (b)-(d) only, no inference

The SEQ walk's base, per method (v2.5 section C):
 - BC: the pinned rtn_<form> file and its biases (the shipped int8 ones); BASE: the arm's non-bias initializers equal
   the base's.
 - AdaRound + BC, GPTQ + BC: the pinned tier-1 ar_<form> / gptq_<form> file (tools/w4_recover.py's emit base) and its
   biases; BASE as for BC.
 - VC: the arm's own weights (rescaled by design), starting from the shipped biases read from the XINT8 file; no BASE
   check. On VC this checks only the BC step on the rescaled weights, not the rescale or the re-quantization.

The gate's criteria, fixed before any run. An arm is a DEFECT if any of these holds:
 (a) in any Conv, the emitted int32 bias dequantized at its stored scale differs from an independent recomputation
     of section 3.2 by more than 1 LSB (of s_x * s_w for that channel) in any channel;
 (b) for any Conv, the int32 bias's scale is not that Conv's s_x * s_w as read from the QDQ graph (the input DQ's
     scale times the weight DQ's scale, per channel in form a);
 (c) any bias channel is clamped or saturated short of make_bias's overflow refusal;
 (d) any int32 bias is wired to the wrong Conv, has the wrong length, or carries a nonzero zero point or a scale
     other than (b)'s.

Every file is checked against its pin (tools/w4v_verdict.py pins(): the committed, clean prereg and pins logs) before
it is read. Convs are matched across files by name. "XINT8 order" is the shipped models/<model>_cut_xint8.onnx's Conv
node order; the emitted arm files list their nodes in another order (the same set), printed as INFO.

The recomputation for (a), from the text:
 - calibration: the first 64 of data/coco_calib by sorted name, checked against the pinned listing sha256; each is
   cv2.imread, then npu.yolo.letterbox at the graph's input size (what quant.sources.CocoSource and the eval do);
 - y_f: the float head-cut model's Conv output, on the float network's own input;
 - y_q: the QDQ model's Conv output (the Conv node's own float output, before its output QuantizeLinear) with the
   current weights and bias, on its sequential inputs;
 - every session: ORT_DISABLE_ALL, 8 intra-op threads, the CPU EP;
 - per Conv in XINT8 order, per output channel: b'_c = b_c + mean(y_f,c) - mean(y_q,c), each mean over every
   output position of the 64 images, accumulated in float64;
 - b' is stored as int32 at s_x * s_w (rint) and dequantized in float32 as ORT's DequantizeLinear does; that value
   is the Conv's current bias for every later Conv.
Two walks, both printed per Conv. (a) is judged on SEQ; if ANCHORED disagrees, that goes to the gate as it stands.
 - SEQ, the independent walk: from the base file (above), every earlier Conv holding this script's own int32
   bias. The Conv biases are made graph inputs (each bias DQ dropped), so one session serves
   every step; a step fetches only its own Conv's output (only_execute_path_to_fetches, checked equal to a full run).
 - ANCHORED: every earlier Conv holding the arm file's own bias, which is exactly the file's sequential input, so a
   rounding flip upstream cannot move a later Conv. The Conv adds its bias per channel after the accumulation, so
   y_q(b_c) = y_arm - b_arm,c + b_c, and b'_c = b_arm,c + mean(y_f,c) - mean(y_arm,c): one pass over the arm file.
Self-checks before the walk, bit for bit on 2 calibration images: the bias-input model fed the base file's own
biases gives the base's every Conv output; fed the arm's biases, it gives the arm's every Conv output (on VC the base
is the arm, so the two coincide).
(c) with the recomputation: a channel whose |int32| sits on a bound value (2^k - 1 or 2^k, k = 7, 15, 23, 31) while
SEQ's |rint(b'/scale)| lies beyond it by more than 1. Statically: a channel at an int32 extreme. Channels on the
smaller bound values are counted as INFO, since a static read cannot tell a clamp from a genuine value there.
"""
import copy
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from npu.yolo import letterbox  # noqa: E402  the eval's and CocoSource's letterbox
import w4v_verdict as vv  # noqa: E402  stdlib only: the pinned calibration constants and pins()

THREADS = 8
BOUNDS = [2 ** k - 1 for k in (7, 15, 23, 31)] + [2 ** k for k in (7, 15, 23, 31)]
SMALL_BOUNDS = [b for b in BOUNDS if b < 2 ** 31 - 1]
BASE_METHOD = {"bc": "rtn", "arbc": "ar", "gptqbc": "gptq", "vc": None}   # the SEQ walk's base (v2.5 section C)
LABEL = {"rtn": "RTN", "ar": "AdaRound", "gptq": "GPTQ"}


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_lf(path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


PINS = vv.pins()   # basename -> sha256, from the committed, clean prereg and pins logs


def pinned(candidates: list) -> str:
    hits = [p for p in candidates if Path(p).name in PINS]
    if len(hits) != 1:
        raise SystemExit(f"{len(hits)} pinned paths among {candidates}")
    got = sha(ROOT / hits[0])
    ok = got == PINS[Path(hits[0]).name]
    print(f"PIN_CHECK {hits[0]} sha256 {got} {'== pinned' if ok else 'MISMATCH'}", flush=True)
    if not ok:
        raise SystemExit("a file differs from its pin")
    return hits[0]


def arm_file(model: str, arm: str) -> str:
    return pinned([f"scratch/w4v/arms/{model}_w4v_{arm}.onnx", f"scratch/w4v/arms/{model}_w4v_{arm}_ortonly.onnx"])


def model_file(model: str, xint8: bool) -> str:
    return pinned([f"models/{model}_cut{'_xint8' if xint8 else ''}.onnx"])


def base_of(arm: str) -> str | None:
    """The SEQ walk's base arm (rtn_<form>, ar_<form> or gptq_<form>), or None on VC (the arm's own weights)."""
    method, form = arm.split("_")
    if method not in BASE_METHOD:
        raise SystemExit(f"{arm}: not an int32-bias arm ({', '.join(BASE_METHOD)})")
    return f"{BASE_METHOD[method]}_{form}" if BASE_METHOD[method] else None


def label(arm: str) -> str:
    method, form = arm.split("_")
    return f"{LABEL[method]} {vv.FORM_TAG[form]}"


def attr(node, name, default):
    for a in node.attribute:
        if a.name == name:
            return helper.get_attribute_value(a)
    return default


class G:
    """One ONNX file, indexed; its Convs by name."""

    def __init__(self, path: str):
        self.path = path
        self.m = onnx.load(str(ROOT / path))
        self.init = {t.name: t for t in self.m.graph.initializer}
        self.prod = {o: n for n in self.m.graph.node for o in n.output}
        self.cons = {}
        for n in self.m.graph.node:
            for i in n.input:
                self.cons.setdefault(i, []).append(n)
        self.convs = [n for n in self.m.graph.node if n.op_type == "Conv"]
        self.conv = {c.name: c for c in self.convs}
        if len(self.conv) != len(self.convs):
            raise SystemExit(f"{path}: two Convs share a name")
        inputs = [i for i in self.m.graph.input if i.name not in self.init]
        if len(inputs) != 1:
            raise SystemExit(f"{path}: {len(inputs)} graph inputs")
        self.input = inputs[0].name
        self.imgsz = inputs[0].type.tensor_type.shape.dim[2].dim_value

    def a(self, name: str) -> np.ndarray:
        return numpy_helper.to_array(self.init[name])

    def dq(self, conv, j: int):
        p = self.prod.get(conv.input[j])
        if p is None or p.op_type != "DequantizeLinear":
            raise SystemExit(f"{self.path}: {conv.name} input {j} is not a DequantizeLinear output")
        return p

    def cout(self, conv) -> int:
        return int(self.init[self.dq(conv, 1).input[0]].dims[0])

    def out(self, name: str) -> str:
        return self.conv[name].output[0]

    def bias_float(self, conv) -> np.ndarray:
        """The Conv's bias as ORT's DequantizeLinear gives it: float32(q - zp) * scale, in float32."""
        d = self.dq(conv, 2)
        q, s = self.a(d.input[0]), self.a(d.input[1])
        zp = self.a(d.input[2]) if len(d.input) > 2 and d.input[2] else np.zeros((), q.dtype)
        v = (q.astype(np.int64) - zp.astype(np.int64)).astype(np.float32) * s.astype(np.float32)
        return np.broadcast_to(v.astype(np.float32), (self.cout(conv),)).copy()

    def lsb(self, conv) -> np.ndarray:
        """s_x * s_w from the graph, float32, per output channel."""
        sx, sw = self.a(self.dq(conv, 0).input[1]), self.a(self.dq(conv, 1).input[1])
        if sx.shape != ():
            raise SystemExit(f"{conv.name}: the input scale is not a scalar")
        v = (np.float32(sx) * sw.astype(np.float32)).astype(np.float32)
        return np.broadcast_to(v, (self.cout(conv),)).copy()


def same_convs(order: list, *files) -> None:
    for g in files:
        if sorted(g.conv) != sorted(order):
            raise SystemExit(f"{g.path}: its Conv names differ from the shipped file's")
        print(f"INFO {g.path}: the shipped Conv names; node order == XINT8 order "
              f"{[c.name for c in g.convs] == order}", flush=True)


def with_outputs(model: onnx.ModelProto, outs: list) -> onnx.ModelProto:
    have = {o.name for o in model.graph.output}
    for name, elem in outs:
        if name not in have:
            model.graph.output.append(helper.make_tensor_value_info(name, elem, None))
            have.add(name)
    return model


def session(model: onnx.ModelProto):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.intra_op_num_threads = THREADS
    so.log_severity_level = 3
    return ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])


def load(path: Path, imgsz: int) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"cv2 could not read {path.name}")
    return letterbox(img, imgsz)[0]


def calib_images(imgsz: int) -> list:
    d = ROOT / vv.CALIB["dir"]
    names = sorted(p.name for p in d.glob("*.jpg"))[: vv.CALIB["n"]]
    got = hashlib.sha256("".join(f"{n} {sha(d / n)}\n" for n in names).encode()).hexdigest()
    ok = (len(names) == vv.CALIB["n"] and names[0] == vv.CALIB["first"] and names[-1] == vv.CALIB["last"]
          and got == vv.CALIB["listing_sha256"])
    print(f"CALIB {len(names)} images {names[0]} .. {names[-1]}, listing sha256 {got} "
          f"{'== pinned' if ok else 'MISMATCH'}; letterbox {imgsz}", flush=True)
    if not ok:
        raise SystemExit("the calibration listing differs from its pin")
    return [load(d / n, imgsz) for n in names]


def conv_means(g: G, order: list, images: list, extra: list = ()) -> tuple:
    """Per Conv (by name): the per-channel mean of its output over every position of `images`, in float64. `extra`
    tensors are fetched too and handed back per image to the caller's counter."""
    sess = session(with_outputs(copy.deepcopy(g.m), [(g.out(n), TensorProto.FLOAT) for n in order] + list(extra)))
    fetch = [g.out(n) for n in order] + [name for name, _ in extra]
    sums = {n: 0.0 for n in order}
    count = {n: 0 for n in order}
    extras = []
    for x in images:
        ys = sess.run(fetch, {g.input: x})
        for n, y in zip(order, ys):
            sums[n] = sums[n] + np.sum(y, axis=(0, 2, 3), dtype=np.float64)
            count[n] += y.shape[0] * y.shape[2] * y.shape[3]
        extras.append(ys[len(order):])
    return {n: sums[n] / count[n] for n in order}, extras, sess


def bias_input_model(g: G):
    """g with every Conv bias a float graph input and every Conv output a graph output; {conv name: input name}."""
    m = copy.deepcopy(g.m)
    prod = {o: n for n in m.graph.node for o in n.output}
    cons = {}
    for n in m.graph.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    names, drop = {}, []
    for k, c in enumerate([n for n in m.graph.node if n.op_type == "Conv"]):
        d = prod[c.input[2]]
        if d.op_type != "DequantizeLinear" or len(cons[d.output[0]]) != 1:
            raise SystemExit(f"{c.name}: its bias is not a DequantizeLinear read by this Conv alone")
        names[c.name] = f"audit_bias_{k}"
        c.input[2] = names[c.name]
        m.graph.input.append(helper.make_tensor_value_info(names[c.name], TensorProto.FLOAT,
                                                           [int(g.init[prod[c.input[1]].input[0]].dims[0])]))
        drop.append(d)
    for d in drop:
        m.graph.node.remove(d)
    used = {i for n in m.graph.node for i in n.input}
    keep = [t for t in m.graph.initializer if t.name in used]
    del m.graph.initializer[:]
    m.graph.initializer.extend(keep)
    return with_outputs(m, [(c.output[0], TensorProto.FLOAT) for c in g.convs]), names


def run_options():
    ro = ort.RunOptions()
    try:
        ro.only_execute_path_to_fetches = True
        return ro, True
    except AttributeError:
        return None, False


# ------------------------------------------------------------------------------------------------ (b), (c), (d)

def static(model: str, arm: str, ship: G, g: G) -> dict:
    order = [c.name for c in ship.convs]
    same_convs(order, g)
    n = len(order)
    b_bad, d_bad, n_i32, extremes, on_bound, max_abs = [], [], 0, 0, 0, 0
    users = {}
    for node in g.m.graph.node:
        for i in node.input:
            users[i] = users.get(i, 0) + 1
    for name in order:
        c, cs = g.conv[name], ship.conv[name]
        db, ds = g.dq(c, 2), ship.dq(cs, 2)
        q, s = g.a(db.input[0]), g.a(db.input[1])
        zp = g.a(db.input[2]) if len(db.input) > 2 and db.input[2] else None
        want = (np.float32(g.a(g.dq(c, 0).input[1])) * g.a(g.dq(c, 1).input[1]).astype(np.float32)).astype(np.float32)
        n_i32 += q.dtype == np.int32
        if not (s.dtype == np.float32 and s.shape == want.shape and np.array_equal(s, want)):
            b_bad.append(name)
        why = []
        if list(db.input[:3]) != [x + "_w4v" for x in ds.input[:3]]:
            why.append("not the shipped bias's initializers + _w4v")
        if c.input[2] != cs.input[2]:
            why.append("the Conv reads another bias tensor than shipped")
        readers = g.cons.get(db.output[0], [])
        if len(readers) != 1 or readers[0] is not c or list(c.input).index(db.output[0]) != 2:
            why.append("the bias DQ feeds another node or slot")
        if q.dtype != np.int32 or q.shape != (g.cout(c),) or g.cout(c) != ship.cout(cs):
            why.append(f"dtype {q.dtype} or length {q.shape} vs cout {g.cout(c)}")
        if zp is None or zp.dtype != np.int32 or zp.shape != s.shape or np.any(zp != 0):
            why.append("zero point not int32 zeros of the scale's shape")
        if not np.array_equal(s, want):
            why.append("scale not (b)'s")
        if s.ndim == 1 and attr(db, "axis", 1) not in (0, -1):
            why.append(f"per-channel scale with axis {attr(db, 'axis', 1)}")
        if any(users.get(x, 0) != 1 for x in db.input[:3]):
            why.append("an initializer shared with another node")
        if why:
            d_bad.append((name, why))
        v = q.astype(np.int64)
        max_abs = max(max_abs, int(np.abs(v).max()))
        extremes += int(np.sum((v == 2 ** 31 - 1) | (v == -2 ** 31)))
        on_bound += int(np.sum(np.isin(np.abs(v), SMALL_BOUNDS)))
    print(f"STATIC {model} {arm} ({g.path}): {n} Conv, {n_i32} int32 biases", flush=True)
    print(f"(b) {model} {arm}: bias scale == float32(s_x) * float32(s_w) from the graph on {n - len(b_bad)}/{n} Conv"
          + (f"; first failing {b_bad[:3]}" if b_bad else ""), flush=True)
    print(f"(c) {model} {arm} static: channels at an int32 extreme {extremes}; max |int32| {max_abs} (2^31 = "
          f"{2 ** 31}); INFO channels on a smaller bound value (2^k - 1 or 2^k, k = 7, 15, 23) {on_bound}", flush=True)
    print(f"(d) {model} {arm}: wiring, length, zero point and scale right on {n - len(d_bad)}/{n} Conv"
          + (f"; first failing {d_bad[:3]}" if d_bad else ""), flush=True)
    held = {"b": bool(b_bad), "c": extremes > 0, "d": bool(d_bad) or n_i32 != n}
    print(f"HELD {model} {arm} (b) {held['b']} (c, static) {held['c']} (d) {held['d']}", flush=True)
    return held


# ------------------------------------------------------------------------------------------------------- (a)

def recompute(model: str, arm: str) -> dict:
    t0 = time.time()
    base = base_of(arm)
    F, S = G(model_file(model, False)), G(model_file(model, True))
    A = G(arm_file(model, arm))
    R = G(arm_file(model, base)) if base else A
    order = [c.name for c in S.convs]
    same_convs(order, F, *([R] if base else []), A)
    held = static(model, arm, S, A)

    if base:
        bias_names_r = {i for c in R.convs for i in R.dq(c, 2).input}
        rest = [k for k in R.init if k not in bias_names_r]
        same = sum(1 for k in rest if k in A.init and A.init[k].data_type == R.init[k].data_type
                   and np.array_equal(R.a(k), A.a(k)))
        base_ok = same == len(rest)
        print(f"BASE {arm} holds {label(base)}'s {same}/{len(rest)} non-bias initializers unchanged", flush=True)
    else:
        base_ok = True
        print(f"BASE {arm}: none (VC rescales the weights by design; the walk starts from the shipped biases)",
              flush=True)

    X = calib_images(F.imgsz)
    mean_f, _, _ = conv_means(F, order, X)
    print(f"Y_F float Conv means on {len(X)} images ({time.time() - t0:.0f} s)", flush=True)
    mean_arm, _, arm_sess = conv_means(A, order, X)
    print(f"Y_ARM {arm} Conv means on {len(X)} images ({time.time() - t0:.0f} s)", flush=True)

    bm, bname = bias_input_model(R)
    bs = session(bm)
    b_rtn = {n: R.bias_float(R.conv[n]) for n in order}   # the base file's own biases (the self-check's)
    b_start = b_rtn if base else {n: S.bias_float(S.conv[n]) for n in order}   # the walk's start
    b_arm = {n: A.bias_float(A.conv[n]) for n in order}
    rtn_sess = session(with_outputs(copy.deepcopy(R.m), [(R.out(n), TensorProto.FLOAT) for n in order]))
    feeds = lambda x, b: {R.input: x, **{bname[n]: b[n] for n in order}}  # noqa: E731
    ok_r = ok_a = True
    for x in X[:2]:
        want_r = rtn_sess.run([R.out(n) for n in order], {R.input: x})
        want_a = arm_sess.run([A.out(n) for n in order], {A.input: x})
        got_r = bs.run([R.out(n) for n in order], feeds(x, b_rtn))
        got_a = bs.run([R.out(n) for n in order], feeds(x, b_arm))
        ok_r &= all(np.array_equal(u, v) for u, v in zip(want_r, got_r))
        ok_a &= all(np.array_equal(u, v) for u, v in zip(want_a, got_a))
    ro, has_ro = run_options()
    ok_p = True
    if has_ro:
        for n in (order[0], order[len(order) // 2], order[-1]):
            ok_p &= np.array_equal(bs.run([R.out(n)], feeds(X[0], b_rtn), ro)[0],
                                   bs.run([R.out(n)], feeds(X[0], b_rtn))[0])
    print(f"SELFCHECK bias-input model == {label(base) if base else arm} on every Conv output (2 images): {ok_r}; "
          f"fed {arm}'s biases == {arm}: {ok_a}; only_execute_path_to_fetches "
          f"{'available' if has_ro else 'NOT available'}, path-only == full run on 3 Conv: {ok_p}", flush=True)
    if not (ok_r and ok_a and ok_p):
        raise SystemExit("a self-check failed: the walk would not measure what it claims")

    cur = {n: b.copy() for n, b in b_start.items()}
    seq_bad, anc_bad, clamped, over = [], [], 0, []
    seq_max = anc_max = 0.0
    for k, name in enumerate(order):
        tk = time.time()
        c = R.conv[name]
        cout = R.cout(c)
        s, cnt = np.zeros(cout, np.float64), 0
        for x in X:
            y = bs.run([R.out(name)], feeds(x, cur), ro)[0]
            s += np.sum(y, axis=(0, 2, 3), dtype=np.float64)
            cnt += y.shape[0] * y.shape[2] * y.shape[3]
        b_seq = cur[name].astype(np.float64) + mean_f[name] - s / cnt
        lsb_r = R.lsb(c)
        q = np.rint(b_seq / lsb_r.astype(np.float64))
        if np.abs(q).max() >= 2 ** 31:
            over.append(name)
        cur[name] = (q.astype(np.float32) * lsb_r).astype(np.float32)

        b_anc = b_arm[name].astype(np.float64) + mean_f[name] - mean_arm[name]
        dq = A.dq(A.conv[name], 2)
        i32 = A.a(dq.input[0]).astype(np.int64)
        em = np.broadcast_to(i32.astype(np.float64) * A.a(dq.input[1]).astype(np.float64), (cout,))
        lsb = A.lsb(A.conv[name]).astype(np.float64)
        d_seq, d_anc = np.abs(em - b_seq) / lsb, np.abs(em - b_anc) / lsb
        seq_max, anc_max = max(seq_max, float(d_seq.max())), max(anc_max, float(d_anc.max()))
        if (d_seq > 1).any():
            seq_bad.append(k)
        if (d_anc > 1).any():
            anc_bad.append(k)
        cl = np.isin(np.abs(i32), BOUNDS) & (np.abs(q) > np.abs(i32) + 1) & (np.sign(q) == np.sign(i32))
        clamped += int(cl.sum())
        print(f"CONV {k:2d} {name} cout {cout} | SEQ max {d_seq.max():.3f} LSB, >1 LSB {int((d_seq > 1).sum())} "
              f"| ANCHORED max {d_anc.max():.3f} LSB, >1 LSB {int((d_anc > 1).sum())} | int32 == SEQ's "
              f"{int(np.sum(i32 == q))}/{cout} | {time.time() - tk:.1f} s", flush=True)

    a_held = bool(seq_bad)
    first = lambda ks: f"first {ks[0]} {order[ks[0]]}" if ks else "none"  # noqa: E731
    print(f"(a) {model} {arm} SEQ: Conv with a channel beyond 1 LSB {len(seq_bad)}/{len(order)} ({first(seq_bad)}); "
          f"max {seq_max:.3f} LSB", flush=True)
    print(f"(a) {model} {arm} ANCHORED: Conv with a channel beyond 1 LSB {len(anc_bad)}/{len(order)} "
          f"({first(anc_bad)}); max {anc_max:.3f} LSB", flush=True)
    print(f"(c) {model} {arm} with the recomputation: clamped channels {clamped}; SEQ int32 overflow at "
          f"{over or 'none'}", flush=True)
    held["c"] = held["c"] or clamped > 0
    genuine = not (a_held or any(held.values()))
    print(f"HELD {model} {arm} (a) {a_held} (ANCHORED agrees: {bool(anc_bad) == a_held}) (b) {held['b']} "
          f"(c) {held['c']} (d) {held['d']} -> {'DEFECT' if not genuine else 'GENUINE'} "
          f"({time.time() - t0:.0f} s)", flush=True)
    return {"genuine": genuine, "base_ok": base_ok}


def main() -> int:
    me = Path(__file__)
    print(f"AUDIT {' '.join(sys.argv[1:])} | {now()} | script LF sha256 {sha_lf(me)} | onnxruntime {ort.__version__} | "
          f"threads {THREADS} | ORT_DISABLE_ALL", flush=True)
    mode, model = sys.argv[1], sys.argv[2]
    res = None
    if mode == "recompute":
        if len(sys.argv) != 4:
            raise SystemExit("recompute takes exactly <model> <arm>")
        res = recompute(model, sys.argv[3])
    elif mode == "static":
        ship = G(model_file(model, True))
        for arm in sys.argv[3:]:
            static(model, arm, ship, G(arm_file(model, arm)))
    else:
        raise SystemExit(f"unknown mode {mode}")
    loaded = sorted(m for m in sys.modules if m == "w4_recover" or m == "quant" or m.startswith("quant."))
    print(f"IMPORTS w4_recover or quant loaded: {loaded or 'none'} | END {now()}", flush=True)
    if res is None:
        return 0
    ok = res["genuine"] and res["base_ok"] and not loaded
    print(f"C5B {model} {sys.argv[3]} {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

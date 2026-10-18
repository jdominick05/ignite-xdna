# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The screen1 STOP audit (the gate's ruling of 2026-09-26): is yolov8s bc_e1 a DEFECT or BC's genuine output?

Report-only, outside the runner, uncommitted. Written from the v2.4 text (section 3's common part, and 3.2) and the
gate's criteria. It never imports tools/w4_recover.py or quant/; the last line of every mode says so.

    w4v_bc_audit.py recompute <model>          (a)-(d) on <model> bc_e1
    w4v_bc_audit.py static <model> <arm>...    (b)-(d) only, no inference
    w4v_bc_audit.py tensors <model>            (i) and (ii) on the 16 C3 images

The gate's criteria, fixed before any run. An arm is a DEFECT if any of these holds:
 (a) in any Conv, the emitted int32 bias dequantized at its stored scale differs from an independent recomputation
     of section 3.2 by more than 1 LSB (of s_x * s_w for that channel) in any channel;
 (b) for any Conv, the int32 bias's scale is not that Conv's s_x * s_w as read from the QDQ graph (the input DQ's
     scale times the weight DQ's scale, per channel in form a);
 (c) any bias channel is clamped or saturated short of make_bias's overflow refusal;
 (d) any int32 bias is wired to the wrong Conv, has the wrong length, or carries a nonzero zero point or a scale
     other than (b)'s.

Every file is checked against its PIN line in the committed prereg log before it is read. Convs are matched across
files by name. "XINT8 order" is the shipped models/<model>_cut_xint8.onnx's Conv node order; the emitted arm files
list their nodes in another order (the same set), printed as INFO.

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
 - SEQ, the independent walk: from the pinned RTN E1 file (BC sits on RTN in tier 1), every earlier Conv holding
   this script's own int32 bias. The Conv biases are made graph inputs (each bias DQ dropped), so one session serves
   every step; a step fetches only its own Conv's output (only_execute_path_to_fetches, checked equal to a full run).
 - ANCHORED: every earlier Conv holding the arm file's own bias, which is exactly the file's sequential input, so a
   rounding flip upstream cannot move a later Conv. The Conv adds its bias per channel after the accumulation, so
   y_q(b_c) = y_arm - b_arm,c + b_c, and b'_c = b_arm,c + mean(y_f,c) - mean(y_arm,c): one pass over the arm file.
Self-checks before the walk, bit for bit on 2 calibration images: the bias-input model fed RTN E1's biases gives
RTN E1's every Conv output; fed the arm's biases, it gives the arm's every Conv output.
(c) with the recomputation: a channel whose |int32| sits on a bound value (2^k - 1 or 2^k, k = 7, 15, 23, 31) while
SEQ's |rint(b'/scale)| lies beyond it by more than 1. Statically: a channel at an int32 extreme. Channels on the
smaller bound values are counted as INFO, since a static read cannot tell a clamp from a genuine value there.

The tensor facts, on the 16 C3 images (C3's set: the first 16 of sorted data/coco/val2017/*.jpg), letterboxed as
the eval does (C3's own compare stretches them; the question here is the eval's):
 (i)  for bc_e1 and rtn_e1, the fraction of class-logit outputs (the QuantizeLinear after each
      /model.22/cv3.i/cv3.i.2/Conv) at that quantizer's minimum code;
 (ii) the first Conv, in XINT8 order, where bc_e1's per-channel |mean(y) - mean(y_f)| exceeds rtn_e1's in more than
      half of its channels (index and name only).
"""
import copy
import hashlib
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from npu.yolo import letterbox  # noqa: E402  the eval's and CocoSource's letterbox
import w4v_verdict as vv  # noqa: E402  stdlib only: the pinned calibration constants

PREREG_LOG = ROOT / "results" / "int4" / "w4v_prereg_desktop2_20260926.log"
THREADS = 8
BOUNDS = [2 ** k - 1 for k in (7, 15, 23, 31)] + [2 ** k for k in (7, 15, 23, 31)]
SMALL_BOUNDS = [b for b in BOUNDS if b < 2 ** 31 - 1]
CLASS_CONV = re.compile(r"^/model\.22/cv3\.(\d)/cv3\.\1\.2/Conv$")


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_pins() -> dict:
    out = {}
    for line in PREREG_LOG.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^PIN ([0-9a-f]{64})  (\S+)$", line)
        if m:
            out[m.group(2)] = m.group(1)
    return out


PINS = read_pins()


def pinned(pattern: str) -> str:
    hits = [p for p in PINS if re.fullmatch(pattern, p)]
    if len(hits) != 1:
        raise SystemExit(f"{len(hits)} pinned paths match {pattern}")
    got = sha(ROOT / hits[0])
    ok = got == PINS[hits[0]]
    print(f"PIN_CHECK {hits[0]} sha256 {got} {'== pinned' if ok else 'MISMATCH'}", flush=True)
    if not ok:
        raise SystemExit("a file differs from its pin")
    return hits[0]


def arm_file(model: str, arm: str) -> str:
    return pinned(rf"scratch/w4v/arms/{model}_w4v_{arm}(_ortonly)?\.onnx")


def model_file(model: str, xint8: bool) -> str:
    return pinned(rf"models/{model}_cut{'_xint8' if xint8 else ''}\.onnx")


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


def c3_images(imgsz: int) -> list:
    files = sorted((ROOT / "data" / "coco" / "val2017").glob("*.jpg"))[:16]
    listing = hashlib.sha256("".join(f"{p.name} {sha(p)}\n" for p in files).encode()).hexdigest()
    print(f"C3_IMAGES {len(files)} images {files[0].name} .. {files[-1].name}, listing sha256 {listing}; "
          f"letterbox {imgsz}", flush=True)
    if len(files) != 16:
        raise SystemExit("fewer than 16 val2017 images")
    return [load(p, imgsz) for p in files]


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

def recompute(model: str) -> None:
    t0 = time.time()
    F, S = G(model_file(model, False)), G(model_file(model, True))
    R, A = G(arm_file(model, "rtn_e1")), G(arm_file(model, "bc_e1"))
    order = [c.name for c in S.convs]
    same_convs(order, F, R, A)
    held = static(model, "bc_e1", S, A)

    bias_names_r = {i for c in R.convs for i in R.dq(c, 2).input}
    rest = [k for k in R.init if k not in bias_names_r]
    same = sum(1 for k in rest if k in A.init and A.init[k].data_type == R.init[k].data_type
               and np.array_equal(R.a(k), A.a(k)))
    print(f"BASE bc_e1 holds RTN E1's {same}/{len(rest)} non-bias initializers unchanged", flush=True)

    X = calib_images(F.imgsz)
    mean_f, _, _ = conv_means(F, order, X)
    print(f"Y_F float Conv means on {len(X)} images ({time.time() - t0:.0f} s)", flush=True)
    mean_arm, _, arm_sess = conv_means(A, order, X)
    print(f"Y_ARM bc_e1 Conv means on {len(X)} images ({time.time() - t0:.0f} s)", flush=True)

    bm, bname = bias_input_model(R)
    bs = session(bm)
    b_rtn = {n: R.bias_float(R.conv[n]) for n in order}
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
    print(f"SELFCHECK bias-input model == RTN E1 on every Conv output (2 images): {ok_r}; fed bc_e1's biases == "
          f"bc_e1: {ok_a}; only_execute_path_to_fetches {'available' if has_ro else 'NOT available'}, "
          f"path-only == full run on 3 Conv: {ok_p}", flush=True)
    if not (ok_r and ok_a and ok_p):
        raise SystemExit("a self-check failed: the walk would not measure what it claims")

    cur = {n: b.copy() for n, b in b_rtn.items()}
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
    print(f"(a) {model} bc_e1 SEQ: Conv with a channel beyond 1 LSB {len(seq_bad)}/{len(order)} ({first(seq_bad)}); "
          f"max {seq_max:.3f} LSB", flush=True)
    print(f"(a) {model} bc_e1 ANCHORED: Conv with a channel beyond 1 LSB {len(anc_bad)}/{len(order)} "
          f"({first(anc_bad)}); max {anc_max:.3f} LSB", flush=True)
    print(f"(c) {model} bc_e1 with the recomputation: clamped channels {clamped}; SEQ int32 overflow at "
          f"{over or 'none'}", flush=True)
    held["c"] = held["c"] or clamped > 0
    print(f"HELD {model} bc_e1 (a) {a_held} (ANCHORED agrees: {bool(anc_bad) == a_held}) (b) {held['b']} "
          f"(c) {held['c']} (d) {held['d']} -> {'DEFECT' if a_held or any(held.values()) else 'GENUINE'} "
          f"({time.time() - t0:.0f} s)", flush=True)


# ------------------------------------------------------------------------------------------------ (i), (ii)

def tensors(model: str) -> None:
    t0 = time.time()
    F, S = G(model_file(model, False)), G(model_file(model, True))
    R, A = G(arm_file(model, "rtn_e1")), G(arm_file(model, "bc_e1"))
    order = [c.name for c in S.convs]
    same_convs(order, F, R, A)
    X = c3_images(F.imgsz)
    mean_f, _, _ = conv_means(F, order, X)
    means = {}
    for tag, g in (("rtn_e1", R), ("bc_e1", A)):
        qs = []
        for name in order:
            if CLASS_CONV.match(name):
                (qn,) = [n for n in g.cons[g.out(name)] if n.op_type == "QuantizeLinear"]
                zp = g.a(qn.input[2])
                qs.append((qn.output[0], zp.dtype, int(np.iinfo(zp.dtype).min)))
        if len(qs) != 3:
            raise SystemExit(f"{tag}: {len(qs)} class-logit convs")
        means[tag], extras, _ = conv_means(g, order, X, [(o, helper.np_dtype_to_tensor_dtype(dt)) for o, dt, _ in qs])
        at_min = [sum(int(np.sum(e[j] == qs[j][2])) for e in extras) for j in range(3)]
        total = [sum(int(e[j].size) for e in extras) for j in range(3)]
        print(f"(i) {model} {tag}: class-logit outputs at the quantizer's minimum code ({qs[0][1].name} {qs[0][2]}): "
              f"{sum(at_min)}/{sum(total)} = {sum(at_min) / sum(total):.6f}; per scale "
              f"{', '.join(f'{a / t:.6f}' for a, t in zip(at_min, total))}", flush=True)
    first = None
    for k, name in enumerate(order):
        e_bc = np.abs(means["bc_e1"][name] - mean_f[name])
        e_rtn = np.abs(means["rtn_e1"][name] - mean_f[name])
        if int(np.sum(e_bc > e_rtn)) > len(e_bc) / 2:
            first = (k, name)
            break
    print(f"(ii) {model}: first Conv (XINT8 order) where bc_e1's per-channel mean error exceeds rtn_e1's in more than "
          f"half its channels: {f'{first[0]} {first[1]}' if first else 'none'} ({time.time() - t0:.0f} s)", flush=True)


def main() -> int:
    me = Path(__file__)
    print(f"AUDIT {' '.join(sys.argv[1:])} | {now()} | script sha256 {sha(me)} | onnxruntime {ort.__version__} | "
          f"threads {THREADS} | ORT_DISABLE_ALL", flush=True)
    mode, model = sys.argv[1], sys.argv[2]
    if mode == "recompute":
        recompute(model)
    elif mode == "static":
        ship = G(model_file(model, True))
        for arm in sys.argv[3:]:
            static(model, arm, ship, G(arm_file(model, arm)))
    elif mode == "tensors":
        tensors(model)
    else:
        raise SystemExit(f"unknown mode {mode}")
    loaded = sorted(m for m in sys.modules if m == "w4_recover" or m == "quant" or m.startswith("quant."))
    print(f"IMPORTS w4_recover or quant loaded: {loaded or 'none'} | END {now()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

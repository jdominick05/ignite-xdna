#!/usr/bin/env python3
"""Checks and times the native int8 head decode (pipelines/decode_native.c) against YoloDecoder's numpy path.

    python tools/decode_native_check.py stress --trials 2000                  # offline, no device
    python tools/decode_native_check.py record OUT_DIR --stills data/user_calib --camera-frames 120   # NPU
    python tools/decode_native_check.py replay OUT_DIR                        # offline
    python tools/decode_native_check.py live --stills data/user_calib --camera-frames 240             # NPU
    python tools/decode_native_check.py slowdown [--pipeline]                 # host (NPU with --pipeline)

stress    random int8 heads built to stress the decode: same-class clusters within and across heads (overlaps
          around the IoU threshold), score ties from shared logits, argmax ties between classes, saturated and
          uniform DFL bins, nonzero zero points, pads that push boxes negative, many candidates (the heap path),
          and a confidence threshold equal to a reachable score; half the trials pass per-anchor class maxima.
          Every trial must give identical detections (YoloDetection equality, exact floats).
record    runs the graph-engine container on still images and live camera frames and saves each frame's int8
          heads, class maxima, scales, pad and scale as .npz.
replay    decodes recorded heads with both paths: identical detections, and hot-loop medians of each.
live      predict_sync(use_oracle_for_boxes=False) per frame with the decode path alternating by frame, so
          postprocess_ms is the live decode of that path; the other path then decodes the same heads and the
          detections must be identical.
slowdown  zlib.crc32 over private buffers of ~0.010 and ~0.090 ms in a tight loop, then the same work after
          sleep or spin waits, and with --pipeline after an NPU dispatch awaited by run.wait() or by polling.
Exit status 0 only when every parity check passes. NPU modes print xrt-smi's partition state before and after.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
import zlib
from pathlib import Path
from statistics import mean, median

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

HEADS = ("p3_box", "p4_box", "p5_box", "p3_cls", "p4_cls", "p5_cls")
CLS_NAMES = ("p3_cls", "p4_cls", "p5_cls")
GRIDS = (("p3", 80), ("p4", 40), ("p5", 20))
XRT_SMI = Path(r"C:\Windows\System32\AMD\xrt-smi.exe")
pc = time.perf_counter


def _decoder_modules():
    import ignite_xdna
    from ignite_xdna.pipelines import decode_native
    from ignite_xdna.pipelines.yolo_pipeline import YoloDecoder
    src = (ROOT / "src").resolve()
    if not Path(ignite_xdna.__file__).resolve().is_relative_to(src):
        raise SystemExit(f"ignite_xdna was imported from {ignite_xdna.__file__}, not from {src}")
    return decode_native, YoloDecoder


def _header(mode: str) -> None:
    import cv2
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    print(f"decode_native_check {mode}: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} host {platform.node()} "
          f"commit {commit} python {sys.version.split()[0]} numpy {np.__version__} cv2 {cv2.__version__}", flush=True)


def _xrt_smi(label: str) -> None:
    if not XRT_SMI.exists():
        print(f"xrt-smi {label}: not found", flush=True)
        return
    out = subprocess.run([str(XRT_SMI), "examine", "-r", "aie-partitions"], capture_output=True, text=True,
                         timeout=60).stdout.strip().splitlines()
    print(f"xrt-smi {label}: {out[-1].strip() if out else '(no output)'}", flush=True)


def _pct(xs, q):
    return float(np.percentile(xs, q)) if xs else float("nan")


# ---------------------------------------------------------------------------------------------- stress
def _random_heads(rng: np.random.Generator):
    """int8 heads with dense same-class clusters and ties, plus their scales."""
    heads, scales = {}, {}
    for p, g in GRIDS:
        sb = float(2.0 ** int(rng.integers(-5, 1))) if rng.random() < 0.6 else float(np.float32(rng.uniform(0.02, 1.0)))
        sc = float(2.0 ** int(rng.integers(-4, 0))) if rng.random() < 0.6 else float(np.float32(rng.uniform(0.0625, 0.5)))
        zb = int(rng.integers(-20, 21)) if rng.random() < 0.3 else 0
        zc = int(rng.integers(-20, 21)) if rng.random() < 0.3 else 0
        scales[p + "_box"], scales[p + "_cls"] = (sb, zb), (sc, zc)
        # Background logits sit below any threshold the trials use (sigmoid < 0.01).
        heads[p + "_cls"] = rng.integers(-128, -111, (1, 80, g, g), dtype=np.int8)
        heads[p + "_box"] = rng.integers(-128, 128, (1, 64, g, g), dtype=np.int8)
    many = rng.random() < 0.15
    for _ in range(int(rng.integers(0, 60 if many else 25))):
        h = int(rng.integers(0, 3))
        p, g = GRIDS[h]
        cy, cx = int(rng.integers(0, g)), int(rng.integers(0, g))
        cls_id = int(rng.integers(0, 80)) if rng.random() < 0.5 else int(rng.integers(0, 3))
        q = 127 if rng.random() < 0.2 else int(rng.integers(-20, 128))
        radius = int(rng.integers(0, 4 if many else 3))
        pattern = rng.integers(-128, 128, 64).astype(np.int16)
        kind = rng.random()
        if kind < 0.1:
            pattern[:] = int(rng.integers(-128, 128))              # uniform bins
        elif kind < 0.2:
            pattern[:] = -128
            pattern[np.arange(4) * 16 + rng.integers(0, 16, 4)] = 127  # one saturated bin per side
        targets = [(h, cy, cx)]
        if h < 2 and rng.random() < 0.5:                             # the same object on the next coarser head
            targets.append((h + 1, cy // 2, cx // 2))
        for th, ty, tx in targets:
            tp, tg = GRIDS[th]
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    y, x = ty + dy, tx + dx
                    if not (0 <= y < tg and 0 <= x < tg):
                        continue
                    qq = q if rng.random() < 0.6 else int(np.clip(q - rng.integers(0, 4), -128, 127))
                    heads[tp + "_cls"][0, cls_id, y, x] = qq
                    if rng.random() < 0.1:                           # an argmax tie with another class
                        heads[tp + "_cls"][0, (cls_id + 1) % 80, y, x] = qq
                    if rng.random() < 0.15:  # a second class just below, often saturating to the same float32
                        other = (cls_id + int(rng.choice([-1, 1]))) % 80
                        heads[tp + "_cls"][0, other, y, x] = int(np.clip(qq - rng.integers(1, 7), -128, 127))
                    noise = rng.integers(-3, 4, 64) if rng.random() < 0.7 else 0
                    heads[tp + "_box"][0, :, y, x] = np.clip(pattern + noise, -128, 127).astype(np.int8)
    return heads, scales


def cmd_stress(args) -> int:
    decode_native, YoloDecoder = _decoder_modules()
    _header("stress")
    if not decode_native.available():
        print("FAIL: the native decode library is not available", flush=True)
        return 1
    rng = np.random.default_rng(args.seed)
    numpy_dec = YoloDecoder(native_decode=False)
    native_dec = YoloDecoder()
    assert native_dec.uses_native_decode and not numpy_dec.uses_native_decode
    failures = detections = empty = heap = at_conf = with_max = 0
    t_numpy = t_native = 0.0
    for t in range(args.trials):
        heads, scales = _random_heads(rng)
        heads["scales"] = scales
        if rng.random() < 0.5:
            heads["cls_max"] = {n: heads[n].reshape(80, -1).max(0) for n in CLS_NAMES}
            with_max += 1
        pad = (int(rng.integers(0, 240)), int(rng.integers(0, 240)))
        scale = float(np.float32(rng.uniform(0.2, 3.0)))
        r = rng.random()
        if r < 0.3:
            conf, iou = 0.25, 0.5
        elif r < 0.45:
            # A threshold equal to a score some anchor can reach: >= keeps it for max_coord, NMS's > drops it.
            name = CLS_NAMES[int(rng.integers(0, 3))]
            s, zp = scales[name]
            q = int(rng.integers(-20, 128))
            conf = float(decode_native.sigmoid_table(s, zp)[q + 128])
            iou = 0.5
            at_conf += 1
        else:
            conf = float(rng.choice([0.01, 0.1, 0.25, 0.3, 0.45, 0.6, 0.9]))
            iou = float(rng.choice([0.0, 0.3, 0.45, 0.5, 0.7, 1.0, float(rng.uniform(0, 1))]))
        t0 = pc()
        ref = numpy_dec.postprocess(heads, pad, scale, conf, iou)
        t1 = pc()
        got = native_dec.postprocess(heads, pad, scale, conf, iou)
        t2 = pc()
        t_numpy += t1 - t0
        t_native += t2 - t1
        detections += len(ref)
        empty += not ref
        if got != ref:
            failures += 1
            if failures <= 5:
                print(f"trial {t}: MISMATCH conf={conf!r} iou={iou!r} pad={pad} scale={scale!r}: numpy {len(ref)} "
                      f"native {len(got)}", flush=True)
                for a, b in zip(ref, got):
                    if a != b:
                        print(f"   numpy  {a}\n   native {b}", flush=True)
                        break
        survivors = 0
        for n in CLS_NAMES:
            s, zp = scales[n]
            logit_t = np.log(min(max(conf, 1e-12), 1 - 1e-12) / (1 - min(max(conf, 1e-12), 1 - 1e-12)))
            survivors += int(np.count_nonzero(heads[n].reshape(80, -1).max(0) > logit_t / s + zp))
        heap += survivors > 256
    print(f"stress: {args.trials} trials (seed {args.seed}), {with_max} with class maxima, {at_conf} with the "
          f"threshold at a reachable score, {heap} with more than 256 candidates; {detections} detections, "
          f"{empty} empty results; mismatches {failures}", flush=True)
    print(f"stress wall time: numpy {t_numpy:.2f} s, native {t_native:.2f} s", flush=True)
    print("RESULT:", "PASS" if failures == 0 else "FAIL", flush=True)
    return 0 if failures == 0 else 1


# ---------------------------------------------------------------------------------------------- recorded heads
def _load_npz(path: Path):
    z = np.load(path)
    heads = {k: z[k] for k in HEADS}
    maxima = {k[len("cls_max_"):]: z[k] for k in z.files if k.startswith("cls_max_")}
    if maxima:
        heads["cls_max"] = maxima
    heads["scales"] = {k: (float(v[0]), int(v[1])) for k, v in json.loads(str(z["scales"])).items()}
    return heads, tuple(int(x) for x in z["pad"]), float(z["scale"])


def _group(name: str) -> str:
    return "camera" if name.startswith("camera_") else ("bus" if name == "bus" else "stills")


def cmd_replay(args) -> int:
    decode_native, YoloDecoder = _decoder_modules()
    _header("replay")
    if not decode_native.available():
        print("FAIL: the native decode library is not available", flush=True)
        return 1
    numpy_dec, native_dec = YoloDecoder(native_decode=False), YoloDecoder()
    files = sorted(Path(args.heads_dir).glob("*.npz"))
    if not files:
        print(f"FAIL: no .npz files in {args.heads_dir}", flush=True)
        return 1
    rows, failures = {}, 0
    for f in files:
        heads, pad, scale = _load_npz(f)
        ref = numpy_dec.postprocess(heads, pad, scale)
        got = native_dec.postprocess(heads, pad, scale)
        if got != ref:
            failures += 1
            print(f"{f.name}: MISMATCH numpy {len(ref)} native {len(got)}", flush=True)
        times = {}
        for label, dec in (("numpy", numpy_dec), ("native", native_dec)):
            for _ in range(20):
                dec.postprocess(heads, pad, scale)
            ts = []
            for _ in range(args.repeats):
                t0 = pc()
                dec.postprocess(heads, pad, scale)
                ts.append(pc() - t0)
            times[label] = median(ts) * 1e3
        rows.setdefault(_group(f.stem), []).append((len(ref), times["numpy"], times["native"]))
    for grp, rs in rows.items():
        print(f"replay {grp}: {len(rs)} frames, detections/frame {mean(r[0] for r in rs):.2f} | hot medians of "
              f"per-frame medians ({args.repeats} calls): numpy {median(r[1] for r in rs):.4f} ms, native "
              f"{median(r[2] for r in rs):.4f} ms", flush=True)
    print(f"replay: {len(files)} frames, mismatches {failures}", flush=True)
    print("RESULT:", "PASS" if failures == 0 else "FAIL", flush=True)
    return 0 if failures == 0 else 1


# ---------------------------------------------------------------------------------------------- NPU frames
def _open_pipeline(ignite: str):
    from ignite_xdna.pipelines.yolo_pipeline import YoloPipeline
    _decoder_modules()
    pipe = YoloPipeline(ignite)
    s = pipe.session
    if not getattr(s, "direct_ingress", False):
        raise SystemExit("the container's session has no direct ingress (need the graph-engine .ignite)")
    captured = {}
    run, stage = s.run_yolo_monolithic, s.stage_image

    def run_capture(*a, **kw):
        res = run(*a, **kw)
        captured["heads"] = res[0] if isinstance(res, tuple) else res
        return res

    def stage_capture(img):
        captured["pad_scale"] = stage(img)
        return captured["pad_scale"]

    s.run_yolo_monolithic, s.stage_image = run_capture, stage_capture
    return pipe, captured


def _frames(args):
    """Yields (source label, name, BGR frame): images, still folders cycled, then camera frames."""
    import cv2
    for image in args.image:
        yield "stills", Path(image).stem, cv2.imread(str(image))
    for folder in args.stills:
        paths = sorted(Path(folder).glob("*.jpg"))
        imgs = [(p, cv2.imread(str(p))) for p in paths]
        for c in range(args.cycles):
            for p, img in imgs:
                yield "stills", f"{Path(folder).name}_{p.stem}" + (f"_c{c}" if args.cycles > 1 else ""), img
    if args.camera_frames:
        cap = cv2.VideoCapture(args.camera_index, cv2.CAP_MSMF)
        try:
            if not cap.isOpened():
                print(f"camera {args.camera_index} did not open", flush=True)
                return
            for _ in range(30):
                cap.read()
            for i in range(args.camera_frames):
                ok, frame = cap.read()
                if not ok:
                    print(f"camera read failed after {i} frames", flush=True)
                    return
                yield "camera", f"camera_{i:03d}", frame
        finally:
            cap.release()


def cmd_record(args) -> int:
    _header("record")
    _xrt_smi("before")
    pipe, captured = _open_pipeline(args.ignite)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for source, name, img in _frames(args):
        for _ in range(3 if source == "stills" else 0):
            pipe.predict_sync(img, use_oracle_for_boxes=False)
        dets, tm = pipe.predict_sync(img, use_oracle_for_boxes=False)
        h = captured["heads"]
        pad, scale = captured["pad_scale"]
        rec = {k: np.array(h[k], copy=True) for k in HEADS}
        for k, v in (h.get("cls_max") or {}).items():
            rec["cls_max_" + k] = np.array(v, copy=True)
        rec["scales"] = np.array(json.dumps({k: [float(v[0]), int(v[1])] for k, v in h["scales"].items()}))
        rec["pad"] = np.array(pad, dtype=np.int64)
        rec["scale"] = np.array(scale, dtype=np.float64)
        rec["brightness"] = np.array(float(img.mean()))
        np.savez(out / f"{name}.npz", **rec)
        count += 1
        print(f"{name}: {len(dets)} detections, head_source {tm.head_source}, brightness {img.mean():.1f}", flush=True)
    pipe.close()
    _xrt_smi("after")
    print(f"record: {count} frames written to {out}", flush=True)
    return 0 if count else 1


def cmd_live(args) -> int:
    _decoder_modules()
    _header("live")
    _xrt_smi("before")
    pipe, captured = _open_pipeline(args.ignite)
    native = pipe._native
    if native is None:
        print("FAIL: the pipeline has no native decode", flush=True)
        return 1
    stats, failures, warm = {}, 0, {}
    for i, (source, name, img) in enumerate(_frames(args)):
        path = "native" if i % 2 == 0 else "numpy"
        other = "numpy" if path == "native" else "native"
        if warm.get(source, 0) < args.warmup:
            for p in (native, None):
                pipe._native = p
                pipe.predict_sync(img, use_oracle_for_boxes=False)
            warm[source] = warm.get(source, 0) + 1
        pipe._native = native if path == "native" else None
        dets, tm = pipe.predict_sync(img, use_oracle_for_boxes=False)
        pipe._native = native if other == "native" else None
        h = captured["heads"]
        pad, scale = captured["pad_scale"]
        other_dets = pipe.postprocess(h, pad, scale)
        pipe._native = native
        if tm.head_source != "npu":
            print(f"{name}: head_source {tm.head_source}", flush=True)
        if other_dets != dets:
            failures += 1
            print(f"{name}: MISMATCH {path} {len(dets)} vs {other} {len(other_dets)}", flush=True)
        st = stats.setdefault(source, {"native": [], "numpy": [], "g2g_native": [], "g2g_numpy": [], "dets": [],
                                       "bright": []})
        st[path].append(tm.postprocess_ms)
        st["g2g_" + path].append(tm.glass_to_glass_ms)
        st["dets"].append(len(dets))
        st["bright"].append(float(img.mean()))
    for source, st in stats.items():
        print(f"live {source}: {len(st['dets'])} frames (paths alternate), detections/frame {mean(st['dets']):.2f} "
              f"(min {min(st['dets'])}, max {max(st['dets'])}), brightness {mean(st['bright']):.1f}", flush=True)
        for path in ("native", "numpy"):
            xs, g = st[path], st["g2g_" + path]
            print(f"   {path:6s} postprocess median {median(xs):.4f} ms, mean {mean(xs):.4f}, p95 {_pct(xs, 95):.4f} "
                  f"({len(xs)} frames) | glass-to-glass median {median(g):.3f} ms, mean {mean(g):.3f}", flush=True)
    pipe.close()
    _xrt_smi("after")
    print(f"live: mismatches {failures}", flush=True)
    print("RESULT:", "PASS" if failures == 0 else "FAIL", flush=True)
    return 0 if failures == 0 else 1


# ---------------------------------------------------------------------------------------------- slowdown
def cmd_slowdown(args) -> int:
    _header("slowdown")
    print("OMP_WAIT_POLICY =", os.environ.get("OMP_WAIT_POLICY"), flush=True)

    def tight(buf, n=2000):
        for _ in range(100):
            zlib.crc32(buf)
        ts = []
        for _ in range(n):
            t0 = pc()
            zlib.crc32(buf)
            ts.append(pc() - t0)
        return median(ts) * 1e3

    bufs = {}
    for label, target in (("small", 0.010), ("large", 0.090)):
        n = 16384
        for _ in range(8):
            n = max(1024, int(n * target / tight(os.urandom(n), 300)))
        bufs[label] = os.urandom(n)
    base = {k: tight(v) for k, v in bufs.items()}
    print(f"tight loop: small {base['small']:.4f} ms ({len(bufs['small'])} B), large {base['large']:.4f} ms "
          f"({len(bufs['large'])} B)", flush=True)

    def measure(label, before, rounds):
        rows = {"small": [], "large": []}
        for i in range(rounds):
            before(i)
            for k in (("small", "large") if i % 2 == 0 else ("large", "small")):
                t0 = pc()
                zlib.crc32(bufs[k])
                rows[k].append((pc() - t0) * 1e3)
        s, l = median(rows["small"]), median(rows["large"])
        print(f"{label}: small {s:.4f} ms (x{s / base['small']:.2f}), large {l:.4f} ms (x{l / base['large']:.2f})",
              flush=True)

    def spin(seconds):
        end = pc() + seconds
        while pc() < end:
            pass

    if not args.pipeline:
        for wait in args.waits:
            measure(f"after sleep {wait} ms", lambda i, w=wait: time.sleep(w / 1e3), args.rounds)
            measure(f"after spin {wait} ms", lambda i, w=wait: spin(w / 1e3), args.rounds)
    else:
        import cv2
        _xrt_smi("before")
        pipe, _ = _open_pipeline(args.ignite)
        s = pipe.session
        imgs = [cv2.imread(str(p)) for p in sorted(Path(args.stills[0]).glob("*.jpg"))] if args.stills else []
        if not imgs:
            raise SystemExit("--pipeline needs --stills DIR")
        completed = s._completed

        def dispatch_wait(i):
            s.stage_image(imgs[i % len(imgs)])
            s.dispatch()
            s.read_heads()

        def dispatch_poll(i):
            s.stage_image(imgs[i % len(imgs)])
            s._run.start()
            while s._run.state() != completed:
                pass
            s.read_heads()

        measure("after an NPU dispatch awaited by run.wait()", dispatch_wait, args.rounds)
        measure("after an NPU dispatch awaited by polling run.state()", dispatch_poll, args.rounds)
        pipe.close()
        _xrt_smi("after")
    base2 = {k: tight(v) for k, v in bufs.items()}
    print(f"tight loop at the end: small {base2['small']:.4f} ms, large {base2['large']:.4f} ms", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    default_ignite = str(ROOT / "build" / "yolov8n_full.ignite")

    p = sub.add_parser("stress", help="random int8 heads, native vs numpy (no device)")
    p.add_argument("--trials", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_stress)

    p = sub.add_parser("replay", help="recorded heads, native vs numpy (no device)")
    p.add_argument("heads_dir")
    p.add_argument("--repeats", type=int, default=300)
    p.set_defaults(fn=cmd_replay)

    for name, fn, help_text in (("record", cmd_record, "record int8 heads from the NPU"),
                                ("live", cmd_live, "live decode timing and parity on the NPU")):
        p = sub.add_parser(name, help=help_text)
        if name == "record":
            p.add_argument("out_dir")
        p.add_argument("--ignite", default=default_ignite)
        p.add_argument("--image", action="append", default=[], help="a still image (repeatable)")
        p.add_argument("--stills", action="append", default=[], help="a folder of .jpg frames (repeatable)")
        p.add_argument("--cycles", type=int, default=1, help="passes over each --stills folder")
        p.add_argument("--camera-frames", type=int, default=0)
        p.add_argument("--camera-index", type=int, default=0)
        if name == "live":
            p.add_argument("--warmup", type=int, default=10, help="warm-up frames per source, both paths")
        p.set_defaults(fn=fn)

    p = sub.add_parser("slowdown", help="how waiting changes the speed of fixed host work")
    p.add_argument("--waits", type=float, nargs="*", default=[0.5, 2.0, 7.5, 33.0])
    p.add_argument("--rounds", type=int, default=300)
    p.add_argument("--pipeline", action="store_true")
    p.add_argument("--ignite", default=default_ignite)
    p.add_argument("--stills", action="append", default=[])
    p.set_defaults(fn=cmd_slowdown)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

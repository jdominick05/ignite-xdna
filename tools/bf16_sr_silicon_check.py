"""A bf16 super-resolution container on the NPU, beside its int8 twin, in one sitting.

Three questions, answered in this order because each is only worth asking once the one before it holds:

1. EXACT. Does the device equal the emulator to the bit? Each input is staged by the session, run once
   on the device, and then recomputed offline by ``graph_reference_bf16.run_direct`` (the aligned
   multiply-accumulate, the model measured on silicon) from the very patterns the session staged. The
   comparison is on 16-bit patterns: every lane of the tail, padding lanes included, and every tensor
   still resident in the workspace when the frame ends. The input region is read back after the
   dispatch to show nothing wrote into it, and the first input is staged again as the last frame and
   must come back identical, since a region stale from the frame before only shows on a later frame.
   Each device output is also compared with the W8A16 ORT oracle (``tools/w8a16_oracle.py``). Where the
   device equals the emulator, that comparison has to reproduce what the emulator measured against the
   same oracle offline, and so it is the device's own distance to the model.
2. TIMING. Per tile, split into host staging, NPU dispatch, readback and host postprocess, for the
   int8 container and the bf16 one alternately in the same sitting, and for the float model on this
   CPU. The spans are reported separately and never folded together.
3. QUALITY. PSNR and SSIM on each dataset, tiled exactly as ``pipelines/sesr/5_eval.py`` tiles it
   (256 x 256, overlap 8), for both containers and for three CPU references: the float model, the QDQ
   model with graph optimizations off (which the int8 container is expected to equal pixel for pixel),
   and the W8A16 oracle (which the bf16 container is expected to approach). Datasets are reported
   separately and never averaged together.

Generic over the containers, the QDQ and float models, the images and the datasets. It opens one
hardware context at a time and closes each before the next, prints ``xrt-smi``'s partition report at
the start and after the last close, and never retries a dispatch.

    ./scripts/research-lowlevel.sh --log results/aie/<name>.log --npu --seconds 2400 -- \\
        bash scripts/research-iron.sh tools/bf16_sr_silicon_check.py \\
        --container build/sesr_m7_bf16.ignite --int8-container build/sesr_m7.ignite \\
        --qdq models/sesr_m7_xint8.onnx --fp32 models/sesr_m7_fp32.onnx

WHAT IT IS NOT. One sitting on one machine. The timing is this machine's, on this day, in this power
mode; the CPU arm is ONNX Runtime in the engine's environment, which is not the version the study's
evaluation ran. Quality numbers are means over five and fourteen images.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ignite_xdna.compiler import engine_bf16_emulator as em  # noqa: E402
from ignite_xdna.compiler import engine_schedule_bf16 as eb  # noqa: E402
from ignite_xdna.compiler import graph_ir  # noqa: E402
from ignite_xdna.compiler import graph_reference_bf16 as gr16  # noqa: E402
from ignite_xdna.compiler.graph_reference import host_session  # noqa: E402
from ignite_xdna.compiler.serializer import ELEM_BF16, IgniteModelReader  # noqa: E402
from ignite_xdna.pipelines import npu_power  # noqa: E402
from ignite_xdna.pipelines import preprocess as pp  # noqa: E402
from ignite_xdna.runtime import graph_session as gs  # noqa: E402

import w8a16_oracle as wo  # noqa: E402
from bf16_engine_vs_oracle import compare, depth_to_space, ordered  # noqa: E402
from npu import sesr  # noqa: E402

TILE = 256
DEFAULT_IMAGES = [ROOT / "data" / "sesr_val" / "Set5_LR_x2" / f"{n}.png"
                  for n in ("baby", "bird", "butterfly", "head", "woman")]


def emit(tag: str, obj) -> None:
    print(f"{tag} {json.dumps(obj, sort_keys=True)}", flush=True)


def rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return Path(p).name


def sha16(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def stats(v) -> dict:
    a = np.asarray(v, np.float64)
    return {"mean": round(float(a.mean()), 4), "p50": round(float(np.percentile(a, 50)), 4),
            "p99": round(float(np.percentile(a, 99)), 4), "min": round(float(a.min()), 4),
            "max": round(float(a.max()), 4)}


# --------------------------------------------------------------------------- setup and witnesses
def witness(when: str) -> bool:
    rc, text = npu_power.run_xrt_smi(["examine", "-r", "aie-partitions"])
    idle = "No hardware contexts running" in text
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not set(ln.strip()) <= set("-")]
    emit("BF16_SR_WITNESS", {"when": when, "idle": idle, "rc": rc, "report": lines[-6:]})
    return idle


def xrt_host() -> dict:
    _, text = npu_power.run_xrt_smi(["examine"])
    want = {"xrt_version": r"^\s*Version\s*:\s*(\S+)", "npu_driver": r"NPU Driver Version\s*:\s*(\S+)",
            "npu_firmware": r"NPU Firmware Version\s*:\s*(\S+)", "device": r"\|\s*\[[^\]]+\]\s*\|\s*([^|]+?)\s*\|"}
    out = {}
    for key, pat in want.items():
        m = re.search(pat, text, re.M)
        out[key] = m.group(1) if m else None
    return out


def container_facts(path: Path) -> dict:
    with IgniteModelReader(str(path)) as r:
        m = r.manifest
        insts, wp = r.get_blob_bytes("insts.bin"), r.get_blob_bytes("wpackets.bin")
    ge = m["graph_engine"]
    return {"path": rel(path), "sha256": sha16(path), "engine": m.get("engine"),
            "elem": gs.container_elem(m), "task": m.get("task"),
            "kernel_object_sha256": (m.get("kernel_object_sha256") or ge.get("kernel_object_sha256") or "")[:16],
            "xclbin_sha256": (m.get("xclbin_sha256") or ge.get("xclbin_sha256") or "")[:16],
            "source_model_sha256": (m.get("source_model_sha256") or "")[:16],
            "insts_sha256": hashlib.sha256(insts).hexdigest()[:16],
            "wpackets_sha256": hashlib.sha256(wp).hexdigest()[:16],
            "egress_bytes": m.get("egress_bytes"), "workspace_bytes": ge.get("workspace_bytes")}


def provenance() -> dict:
    mods = {m.__name__.rsplit(".", 1)[-1]: Path(m.__file__).resolve() for m in (eb, gr16, em, gs, graph_ir)}
    src = (ROOT / "src").resolve()
    stray = {k: str(v) for k, v in mods.items() if src not in v.parents}
    if stray:
        raise SystemExit(f"engine modules imported from outside {src}: {stray}")
    files = [Path(__file__).resolve(), Path(wo.__file__).resolve(), *mods.values()]
    return {rel(p): sha16(p) for p in files}


def open_session(path: Path, want_bf16: bool):
    s = gs.open_sr_session(path)
    if isinstance(s, gs.Bf16DenseGraphSession) != want_bf16:
        s.close()
        raise RuntimeError(f"{rel(path)} opened in {type(s).__name__}; expected the "
                         f"{'bf16' if want_bf16 else 'int8'} session")
    return s


def close_session(s) -> None:
    if s is not None:
        s.close()
    gc.collect()


# --------------------------------------------------------------------------- 1. exactness
def survivors(ge: dict) -> dict:
    """For each workspace slot, the tensor that holds it when the frame ends: the last one produced."""
    last = {}
    order = {L["output"]: int(L["index"]) for L in ge["layers"]}
    for name, p in ge["placements"].items():
        idx = order.get(name, -1)
        base = int(p["base"])
        if base not in last or idx > last[base][0]:
            last[base] = (idx, name)
    return {name: idx for idx, name in last.values()}


def read_input_region(s) -> np.ndarray:
    p = s.input_placement
    base = int(p["base"])
    s.bo_ws.sync(s.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, s._input_bytes, base)
    if s._ws_map is not None:
        raw = np.array(s._ws_map[base:base + s._input_bytes])
    else:
        raw = np.frombuffer(s.bo_ws.read(s._input_bytes, base), dtype=np.uint8).copy()
    return raw.view(np.uint16).reshape(s._input_plane.shape)


def device_frames(s, frames, timeout_ms: int) -> list:
    """Run each (label, kind, payload) frame once; keep what was staged and what came back."""
    ge = s.ge
    keep = survivors(ge)
    out = []
    for label, kind, payload in frames:
        if kind == "image":
            s.stage_image(payload)
        else:
            s.stage_quantized(payload)
        staged = s._input_plane.copy()
        t0 = time.perf_counter()
        s.dispatch(timeout_ms=timeout_ms)
        ms = (time.perf_counter() - t0) * 1e3
        tail = s.read_output().copy()
        tensors = {name: s.read_tensor(name).copy() for name in keep}
        out.append({"label": label, "staged": staged, "tail": tail, "tensors": tensors,
                    "input_after": read_input_region(s), "dispatch_ms": ms})
    return out


def check_frames(frames: list, ir, placement: dict, dense: dict, oracle_bytes: bytes, mac_model) -> list:
    """Offline, after the device is closed: every frame against the emulator and the oracle."""
    h = int(placement["halo"])
    H, W = int(placement["height"]), int(placement["width"])
    tail_name = dense["tensor"]
    channels, bs = int(dense["channels"]), int(dense["transform"]["blocksize"])
    mode = dense["transform"].get("mode", "DCR")
    reports, seen = [], {}
    for i, f in enumerate(frames):
        rep = {"label": f["label"], "dispatch_ms": round(f["dispatch_ms"], 3)}
        staged = f["staged"]
        rep["input_region_intact"] = bool(np.array_equal(staged, f["input_after"]))
        x = em.from_bf16_bits(np.ascontiguousarray(staged[h:h + H, h:h + W, :3])).transpose(2, 0, 1).copy()
        rep["staged_is_the_compilers_plane"] = bool(np.array_equal(staged, eb.input_plane(placement, x)))
        nb, oh, ow, _ = f["tail"].shape
        dev = np.moveaxis(f["tail"], 3, 1).reshape(nb * 8, oh, ow)
        rep["nonfinite_tail"] = int(np.count_nonzero(~np.isfinite(em.from_bf16_bits(dev[:channels]))))
        key = staged.tobytes()
        if key in seen:
            # The same input again, later in the sitting: it must come back identical to its first frame.
            first = frames[seen[key]]
            rep["repeat_of"] = first["label"]
            rep["repeat_tail_identical"] = bool(np.array_equal(f["tail"], first["tail"]))
            rep["repeat_tensors_identical"] = all(np.array_equal(f["tensors"][n], first["tensors"][n])
                                                  for n in f["tensors"])
            reports.append(rep)
            continue
        seen[key] = i
        t0 = time.perf_counter()
        ref = gr16.run_direct(ir, x, mac_model=mac_model)
        rep["emulator_s"] = round(time.perf_counter() - t0, 1)
        want = em.bf16_bits(ref[tail_name])
        ulp = np.abs(ordered(dev) - ordered(want))
        rep["tail"] = {"values": int(dev.size), "lanes": int(dev.shape[0]), "differ": int(np.count_nonzero(dev != want)),
                       "differ_padding_lanes": int(np.count_nonzero(dev[channels:] != want[channels:])),
                       "zero_sign_only": int(np.count_nonzero((dev != want) & (ulp == 0))), "max_ulp": int(ulp.max())}
        per = {}
        for name, got in f["tensors"].items():
            w = em.bf16_bits(ref[name][:got.shape[0]])
            per[name] = int(np.count_nonzero(got != w))
        rep["resident_tensors"] = {"compared": len(per), "values": int(sum(t.size for t in f["tensors"].values())),
                                   "differ": int(sum(per.values())), "per_tensor_differ": per}
        # The device against the W8A16 oracle, at the model's output.
        y = wo.run(oracle_bytes, x[None])[0]
        dev_img = depth_to_space(em.from_bf16_bits(dev[:channels]), bs, mode)
        rep["vs_oracle"] = compare(dev_img, em.to_bf16(y.astype(np.float32)))
        rep["exact"] = (rep["tail"]["differ"] == 0 and rep["resident_tensors"]["differ"] == 0
                        and rep["input_region_intact"] and rep["nonfinite_tail"] == 0)
        reports.append(rep)
    return reports


# --------------------------------------------------------------------------- 2. timing
def time_container(path: Path, bf16: bool, tile: np.ndarray, frames: int, warmup: int, timeout_ms: int) -> dict:
    s = open_session(path, bf16)
    try:
        rows = []
        for i in range(warmup + frames):
            t0 = time.perf_counter()
            s.stage_image(tile)
            t1 = time.perf_counter()
            s.dispatch(timeout_ms=timeout_ms)
            t2 = time.perf_counter()
            blocks = s.read_output()
            t3 = time.perf_counter()
            s.postprocess(blocks)
            t4 = time.perf_counter()
            if i >= warmup:
                rows.append(((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t4 - t3) * 1e3, (t4 - t0) * 1e3))
    finally:
        close_session(s)
    a = np.array(rows)
    return {"frames": frames, "warmup": warmup, "stage_ms": stats(a[:, 0]), "dispatch_ms": stats(a[:, 1]),
            "readback_ms": stats(a[:, 2]), "postprocess_ms": stats(a[:, 3]), "tile_total_ms": stats(a[:, 4])}


def time_cpu(sess, tile: np.ndarray, frames: int, warmup: int) -> dict:
    name = sess.get_inputs()[0].name
    rows = []
    for i in range(warmup + frames):
        t0 = time.perf_counter()
        x, _ = sesr.preprocess(tile, TILE)
        t1 = time.perf_counter()
        y = sess.run(None, {name: x})[0]
        t2 = time.perf_counter()
        sesr.postprocess(y)
        t3 = time.perf_counter()
        if i >= warmup:
            rows.append(((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t3 - t0) * 1e3))
    a = np.array(rows)
    return {"frames": frames, "warmup": warmup, "preprocess_ms": stats(a[:, 0]), "run_ms": stats(a[:, 1]),
            "postprocess_ms": stats(a[:, 2]), "tile_total_ms": stats(a[:, 3])}


# --------------------------------------------------------------------------- 3. quality
def pairs(data: Path, dataset: str) -> list:
    hr_dir, lr_dir = data / f"{dataset}_HR", data / f"{dataset}_LR_x2"
    out = []
    for hr in sorted(hr_dir.glob("*.png")):
        lr = lr_dir / hr.name
        if lr.exists():
            out.append((lr, hr))
    if not out:
        raise SystemExit(f"no image pairs under {rel(data)} for {dataset}")
    return out


def upscale(lr: np.ndarray, run_tile, overlap: int) -> np.ndarray:
    """``5_eval.py``'s tiling: one run when the image is the tile size, else overlapped tiles merged."""
    if lr.shape[:2] == (TILE, TILE):
        return run_tile(lr)
    tiles, orig_hw, padded_hw, grid_hw = sesr.split_into_tiles(lr, patch_size=(TILE, TILE), overlap=overlap)
    return sesr.merge_tiles([run_tile(t) for t in tiles], orig_hw, padded_hw, grid_hw,
                            scale=sesr.SCALE, overlap=overlap)


def score(sr: np.ndarray, hr: np.ndarray) -> dict:
    h, w = min(sr.shape[0], hr.shape[0]), min(sr.shape[1], hr.shape[1])
    a, b = sr[:h, :w], hr[:h, :w]
    return {"psnr_y": sesr.compute_psnr(a, b, True), "ssim_y": sesr.compute_ssim(a, b, True),
            "psnr_rgb": sesr.compute_psnr(a, b, False), "ssim_rgb": sesr.compute_ssim(a, b, False)}


def cpu_tile_fn(sess):
    name = sess.get_inputs()[0].name

    def run(tile):
        x, _ = sesr.preprocess(tile, TILE)
        return sesr.postprocess(sess.run(None, {name: x})[0])
    return run


def upscale_all(data: Path, datasets, run_tile, overlap: int) -> dict:
    """{dataset: [(image name, upscaled BGR)]}"""
    return {d: [(lr.stem, upscale(cv2.imread(str(lr)), run_tile, overlap)) for lr, _ in pairs(data, d)]
            for d in datasets}


def quality_report(data: Path, datasets, arms: dict) -> None:
    for d in datasets:
        hrs = {lr.stem: cv2.imread(str(hr)) for lr, hr in pairs(data, d)}
        per_arm = {}
        for arm, images in arms.items():
            scores = {name: score(img, hrs[name]) for name, img in images[d]}
            per_arm[arm] = scores
            emit("BF16_SR_QUALITY", {
                "dataset": d, "arm": arm, "images": len(scores),
                **{k: round(float(np.mean([s[k] for s in scores.values()])), 4)
                   for k in ("psnr_y", "ssim_y", "psnr_rgb", "ssim_rgb")},
                "per_image_psnr_y": {n: round(s["psnr_y"], 4) for n, s in scores.items()}})
        for a, b in (("int8_npu", "xint8_cpu_noopt"), ("bf16_npu", "w8a16_oracle_cpu"), ("bf16_npu", "int8_npu")):
            if a in arms and b in arms:
                ia, ib = dict(arms[a][d]), dict(arms[b][d])
                diff = [np.abs(ia[n].astype(np.int16) - ib[n].astype(np.int16)) for n in ia]
                emit("BF16_SR_AGREEMENT", {"dataset": d, "a": a, "b": b,
                                           "pixels": int(sum(x.size for x in diff)),
                                           "differ": int(sum(np.count_nonzero(x) for x in diff)),
                                           "max_abs": int(max(int(x.max()) for x in diff))})


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--container", type=Path, required=True, help="the bf16 super-resolution container")
    ap.add_argument("--int8-container", type=Path, default=None, help="its int8 twin, for timing and quality")
    ap.add_argument("--qdq", type=Path, required=True, help="the QDQ model both were compiled from")
    ap.add_argument("--fp32", type=Path, required=True, help="the float model: the CPU arm and the oracle's graph")
    ap.add_argument("--image", type=Path, action="append", default=None,
                    help="exactness inputs, in order (default: the five Set5 LR x2 images)")
    ap.add_argument("--seed", type=int, action="append", default=None,
                    help="seeded integer inputs after the images, staged as bf16 patterns (default: 0)")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "sesr_val")
    ap.add_argument("--dataset", action="append", default=None, help="default: Set5 and Set14")
    ap.add_argument("--overlap", type=int, default=8)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=2, help="int8 / bf16 timing alternations")
    ap.add_argument("--timeout-ms", type=int, default=10000)
    ap.add_argument("--mac-model", default=None, help="the emulator's accumulate model (default: aligned)")
    ap.add_argument("--no-exact", action="store_true")
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--no-quality", action="store_true")
    args = ap.parse_args()

    import onnxruntime as ort
    images = args.image or DEFAULT_IMAGES
    seeds = [0] if args.seed is None else args.seed
    datasets = args.dataset or ["Set5", "Set14"]
    bf16_facts = container_facts(args.container)
    if bf16_facts["elem"] != ELEM_BF16:
        raise SystemExit(f"{rel(args.container)} is not a bf16 container ({bf16_facts['elem']})")
    emit("BF16_SR_SETUP", {
        "host": platform.node(), "bf16": bf16_facts,
        "int8": container_facts(args.int8_container) if args.int8_container else None,
        "qdq": {"path": rel(args.qdq), "sha256": sha16(args.qdq)},
        "fp32": {"path": rel(args.fp32), "sha256": sha16(args.fp32)},
        "npu_power_mode": npu_power.read_mode(), "xrt": xrt_host(),
        "onnxruntime": ort.__version__, "numpy": np.__version__, "cv2": cv2.__version__,
        # int8 staging and DepthToSpace take the native paths when this library loaded; bf16 never does.
        "native_preprocess_loaded": pp._LIB is not None,
        "mac_model": args.mac_model or em.MAC_MODEL, "provenance": provenance(),
        "images": [rel(p) for p in images], "seeds": seeds, "datasets": datasets,
        "frames": args.frames, "warmup": args.warmup, "rounds": args.rounds})
    if not witness("start"):
        raise SystemExit("the device is not idle; a number measured beside another context is contention")

    ir = graph_ir.lower_yolov8n(str(args.qdq))
    oracle_model, _ = wo.build(args.qdq, args.fp32)
    oracle_bytes = oracle_model.SerializeToString()
    with IgniteModelReader(str(args.container)) as r:
        m = r.manifest
    placement = m["graph_engine"]["placements"][m["graph_engine"]["input_tensor"]]
    tile = cv2.imread(str(images[0]))
    if tile.shape[:2] != (TILE, TILE):
        tile = sesr.split_into_tiles(tile, patch_size=(TILE, TILE), overlap=args.overlap)[0][0]
    failed = False
    frames, npu_arms, timing = [], {}, []
    try:
        # ---- everything that needs the device, one context at a time
        if not args.no_exact:
            plan = [(p.stem, "image", cv2.imread(str(p))) for p in images]
            plan += [(f"seed{k}", "bits", em.bf16_bits(em.to_bf16(wo.seeded_input(args.qdq, k)[0]))) for k in seeds]
            plan += [(f"{images[0].stem}_again", "image", cv2.imread(str(images[0])))]
            s = open_session(args.container, True)
            try:
                frames = device_frames(s, plan, args.timeout_ms)
            finally:
                close_session(s)
            emit("BF16_SR_DEVICE_FRAMES", {"frames": [f["label"] for f in frames],
                                            "dispatch_ms": [round(f["dispatch_ms"], 3) for f in frames]})
        if not args.no_timing:
            arms = ([("int8", args.int8_container, False)] if args.int8_container else []) + \
                   [("bf16", args.container, True)]
            for rnd in range(args.rounds):
                for name, path, is_bf16 in arms:
                    t = time_container(path, is_bf16, tile, args.frames, args.warmup, args.timeout_ms)
                    timing.append(t)
                    emit("BF16_SR_TIMING", {"arm": f"{name}_npu", "round": rnd, "span": "one 256x256 tile", **t})
        if not args.no_quality:
            for name, path, is_bf16 in ([("int8_npu", args.int8_container, False)] if args.int8_container else []) + \
                    [("bf16_npu", args.container, True)]:
                s = open_session(path, is_bf16)
                try:
                    npu_arms[name] = upscale_all(args.data, datasets, lambda t, s=s: s.run(t)[0], args.overlap)
                finally:
                    close_session(s)
    except Exception as exc:  # noqa: BLE001 - a device failure is the finding; say it and stop
        emit("BF16_SR_ERROR", {"type": type(exc).__name__, "message": str(exc)[:500]})
        failed = True
    idle = witness("end")
    if failed:
        return 3

    # ---- CPU only from here
    if not args.no_timing:
        fp32 = ort.InferenceSession(str(args.fp32), ort.SessionOptions(), providers=["CPUExecutionProvider"])
        emit("BF16_SR_TIMING", {"arm": "fp32_cpu", "round": 0, "span": "one 256x256 tile",
                                "ort_session_options": "default", **time_cpu(fp32, tile, args.frames, args.warmup)})
    if not args.no_quality:
        cpu = {"fp32_cpu": ort.InferenceSession(str(args.fp32), ort.SessionOptions(),
                                                providers=["CPUExecutionProvider"]),
               "xint8_cpu_noopt": host_session(Path(args.qdq).read_bytes()),
               "w8a16_oracle_cpu": host_session(oracle_bytes)}
        arms = dict(npu_arms)
        for name, sess in cpu.items():
            arms[name] = upscale_all(args.data, datasets, cpu_tile_fn(sess), args.overlap)
        quality_report(args.data, datasets, arms)

    reports = []
    if frames:
        reports = check_frames(frames, ir, placement, m["dense_output"], oracle_bytes, args.mac_model)
        for rep in reports:
            emit("BF16_SR_EXACT", rep)
    fresh = [r for r in reports if "repeat_of" not in r]
    repeats = [r for r in reports if "repeat_of" in r]
    exact = bool(fresh) and all(r["exact"] and r["staged_is_the_compilers_plane"] for r in fresh) and \
        all(r["repeat_tail_identical"] and r["repeat_tensors_identical"] and r["input_region_intact"] for r in repeats)
    emit("BF16_SR_RESULT", {
        "device_idle_after": idle,
        "exact_inputs": len(fresh), "repeat_frames": len(repeats), "device_equals_emulator": exact if frames else None,
        "tail_values_compared": int(sum(r["tail"]["values"] for r in fresh)),
        "resident_values_compared": int(sum(r["resident_tensors"]["values"] for r in fresh)),
        "worst_vs_oracle": max(fresh, key=lambda r: r["vs_oracle"]["differ"])["vs_oracle"] if fresh else None})
    return 0 if (exact or not frames) and idle else 1


if __name__ == "__main__":
    sys.exit(main())

"""A bf16 super-resolution container on the NPU, beside its int8 twin, in one sitting.

Three questions, answered in this order because each is only worth asking once the one before it holds:

1. EXACT. Does the device equal the emulator to the bit? Each input is staged by the session, run once
   on the device, and then recomputed offline by ``graph_reference_bf16.run_direct`` (under
   ``engine_bf16_emulator.MAC_MODEL``, the model in force: exp_sum since 2026-09-23, aligned before)
   from the very patterns the session staged. The
   comparison is on 16-bit patterns: every lane of the tail, padding lanes included, and every tensor
   still resident in the workspace when the frame ends. The input region is read back after the
   dispatch to show nothing wrote into it, and the first input is staged again as the last frame and
   must come back identical, since a region stale from the frame before only shows on a later frame.
   Each device output is also compared with the W8A16 ORT oracle (``tools/w8a16_oracle.py``). Where the
   device equals the emulator, that comparison has to reproduce what the emulator measured against the
   same oracle offline, and so it is the device's own distance to the model.
   The host ends are checked on the same frames. Each staged image's values must be
   ``npu/sesr.py``'s own preprocess, and each tail's image from the session's host path must equal
   ``bf16_dense_image``'s numpy one, byte for byte.
2. TIMING. Per tile, split into host staging, NPU dispatch, readback and host postprocess, for the
   int8 container and the bf16 one alternately in the same sitting, and for the float model on this
   CPU. The bf16 container is timed once per host path it is given (``--bf16-host``): native
   (``preprocess_simd``) and numpy, so the host ends are the only thing that differs between those two
   arms. The spans are reported separately and never folded together, and each arm says which host
   path actually ran.
3. QUALITY. PSNR and SSIM on each dataset, tiled exactly as ``pipelines/sesr/5_eval.py`` tiles it
   (256 x 256, overlap 8), for both containers and for three CPU references: the float model, the QDQ
   model with graph optimizations off (which the int8 container is expected to equal pixel for pixel),
   and the W8A16 oracle (which the bf16 container is expected to approach). Datasets are reported
   separately and never averaged together. Every bf16 tile's image is also recomputed through the
   numpy egress from the same device output and must be identical.

WHEN THE DEVICE AND THE EMULATOR DISAGREE. The chained check feeds every layer the emulator's own
previous output, so it says THAT a frame differs, and where it is first visible, but not where it
starts: a layer that is overwritten before the frame ends (SESR's body.0 to body.3) is never compared.
``--isolate`` (on the inexact frames by default) replays each layer whose inputs were all still
resident, fed the DEVICE's own input tensors, under every candidate accumulate model the emulator
knows (``engine_bf16_emulator.MAC_MODELS``), and lists each position where any of them differs from
the device: the device's pattern, each model's, the tile it fell in, and the accumulator's exact
value before rounding. ``--dump-frames DIR`` writes the device frames out, and ``--from-dump DIR``
checks them again later with no device at all, refusing a dump from another container or model.

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
import math
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
from ignite_xdna.compiler.engine_schedule import tile_origins  # noqa: E402
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
    mods = {m.__name__.rsplit(".", 1)[-1]: Path(m.__file__).resolve() for m in (eb, gr16, em, gs, graph_ir, pp)}
    src = (ROOT / "src").resolve()
    stray = {k: str(v) for k, v in mods.items() if src not in v.parents}
    if pp._LIB is not None and src not in Path(pp._LIB._name).resolve().parents:
        stray["preprocess_simd"] = pp._LIB._name
    if stray:
        raise SystemExit(f"engine modules imported from outside {src}: {stray}")
    files = [Path(__file__).resolve(), Path(wo.__file__).resolve(), *mods.values()]
    if pp._LIB is not None:
        files.append(Path(pp._LIB._name).resolve())
    return {rel(p): sha16(p) for p in files}


def native_library() -> dict:
    return {"loaded": pp._LIB is not None,
            "bf16_ingress": pp.has_native("bgr_to_c8_plane_bf16"),
            "bf16_egress": pp.has_native("depth_to_space_crd_bgr_bf16")}


def open_session(path: Path, want_bf16: bool, native_host: bool = True):
    s = gs.open_sr_session(path, native_host=native_host)
    if isinstance(s, gs.Bf16DenseGraphSession) != want_bf16:
        s.close()
        raise RuntimeError(f"{rel(path)} opened in {type(s).__name__}; expected the "
                         f"{'bf16' if want_bf16 else 'int8'} session")
    return s


def numpy_image(s, blocks: np.ndarray) -> np.ndarray:
    """The numpy egress of a bf16 session's tail: what its host path must equal."""
    return gs.bf16_dense_image(blocks, s._out_channels, s._bs, s._mean)


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
        host = {}
        if kind == "image":
            s.stage_image(payload)
            host["stage_path"] = s.stage_path
            host["sesr_input"] = sesr.preprocess(payload, TILE)[0][0]
        else:
            s.stage_quantized(payload)
        staged = s._input_plane.copy()
        t0 = time.perf_counter()
        s.dispatch(timeout_ms=timeout_ms)
        ms = (time.perf_counter() - t0) * 1e3
        tail = s.read_output().copy()
        tensors = {name: s.read_tensor(name).copy() for name in keep}
        # The session's host egress against numpy's, on this very device output. Host only.
        try:
            host["native_egress_equals_numpy"] = bool(np.array_equal(s.postprocess(tail), numpy_image(s, tail)))
            host["egress_path"] = s.egress_path
        except RuntimeError as exc:
            host["egress_error"] = str(exc)[:200]
        out.append({"label": label, "staged": staged, "tail": tail, "tensors": tensors,
                    "input_after": read_input_region(s), "dispatch_ms": ms, "host": host})
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
        host = {k: v for k, v in f["host"].items() if k != "sesr_input"}
        if "sesr_input" in f["host"]:
            # The staged values are the float pipeline's own input, not only laid out the compiler's way.
            host["staged_is_npu_sesr_preprocess"] = bool(np.array_equal(x, f["host"]["sesr_input"]))
        rep["host"] = host
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
        per, where = {}, {}
        for name, got in f["tensors"].items():
            w = em.bf16_bits(ref[name][:got.shape[0]])
            per[name] = int(np.count_nonzero(got != w))
            if per[name]:
                where[name] = where_differ(got, w, ir.tensors[name])
        rep["resident_tensors"] = {"compared": len(per), "values": int(sum(t.size for t in f["tensors"].values())),
                                   "differ": int(sum(per.values())), "per_tensor_differ": per, "where": where}
        # The device against the W8A16 oracle, at the model's output.
        y = wo.run(oracle_bytes, x[None])[0]
        dev_img = depth_to_space(em.from_bf16_bits(dev[:channels]), bs, mode)
        rep["vs_oracle"] = compare(dev_img, em.to_bf16(y.astype(np.float32)))
        rep["exact"] = (rep["tail"]["differ"] == 0 and rep["resident_tensors"]["differ"] == 0
                        and rep["input_region_intact"] and rep["nonfinite_tail"] == 0)
        reports.append(rep)
    return reports


# --------------------------------------------------------------------------- 1b. the dump and the replay
def dump_frames(frames: list, directory: Path, facts: dict) -> dict:
    """Write each device frame to ``directory``, one npz apiece plus ``dump.json``, for ``--from-dump``.

    The patterns are kept exactly as the device returned them. The dump names the container and the
    QDQ model it came from, and ``--from-dump`` refuses any other pair, so a replay cannot quietly
    check one container's tensors against another's emulator.
    """
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for i, f in enumerate(frames):
        names = sorted(f["tensors"])
        arrays = {"staged": f["staged"], "tail": f["tail"], "input_after": f["input_after"]}
        arrays.update({f"tensor{k}": f["tensors"][n] for k, n in enumerate(names)})
        if "sesr_input" in f["host"]:
            arrays["sesr_input"] = f["host"]["sesr_input"]
        meta = {"label": f["label"], "dispatch_ms": f["dispatch_ms"], "tensor_names": names,
                "host": {k: v for k, v in f["host"].items() if k != "sesr_input"}}
        arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
        path = directory / f"{i:02d}_{f['label']}.npz"
        np.savez_compressed(path, **arrays)
        files[path.name] = sha16(path)
    (directory / "dump.json").write_text(json.dumps({**facts, "files": files}, sort_keys=True, indent=1))
    return files


def load_frames(directory: Path, facts: dict) -> list:
    """The frames ``dump_frames`` wrote, refused unless every file hashes as recorded and the dump came
    from the same container and QDQ model as this run."""
    info = json.loads((directory / "dump.json").read_text())
    for key, want in facts.items():
        if info.get(key) != want:
            raise SystemExit(f"{directory} was dumped with {key} {info.get(key)}; this run has {want}")
    frames = []
    for name in sorted(info["files"]):
        path = directory / name
        if sha16(path) != info["files"][name]:
            raise SystemExit(f"{path} does not hash as dump.json records")
        with np.load(path) as z:
            meta = json.loads(z["meta"].tobytes().decode())
            f = {"label": meta["label"], "dispatch_ms": meta["dispatch_ms"], "staged": z["staged"],
                 "tail": z["tail"], "input_after": z["input_after"], "host": dict(meta["host"]),
                 "tensors": {n: z[f"tensor{k}"] for k, n in enumerate(meta["tensor_names"])}}
            if "sesr_input" in z.files:
                f["host"]["sesr_input"] = z["sesr_input"]
        frames.append(f)
    return frames


def _values(bits: np.ndarray, blocks: int) -> np.ndarray:
    """A device tensor's patterns ``[C][H][W]`` as the float32 values they denote, +0.0 up to whole blocks."""
    v = em.from_bf16_bits(bits)
    if v.shape[0] < blocks * 8:
        v = np.concatenate([v, np.zeros((blocks * 8 - v.shape[0],) + v.shape[1:], np.float32)])
    return v


def _origin(origins, p: int, size: int) -> int:
    """The origin of the LAST tile covering ``p``: edge tiles overlap, and the later one is written last."""
    return max(o for o in origins if o <= p < o + size)


def where_differ(got: np.ndarray, want: np.ndarray, t, limit: int = 16) -> dict:
    """Where two ``[C][H][W]`` pattern tensors differ: the bounding box, and the first ``limit`` positions
    with both patterns, their distance in bf16 steps, and the tile that wrote each. A lone position is
    where a difference can start; a box that widens layer by layer is one spreading."""
    pos = np.argwhere(got != want)
    ys, xs = tile_origins(t.height, em.TILE_ROWS), tile_origins(t.width, em.TILE_COLS)
    rows = []
    for c, y, x in pos[:limit]:
        c, y, x = int(c), int(y), int(x)
        oy, ox = _origin(ys, y, em.TILE_ROWS), _origin(xs, x, em.TILE_COLS)
        d, e = int(got[c, y, x]), int(want[c, y, x])
        rows.append({"c": c, "y": y, "x": x, "device": f"{d:04x}", "emulator": f"{e:04x}",
                     "steps": int(ordered(np.array([d]))[0] - ordered(np.array([e]))[0]),
                     "tile_origin": [oy, ox], "covered_by": [sum(o <= y < o + em.TILE_ROWS for o in ys),
                                                             sum(o <= x < o + em.TILE_COLS for o in xs)]})
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    return {"bbox": {"c": [int(lo[0]), int(hi[0])], "y": [int(lo[1]), int(hi[1])], "x": [int(lo[2]), int(hi[2])]},
            "positions": rows}


def exact_value(layer, values: dict, c: int, y: int, x: int):
    """The accumulator's exact value at one output position, before the epilogue rounds it: the sum of
    every product in float64 by ``math.fsum``, plus the packet's bf16 bias. Single-input layers only."""
    if len(layer.inputs) != 1:
        return None
    seg = layer.inputs[0]
    w = eb.dequantized_weights(layer)
    if c >= w.shape[0]:
        return None
    xin = values[seg.tensor][seg.block_offset * 8:(seg.block_offset + seg.blocks) * 8][:w.shape[1]]
    terms = []
    for ky in range(layer.k):
        for kx in range(layer.k):
            yi, xi = layer.stride * y - layer.pad + ky, layer.stride * x - layer.pad + kx
            if 0 <= yi < xin.shape[1] and 0 <= xi < xin.shape[2]:
                terms += (w[c, :, ky, kx].astype(np.float64) * xin[:, yi, xi].astype(np.float64)).tolist()
    bias = float(em.to_bf16(eb.dequantized_bias(layer)[c:c + 1])[0])
    out = {"acc_exact": math.fsum(terms + [bias])}
    if layer.residual is not None:
        out["residual"] = float(values[layer.residual.tensor][layer.residual.block_offset * 8 + c, y, x])
    return out


def isolated_layers(frames: list, ir, models=em.MAC_MODELS, limit: int = 48) -> list:
    """Each layer whose inputs were all still resident when the frame ended, recomputed from the DEVICE's
    own input tensors under every candidate accumulate model and compared with the device's output.

    The chained check feeds each layer the emulator's own previous output, so one early difference
    reappears in every layer after it. Fed the device's inputs, a layer that still differs differs in
    its own arithmetic, or in something the emulator assumes about it. A model that reproduces the
    device where the one in force does not is a candidate, not a verdict: the candidates agree on
    almost all data and were told apart only by a probe built for it. A layer whose input was
    overwritten before the frame ended cannot be isolated, and is named rather than skipped.
    """
    reports, seen = [], set()
    for f in frames:
        key = f["staged"].tobytes()
        if key in seen:
            continue
        seen.add(key)
        dev = f["tensors"]
        values = {n: _values(b, ir.tensors[n].blocks) for n, b in dev.items()}
        rep = {"label": f["label"], "layers": []}
        for L in ir.layers:
            if L.output not in dev:
                continue
            needs = [s.tensor for s in L.inputs] + ([L.residual.tensor] if L.residual is not None else [])
            missing = [n for n in needs if n not in dev]
            if missing:
                rep["layers"].append({"layer": L.name, "isolated": False, "inputs_not_resident": missing})
                continue
            got = dev[L.output]
            t0 = time.perf_counter()
            outs = {m: em.bf16_bits(gr16.direct_layer(ir, L, values, mac_model=m)[:got.shape[0]]) for m in models}
            miss = np.stack([outs[m] != got for m in models])
            where = np.argwhere(miss.any(axis=0))
            t = ir.tensors[L.output]
            ys, xs = tile_origins(t.height, em.TILE_ROWS), tile_origins(t.width, em.TILE_COLS)
            rows = []
            for c, y, x in where[:limit]:
                c, y, x = int(c), int(y), int(x)
                oy, ox = _origin(ys, y, em.TILE_ROWS), _origin(xs, x, em.TILE_COLS)
                d = int(got[c, y, x])
                rows.append({"c": c, "y": y, "x": x, "tile_origin": [oy, ox], "in_tile": [y - oy, x - ox],
                             # tiles covering the position, rows x columns: 2 is an edge tile's overlap
                             "covered_by": [sum(o <= y < o + em.TILE_ROWS for o in ys),
                                            sum(o <= x < o + em.TILE_COLS for o in xs)],
                             "device": f"{d:04x}", "device_value": float(em.from_bf16_bits(np.array([d], np.uint16))[0]),
                             **{m: f"{int(outs[m][c, y, x]):04x}" for m in models},
                             **(exact_value(L, values, c, y, x) or {})})
            rep["layers"].append({
                "layer": L.name, "output": L.output, "isolated": True, "values": int(got.size),
                "differ_by_model": {m: int(miss[i].sum()) for i, m in enumerate(models)},
                "positions_any_model_differs": int(len(where)),
                "positions_no_model_matches": int(miss.all(axis=0).sum()),
                "seconds": round(time.perf_counter() - t0, 1), "positions": rows})
        reports.append(rep)
    return reports


# --------------------------------------------------------------------------- 2. timing
def time_container(path: Path, bf16: bool, tile: np.ndarray, frames: int, warmup: int, timeout_ms: int,
                   native_host: bool = True) -> dict:
    s = open_session(path, bf16, native_host)
    paths = set()
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
            if bf16:
                paths.add((s.stage_path, s.egress_path))
    finally:
        close_session(s)
    a = np.array(rows)
    out = {"frames": frames, "warmup": warmup, "stage_ms": stats(a[:, 0]), "dispatch_ms": stats(a[:, 1]),
           "readback_ms": stats(a[:, 2]), "postprocess_ms": stats(a[:, 3]), "tile_total_ms": stats(a[:, 4])}
    if bf16:
        # Which host path ran, every frame: a native arm that fell back to numpy would time the wrong thing.
        out["host_paths"] = sorted("/".join(p) for p in paths)
    return out


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
    ap.add_argument("--bf16-host", action="append", choices=("native", "numpy"), default=None,
                    help="host paths the bf16 container is timed with, in order (default: native, numpy)")
    ap.add_argument("--timeout-ms", type=int, default=10000)
    ap.add_argument("--mac-model", default=None,
                    help="the emulator's accumulate model (default: engine_bf16_emulator.MAC_MODEL, the one in force)")
    ap.add_argument("--no-exact", action="store_true")
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--no-quality", action="store_true")
    ap.add_argument("--dump-frames", type=Path, default=None,
                    help="also write the exactness frames here, as npz, for --from-dump (megabytes: keep it "
                         "out of results/; the log records each file's hash)")
    ap.add_argument("--from-dump", type=Path, default=None,
                    help="check frames an earlier sitting dumped, against this container and QDQ model; opens "
                         "no device, and implies --no-timing --no-quality")
    ap.add_argument("--isolate", choices=("inexact", "all", "none"), default="inexact",
                    help="replay each isolable layer from the device's own inputs under every accumulate "
                         "model: on the inexact frames (default), on all of them, or not at all")
    args = ap.parse_args()
    offline = args.from_dump is not None
    if offline:
        args.no_timing = args.no_quality = True

    import onnxruntime as ort
    images = args.image or DEFAULT_IMAGES
    seeds = [0] if args.seed is None else args.seed
    datasets = args.dataset or ["Set5", "Set14"]
    bf16_hosts = args.bf16_host or ["native", "numpy"]
    native = native_library()
    if "native" in bf16_hosts and not (native["bf16_ingress"] and native["bf16_egress"]):
        raise SystemExit(f"a native bf16 host arm was asked for and the loaded library cannot serve it: {native}")
    bf16_facts = container_facts(args.container)
    if bf16_facts["elem"] != ELEM_BF16:
        raise SystemExit(f"{rel(args.container)} is not a bf16 container ({bf16_facts['elem']})")
    emit("BF16_SR_SETUP", {
        "host": platform.node(), "bf16": bf16_facts,
        "int8": container_facts(args.int8_container) if args.int8_container else None,
        "qdq": {"path": rel(args.qdq), "sha256": sha16(args.qdq)},
        "fp32": {"path": rel(args.fp32), "sha256": sha16(args.fp32)},
        "npu_power_mode": None if offline else npu_power.read_mode(), "xrt": None if offline else xrt_host(),
        "onnxruntime": ort.__version__, "numpy": np.__version__, "cv2": cv2.__version__,
        # The int8 session takes its native host paths whenever this library loaded; the bf16 session
        # takes its own when they are exported and native_host is on. Each arm reports what ran.
        "native_library": native,
        "mac_model": args.mac_model or em.MAC_MODEL, "provenance": provenance(),
        "from_dump": rel(args.from_dump) if offline else None, "isolate": args.isolate,
        "images": [rel(p) for p in images], "seeds": seeds, "datasets": datasets,
        "frames": args.frames, "warmup": args.warmup, "rounds": args.rounds, "bf16_hosts": bf16_hosts})
    if not offline and not witness("start"):
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
    # What a dump is bound to: a replay against any other container or model is refused.
    dump_facts = {"container_sha256": bf16_facts["sha256"], "qdq_sha256": sha16(args.qdq)}
    failed = False
    frames, npu_arms, timing, host_agreement = [], {}, [], {}
    try:
        # ---- everything that needs the device, one context at a time
        if not args.no_exact and not offline:
            plan = [(p.stem, "image", cv2.imread(str(p))) for p in images]
            plan += [(f"seed{k}", "bits", em.bf16_bits(em.to_bf16(wo.seeded_input(args.qdq, k)[0]))) for k in seeds]
            plan += [(f"{images[0].stem}_again", "image", cv2.imread(str(images[0])))]
            s = open_session(args.container, True, bf16_hosts[0] == "native")
            try:
                frames = device_frames(s, plan, args.timeout_ms)
            finally:
                close_session(s)
            emit("BF16_SR_DEVICE_FRAMES", {"frames": [f["label"] for f in frames],
                                            "dispatch_ms": [round(f["dispatch_ms"], 3) for f in frames]})
            if args.dump_frames is not None:
                files = dump_frames(frames, args.dump_frames, dump_facts)
                emit("BF16_SR_DUMP", {"dir": rel(args.dump_frames), **dump_facts, "files": files})
        if not args.no_timing:
            arms = ([("int8_npu", args.int8_container, False, None)] if args.int8_container else []) + \
                   [(f"bf16_npu_{h}_host", args.container, True, h) for h in bf16_hosts]
            for rnd in range(args.rounds):
                for name, path, is_bf16, h in arms:
                    t = time_container(path, is_bf16, tile, args.frames, args.warmup, args.timeout_ms, h != "numpy")
                    timing.append((h, t))
                    emit("BF16_SR_TIMING", {"arm": name, "round": rnd, "span": "one 256x256 tile", **t})
        if not args.no_quality:
            for name, path, is_bf16 in ([("int8_npu", args.int8_container, False)] if args.int8_container else []) + \
                    [("bf16_npu", args.container, True)]:
                s = open_session(path, is_bf16, bf16_hosts[0] == "native")
                agree = {"tiles": 0, "identical": 0, "host_paths": set()}

                def run_tile(t, s=s, is_bf16=is_bf16, agree=agree):
                    image = s.run(t)[0]
                    if is_bf16:
                        # The same device output again through numpy's egress: host only, no dispatch.
                        agree["tiles"] += 1
                        agree["identical"] += bool(np.array_equal(image, numpy_image(s, s.read_output())))
                        agree["host_paths"].add(f"{s.stage_path}/{s.egress_path}")
                    return image
                try:
                    npu_arms[name] = upscale_all(args.data, datasets, run_tile, args.overlap)
                finally:
                    close_session(s)
                if is_bf16:
                    host_agreement = dict(agree, host_paths=sorted(agree["host_paths"]))
                    emit("BF16_SR_HOST_AGREEMENT", {"arm": name, **host_agreement})
    except Exception as exc:  # noqa: BLE001 - a device failure is the finding; say it and stop
        emit("BF16_SR_ERROR", {"type": type(exc).__name__, "message": str(exc)[:500]})
        failed = True
    idle = None if offline else witness("end")
    if failed:
        return 3
    if offline and not args.no_exact:
        frames = load_frames(args.from_dump, dump_facts)
        emit("BF16_SR_FROM_DUMP", {"dir": rel(args.from_dump), **dump_facts, "frames": [f["label"] for f in frames]})

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
    isolated = {}
    if frames and args.isolate != "none":
        inexact = {r["label"] for r in fresh if not r["exact"]}
        for rep in isolated_layers([f for f in frames if args.isolate == "all" or f["label"] in inexact], ir):
            emit("BF16_SR_ISOLATED", rep)
            for L in rep["layers"]:
                if L["isolated"]:
                    tot = isolated.setdefault(L["layer"], {m: 0 for m in L["differ_by_model"]})
                    for m, n in L["differ_by_model"].items():
                        tot[m] += n
    exact = bool(fresh) and all(r["exact"] and r["staged_is_the_compilers_plane"] for r in fresh) and \
        all(r["repeat_tail_identical"] and r["repeat_tensors_identical"] and r["input_region_intact"] for r in repeats)
    # The host ends: on the exactness frames, in every timed bf16 arm, and on every quality tile.
    want = bf16_hosts[0]
    frame_host = [r["host"] for r in reports]
    host = {
        "frames_egress_equal_numpy": sum(bool(h.get("native_egress_equals_numpy")) for h in frame_host),
        "frames": len(frame_host),
        "frames_staged_npu_sesr_preprocess": sum(bool(h.get("staged_is_npu_sesr_preprocess")) for h in frame_host),
        "image_frames": sum("staged_is_npu_sesr_preprocess" in h for h in frame_host),
        "frames_on_the_asked_path": sum(h.get("egress_path") == want and h.get("stage_path", want) == want
                                        for h in frame_host),
        "timed_arms_on_their_path": all(t["host_paths"] == [f"{h}/{h}"] for h, t in timing if h is not None),
        "quality_tiles_identical": host_agreement.get("identical"), "quality_tiles": host_agreement.get("tiles"),
        "quality_host_paths": host_agreement.get("host_paths")}
    host_ok = (host["frames_egress_equal_numpy"] == host["frames"] == host["frames_on_the_asked_path"]
               and host["frames_staged_npu_sesr_preprocess"] == host["image_frames"]
               and host["timed_arms_on_their_path"]
               and host["quality_tiles_identical"] == host["quality_tiles"]
               and host["quality_host_paths"] in (None, [f"{want}/{want}"]))
    emit("BF16_SR_RESULT", {
        "device_idle_after": idle, "from_dump": rel(args.from_dump) if offline else None,
        # per isolable layer, the values each accumulate model got wrong, summed over the replayed frames
        "isolated_differ_by_model": isolated or None,
        "exact_inputs": len(fresh), "repeat_frames": len(repeats), "device_equals_emulator": exact if frames else None,
        "tail_values_compared": int(sum(r["tail"]["values"] for r in fresh)),
        "resident_values_compared": int(sum(r["resident_tensors"]["values"] for r in fresh)),
        "worst_vs_oracle": max(fresh, key=lambda r: r["vs_oracle"]["differ"])["vs_oracle"] if fresh else None,
        "host": host, "host_ok": host_ok})
    return 0 if (exact or not frames) and (offline or idle) and host_ok else 1


if __name__ == "__main__":
    sys.exit(main())

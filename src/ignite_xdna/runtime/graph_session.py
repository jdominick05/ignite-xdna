"""Execute a graph-engine ``.ignite`` container on the Phoenix NPU.

The container's instruction stream drives the 16-core convolution engine
through every layer; activations live in one host-visible workspace buffer
in the channel-blocked layout the compiler planned. ``EngineSession`` holds
what every graph container needs (xclbin, instruction, workspace and packet
buffer objects, the halo image, one reusable XRT run, dispatch, tensor
readback); the manifest's ``task`` picks the session on top of it:

* ``GraphSession`` (``detect``, whole YOLOv8 on the NPU): per frame it stages
  the quantized image into the input tensor, dispatches once, reads the six
  head tensors back and presents them through the ``head_layout`` contract of
  ``runtime/heads.py`` as int8 NCHW views of an egress buffer, so
  ``YoloPipeline.predict_sync(use_oracle_for_boxes=False)`` decodes them with
  ``head_source == "npu"``. A ``pose`` container (YOLOv8-pose) opens in the same session, which then reads
  nine heads (one person class, plus the 51-channel keypoint heads) for ``PosePipeline`` to decode.
* ``DenseGraphSession`` (``super_resolution``, SESR): stages the resized RGB
  frame, dispatches, reads the dense output tensor back and applies the
  manifest's ``dense_output`` transform (DepthToSpace and dequantization)
  into an upscaled BGR image. ``Bf16DenseGraphSession`` is the same session
  for a bf16-engine container (W8A16: activations as bf16 in real units), and
  ``sr_session_class`` picks between the two by the width the placements declare.

A container whose manifest lists ``graph_engine.segments`` (YOLO11's C2PSA block, or YOLO-World v2's four
text attention cores, on the host) runs them in order in ``dispatch``: each NPU segment is its own
instruction stream and XRT run over the shared workspace, and each host segment is a ``HostStep`` that
reads its input tensor from the workspace (or the block ranges of a Concat view over several tensors),
runs the layer's ONNX model on ONNX Runtime's CPU provider and writes the output tensor back before the
next NPU segment reads it. ``set_host_constants`` replaces initializers of those host models at run time
(YOLO-World's text guides: another vocabulary without a new container).

All buffers are allocated once at construction.
"""
from __future__ import annotations

import hashlib
import mmap
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ignite_xdna.compiler.engine_bf16_emulator import bf16_bits, from_bf16_bits, to_bf16
from ignite_xdna.compiler.graph_ir import place_host_output
from ignite_xdna.compiler.serializer import (ELEM_BF16, ELEM_INT, ENGINE_CONV_INT8, GRAPH_ENGINES,
                                             IgniteModelReader)
from ignite_xdna.runtime.driver import XrtSiliconHarness, get_repo_root, setup_xrt_environment
from ignite_xdna.runtime.heads import HEAD_NAMES, POSE_HEAD_NAMES, HeadStatus, resolve_head_layout

ENGINE_NAME = ENGINE_CONV_INT8
ZP = 128


def is_graph_container(manifest: Optional[Dict[str, Any]]) -> bool:
    """Whether a manifest names ANY graph engine; ``container_elem`` says what width it carries."""
    return bool(manifest) and manifest.get("engine") in GRAPH_ENGINES


def container_elem(manifest: Optional[Dict[str, Any]]) -> str:
    """The activation element semantics this container's placements declare.

    Absent means ``ELEM_INT``, so a container built before the bf16 engine reads as integer
    without being rebuilt. A container that mixes widths is not something one workspace can
    stage, so it is refused here rather than half-read.
    """
    ge = (manifest or {}).get("graph_engine") or {}
    elems = {str(p.get("elem", ELEM_INT)) for p in (ge.get("placements") or {}).values()}
    if len(elems) > 1:
        raise ValueError(f"container mixes activation element widths {sorted(elems)}; "
                         "one container carries one width")
    return elems.pop() if elems else ELEM_INT


def placement_dtype(p: Dict[str, Any]) -> np.dtype:
    """The numpy dtype a placement's bytes are STORED as.

    Required, not defaulted. Falling back to uint8 on a missing key is exactly how a two-byte
    workspace gets read at half length with nothing raising: the halved byte count equals the
    element count, so every reshape succeeds and the tensor merely comes back wrong. Every
    container ignite-compile emits carries the key.

    This says how WIDE an element is, not what it means. What it means is the placement's
    ``elem`` (see ``container_elem``), and the two are separate because np.dtype("bf16") raises
    outright while np.dtype("bfloat16") resolves only if ml_dtypes was imported first.
    """
    dtype = p.get("dtype")
    if not dtype:
        raise KeyError("placement has no 'dtype': a graph-engine container has to say how wide "
                       "its elements are, because a region length computed without it is right "
                       "only at one byte an element")
    return np.dtype(dtype)


def plane_bytes(p: Dict[str, Any]) -> int:
    """Bytes of ONE channel-block plane, halo ring included, in the placement's own dtype."""
    h, w, halo = int(p["height"]), int(p["width"]), int(p["halo"])
    return (h + 2 * halo) * (w + 2 * halo) * 8 * placement_dtype(p).itemsize


def region_bytes(p: Dict[str, Any], planes: Optional[int] = None) -> int:
    """Bytes spanned by ``planes`` planes of a placement; by default its real channel blocks.

    The eight inside ``plane_bytes`` is CHANNELS A BLOCK, not a width in bytes. Both meanings of
    ``* 8`` live in this file - ``blocks * 8`` is a channel count in a dozen places - and
    conflating them is what made the original region arithmetic look correct at one byte.
    """
    n = int(p["blocks"]) if planes is None else int(planes)
    return n * plane_bytes(p)


def _plane_view(buf: np.ndarray, p: Dict[str, Any], planes: Optional[int] = None) -> np.ndarray:
    h, w, halo = p["height"], p["width"], p["halo"]
    n = p["planes"] if planes is None else planes
    pb = plane_bytes(p)
    dt = placement_dtype(p)
    band = p.get("band_rows", 0)
    if band:
        # Channel blocks interleaved every `band` rows (compiler's Placement.band_rows), so the
        # blocks of one tile sit adjacent. Always halo 0. This copies; it is a read path.
        total = p["planes"] * pb
        v = buf[p["base"]:p["base"] + total].view(dt).reshape(h // band, p["planes"], band, w, 8)
        return np.ascontiguousarray(v[:, :n].transpose(1, 0, 2, 3, 4)).reshape(n, h, w, 8)
    return buf[p["base"]:p["base"] + n * pb].view(dt).reshape(n, h + 2 * halo, w + 2 * halo, 8)


def halo_fill_image(ge: Dict[str, Any]) -> np.ndarray:
    """Workspace image with every halo ring set (the compiler's ``Workspace.halo_fill``)."""
    ws = np.zeros(int(ge["workspace_bytes"]), dtype=np.uint8)
    for p in ge["placements"].values():
        if p["halo"] == 0:
            continue
        planes = _plane_view(ws, p)
        v = p.get("halo_value", ZP)
        planes[:, :p["halo"], :, :] = v
        planes[:, -p["halo"]:, :, :] = v
        planes[:, :, :p["halo"], :] = v
        planes[:, :, -p["halo"]:, :] = v
    return ws


def input_lut(input_scale: float, input_zero_point: int = ZP) -> np.ndarray:
    """int8 preprocessor value (pixel - 128) -> uint8 model input, as the model's QuantizeLinear."""
    v = np.arange(-128, 128, dtype=np.int64)
    pixel = (v + 128).astype(np.float64) / 255.0
    q = np.clip(np.round(pixel / input_scale).astype(np.int64) + input_zero_point, 0, 255).astype(np.uint8)
    return q  # index with (int8_value + 128)


def _region(p: Dict[str, Any]) -> Tuple[int, int]:
    """(base, bytes) of a placement's real channel blocks, halo ring included."""
    return int(p["base"]), region_bytes(p)


def _boundary_placement(session: "EngineSession", binding: Dict[str, Any]) -> Dict[str, Any]:
    """The workspace placement a named boundary lives in, refusing a layout this cannot marshal."""
    p = session.ge["placements"][binding["tensor"]]
    band = int(p.get("band_rows") or 0)
    if band:
        raise ValueError(
            f"{binding['name']}: tensor {binding['tensor']} is band-packed (band_rows {band}), and boundary "
            "marshalling only reads the plane-major layout. A boundary is read by a host step rather than by a "
            "five-row tile reader, so it should never have been banded - fix the layout, do not unpack it here.")
    return p


def _boundary_region(p: Dict[str, Any]) -> Tuple[int, int]:
    """(base, bytes) of a boundary's region. Same arithmetic as ``_region`` - it always was,
    except that this one threaded itemsize and that one did not. One helper now serves both.
    """
    return int(p["base"]), region_bytes(p)


def _boundary_interior(raw: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
    """[blocks][H][W][8] view of a region's real interior, the halo ring excluded."""
    h, w, halo = int(p["height"]), int(p["width"]), int(p["halo"])
    planes = raw.view(placement_dtype(p)).reshape(
        int(p["blocks"]), h + 2 * halo, w + 2 * halo, 8)
    return planes[:, halo:halo + h, halo:halo + w, :]


def write_boundary(session: "EngineSession", binding: Dict[str, Any], value: np.ndarray) -> None:
    """Put a named tensor where the container's next step reads it from.

    A ``storage: host`` boundary never reaches the device: it lives in ``session._host_values`` for the frame,
    so two host steps either side of a CPU-only region hand values over without paying a round trip. Everything
    else is marshalled into the workspace's plane-packed ``[blocks][H+2*halo][W+2*halo][8]`` layout.

    Only the real channel lanes are written. The halo ring, and the padding lanes above ``channels`` in the last
    block, keep whatever the halo image put there - the convolutions read them, so overwriting them with a
    plausible-looking zero would change results rather than raise.
    """
    want = np.dtype(binding["dtype"])
    value = np.asarray(value)
    if value.dtype != want:
        raise ValueError(f"{binding['name']}: expected {want} at this boundary, got {value.dtype}")
    if binding.get("storage") == "host":
        session._host_values[binding["name"]] = value.copy()
        return
    p = _boundary_placement(session, binding)
    base, nbytes = _boundary_region(p)
    h, w = int(p["height"]), int(p["width"])
    chw = value.reshape(-1, h, w)
    mapped = session._ws_map is not None
    if mapped:
        raw = session._ws_map[base:base + nbytes]
    else:
        # Read-modify-write: the halo ring is already on the device and a blind write would flatten it.
        raw = np.frombuffer(session.bo_ws.read(nbytes, base), dtype=np.uint8).copy()
    interior = _boundary_interior(raw, p)
    blocks, rem = divmod(int(chw.shape[0]), 8)
    if blocks:
        interior[:blocks] = np.transpose(chw[:blocks * 8].reshape(blocks, 8, h, w), (0, 2, 3, 1))
    if rem:
        interior[blocks, :, :, :rem] = np.transpose(chw[blocks * 8:blocks * 8 + rem], (1, 2, 0))
    if not mapped:
        session.bo_ws.write(raw, base)
    session.bo_ws.sync(session.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, nbytes, base)


def read_boundary(session: "EngineSession", binding: Dict[str, Any]) -> np.ndarray:
    """Read a named tensor back out, in the binding's own NCHW shape.

    A ``storage: host`` boundary raises ``KeyError`` once the frame that produced it has been cleared, which is
    the point: a stale value from the previous frame is indistinguishable from a fresh one in the output.
    """
    if binding.get("storage") == "host":
        return session._host_values[binding["name"]]
    p = _boundary_placement(session, binding)
    base, nbytes = _boundary_region(p)
    h, w = int(p["height"]), int(p["width"])
    session.bo_ws.sync(session.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, nbytes, base)
    if session._ws_map is not None:
        raw = session._ws_map[base:base + nbytes]
    else:
        raw = np.frombuffer(session.bo_ws.read(nbytes, base), dtype=np.uint8)
    interior = _boundary_interior(raw, p)
    shape = [int(d) for d in binding["shape"]]
    channels = shape[-3] if len(shape) >= 3 else int(p["channels"])
    # A copy, not a view: the workspace slot this came from is reused by a later region in the same frame.
    chw = np.transpose(interior, (0, 3, 1, 2)).reshape(-1, h, w)[:channels].copy()
    return chw.reshape(shape)


class HostStep:
    """One host layer between two NPU segments.

    ``run`` syncs the input tensor from the workspace, runs the layer's extracted uint8 -> uint8 ONNX model
    on ONNX Runtime's CPU provider and writes the result into the output tensor's interior (its halo ring
    keeps the value the halo image set), then syncs that region to the device.
    """

    def __init__(self, session: "EngineSession", seg: Dict[str, Any]):
        import onnxruntime as ort
        blob = session._reader.get_blob_bytes(seg["blob"])
        want = seg.get("onnx_sha256")
        if want and hashlib.sha256(blob).hexdigest() != want:
            raise ValueError(f"{session.path}: host model {seg['blob']} does not match the manifest's sha256")
        self.name = seg.get("name", seg["blob"])
        self.session = session
        self.blob = blob
        from ignite_xdna.pipelines.power import ort_session_options  # the power mode covers host segments too
        self.ort_session = ort.InferenceSession(blob, ort_session_options(), providers=["CPUExecutionProvider"])
        self.input_name = self.ort_session.get_inputs()[0].name
        self.last_ms = 0.0
        self.last_cpu_ms = 0.0        # the ONNX Runtime call alone
        self.last_transfer_ms = 0.0   # marshalling its boundaries in and out
        # Version 2 names every input and output, so a host region may take several tensors and produce
        # several, and a value that never leaves the CPU never reaches the workspace at all. Version 1 is
        # one input tensor to one output tensor and stays exactly as it was.
        self.boundary_version = int(seg.get("boundary_version") or 1)
        self.input_bindings = list(seg.get("input_bindings") or [])
        self.output_bindings = list(seg.get("output_bindings") or [])
        if self.boundary_version >= 2:
            if not self.input_bindings or not self.output_bindings:
                raise ValueError(f"{self.name}: boundary_version 2 needs input_bindings and output_bindings")
            return
        placements = session.ge["placements"]
        self.pin, self.pout = placements[seg["input"]], placements[seg["output"]]
        # The input: one whole tensor (older manifests), or the block ranges of a Concat view over several tensors.
        parts = seg.get("inputs") or [{"tensor": seg["input"], "block_offset": 0, "blocks": int(self.pin["blocks"])}]
        self.in_parts = []
        for part in parts:
            p = placements[part["tensor"]]
            h, w, halo = int(p["height"]), int(p["width"]), int(p["halo"])
            plane = plane_bytes(p)
            self.in_parts.append((int(p["base"]) + int(part["block_offset"]) * plane, int(part["blocks"]) * plane,
                                  int(part["blocks"]), h, w, halo))
        in_channels = int(seg.get("in_channels") or self.pin["channels"])
        self.in_base, self.in_bytes = _region(self.pin)
        self.out_base, self.out_bytes = _region(self.pout)
        self._x = np.empty((1, in_channels, int(self.pin["height"]), int(self.pin["width"])), dtype=np.uint8)
        self.last_ms = 0.0

    def initializer_names(self) -> List[str]:
        import onnx
        return [t.name for t in onnx.load_from_string(self.blob).graph.initializer]

    def replace_constants(self, values: Dict[str, np.ndarray]) -> List[str]:
        """Rebuild the ONNX Runtime session with the named float initializers replaced; returns the names replaced.

        A value keeps the initializer's dtype and rank; other dimensions may change (YOLO-World's text guides take
        the vocabulary size). Stored intermediate shapes are dropped so ONNX Runtime infers them again. The NPU
        segments never see these constants, so the container's program is unchanged.
        """
        import onnx
        import onnxruntime as ort
        from onnx import numpy_helper
        from ignite_xdna.pipelines.power import ort_session_options
        model = onnx.load_from_string(self.blob)
        done = []
        for t in model.graph.initializer:
            if t.name not in values:
                continue
            old = numpy_helper.to_array(t)
            new = np.asarray(values[t.name])
            if new.dtype != old.dtype or new.ndim != old.ndim:
                raise ValueError(f"{self.name}: {t.name} is {old.dtype} rank {old.ndim}, got {new.dtype} rank {new.ndim}")
            t.CopyFrom(numpy_helper.from_array(np.ascontiguousarray(new), t.name))
            done.append(t.name)
        if done:
            del model.graph.value_info[:]
            self.blob = model.SerializeToString()
            self.ort_session = ort.InferenceSession(self.blob, ort_session_options(), providers=["CPUExecutionProvider"])
            self.input_name = self.ort_session.get_inputs()[0].name
        return done

    def _run_named(self) -> None:
        """Boundary version 2: read every named input, run the region, write every named output.

        The two timings are kept apart on purpose. ``last_cpu_ms`` is the region's own arithmetic and is the
        number that decides whether a hybrid container can ever beat a stack that runs the whole graph
        elsewhere; ``last_transfer_ms`` is what moving its boundaries costs and is the part a protocol change
        could remove. Summing them hides which of the two a model is actually paying.
        """
        s = self.session
        t0 = time.perf_counter()
        feeds = {b["name"]: read_boundary(s, b) for b in self.input_bindings}
        t1 = time.perf_counter()
        outputs = self.ort_session.run([b["name"] for b in self.output_bindings], feeds)
        t2 = time.perf_counter()
        for binding, y in zip(self.output_bindings, outputs):
            write_boundary(s, binding, y)
        t3 = time.perf_counter()
        self.last_cpu_ms = (t2 - t1) * 1e3
        self.last_transfer_ms = ((t1 - t0) + (t3 - t2)) * 1e3
        self.last_ms = (t3 - t0) * 1e3

    def run(self) -> None:
        if self.boundary_version >= 2:
            self._run_named()
            return
        s = self.session
        t0 = time.perf_counter()
        d = s.harness.pyxrt.xclBOSyncDirection
        filled = 0
        for base, nbytes, blocks, h, w, halo in self.in_parts:
            s.bo_ws.sync(d.XCL_BO_SYNC_BO_FROM_DEVICE, nbytes, base)
            if s._ws_map is not None:
                raw = s._ws_map[base:base + nbytes]
            else:
                raw = np.frombuffer(s.bo_ws.read(nbytes, base), dtype=np.uint8)
            planes = raw.reshape(blocks, h + 2 * halo, w + 2 * halo, 8)[:, halo:halo + h, halo:halo + w, :]
            take = min(blocks * 8, self._x.shape[1] - filled)
            self._x[0, filled:filled + take] = np.transpose(planes, (0, 3, 1, 2)).reshape(blocks * 8, h, w)[:take]
            filled += take
        t_cpu = time.perf_counter()
        y = self.ort_session.run(None, {self.input_name: self._x})[0]
        self.last_cpu_ms = (time.perf_counter() - t_cpu) * 1e3
        q = self.pout
        oh, ow, ohalo, oblocks = int(q["height"]), int(q["width"]), int(q["halo"]), int(q["blocks"])
        if s._ws_map is not None:
            region = s._ws_map[self.out_base:self.out_base + self.out_bytes]
        else:
            region = np.frombuffer(s.bo_ws.read(self.out_bytes, self.out_base), dtype=np.uint8).copy()
        full = place_host_output(np.asarray(y, dtype=np.uint8)[0], oblocks * 8, oh, ow)
        interior = region.reshape(oblocks, oh + 2 * ohalo, ow + 2 * ohalo, 8)[:, ohalo:ohalo + oh, ohalo:ohalo + ow, :]
        interior[:] = np.transpose(full.reshape(oblocks, 8, oh, ow), (0, 2, 3, 1))
        if s._ws_map is None:
            s.bo_ws.write(region, self.out_base)
        s.bo_ws.sync(d.XCL_BO_SYNC_BO_TO_DEVICE, self.out_bytes, self.out_base)
        self.last_ms = (time.perf_counter() - t0) * 1e3
        self.last_transfer_ms = self.last_ms - self.last_cpu_ms


class EngineSession:
    """The convolution engine and its buffers for one graph-engine container (see ``compiler/engine_compile.py``)."""

    #: Activation element widths this session can stage and read back. Every region length below
    #: is an element count used as a byte count, which is correct at one byte an element and
    #: silently half-length at two, so a session states what it can read and refuses the rest.
    SUPPORTED_ELEMS = (ELEM_INT,)

    def __init__(self, container_path: Union[str, Path], device_index: int = 0,
                 xclbin_cache_dir: Optional[Union[str, Path]] = None, map_workspace: bool = True,
                 weights_path: Optional[Union[str, Path]] = None,
                 bo_ws: Optional[Any] = None, harness: Optional[Any] = None, **_ignored):
        setup_xrt_environment()
        self.path = Path(container_path)
        self.device_index = device_index
        self._closed = False
        self._reader = None
        self.harness = None
        self._owns_harness = False
        self._owns_bo_ws = False
        self._weights_file_handle = None
        self._weights_mmap = None
        self._runs: List[Any] = []
        self._host_steps: List[Any] = []
        # Boundary values that never reach the device, handed between host regions within one frame. Frame
        # scoped on purpose: whoever starts a frame clears it, so a stale value raises instead of being read.
        self._host_values: Dict[str, np.ndarray] = {}
        self._npu_streams: List[Any] = []
        self._ws_map: Optional[np.ndarray] = None
        self._input_plane: Optional[np.ndarray] = None
        # Overwritten from the input placement below when there is one; a session that never
        # stages an input still has to have a width to answer with.
        self._input_dtype: np.dtype = np.dtype(np.uint8)
        self.bo_ws = None
        self.bo_wp = None
        self.bo_instr_exec = None
        self._run = None

        try:
            self._reader = IgniteModelReader(self.path)
            self.ignite_manifest: Dict[str, Any] = self._reader.manifest
            if not is_graph_container(self.ignite_manifest):
                raise ValueError(f"{self.path} is not a graph-engine container: engine "
                                 f"{self.ignite_manifest.get('engine')!r} is not one of "
                                 f"{list(GRAPH_ENGINES)}")
            # Admission, and it belongs HERE rather than at the first read. Every region length in
            # this class is an element count used as a byte count; at two bytes an element that is
            # exactly half the region, so each reshape still succeeds and each sync still returns a
            # plausible tensor of interleaved high and low bytes. A container this session cannot
            # read has to fail now, naming the reason, not several frames later as bad pixels.
            elem = container_elem(self.ignite_manifest)
            if elem not in self.SUPPORTED_ELEMS:
                raise ValueError(
                    f"{self.path} carries {elem!r} activations but {type(self).__name__} reads "
                    f"{list(self.SUPPORTED_ELEMS)}; it needs the session built for that width")
            self.elem = elem
            self.ge: Dict[str, Any] = self.ignite_manifest["graph_engine"]
            self.task = self.ignite_manifest.get("task", "detect")
            self.monolithic_stages: Dict[str, Any] = {}
            self.out_bytes = int(self.ignite_manifest["egress_bytes"])
            self.num_cores = 16
            self.single_dispatch = True

            # pyxrt.xclbin needs a file: cache the blob by hash.
            cache_dir = Path(xclbin_cache_dir or get_repo_root() / "build" / "ignite_cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            xclbin_bytes = self._reader.get_blob_bytes("engine.xclbin")
            sha = hashlib.sha256(xclbin_bytes).hexdigest()[:16]
            self.xclbin_path = cache_dir / f"engine_{sha}.xclbin"
            if not self.xclbin_path.exists() or self.xclbin_path.stat().st_size != len(xclbin_bytes):
                self.xclbin_path.write_bytes(xclbin_bytes)

            if harness is not None:
                self.harness = harness
                self._owns_harness = False
            else:
                self.harness = XrtSiliconHarness(device_idx=device_index)
                self.harness.load_xclbin(str(self.xclbin_path), "MLIR_AIE")
                self._owns_harness = True
            # Execution plan: one instruction stream, or NPU segments with host layers between them.
            self.segments: List[Dict[str, Any]] = list(self.ge.get("segments") or [{"kind": "npu", "blob": "insts.bin"}])
            self.single_dispatch = not self.ge.get("segments")
            self._npu_streams = []
            for seg in self.segments:
                if seg["kind"] == "npu":
                    self._npu_streams.append(self.harness.create_instruction_bo_from_bytes(
                        self._reader.get_blob_memoryview(seg["blob"])))
                elif seg["kind"] != "host":
                    raise ValueError(f"{self.path}: unknown segment kind {seg['kind']!r}")
            self.bo_instr_exec, self.ninstr_exec = self._npu_streams[0]
            self.workspace_bytes = int(self.ge["workspace_bytes"])
            if bo_ws is not None:
                if bo_ws.size() < self.workspace_bytes:
                    raise ValueError(
                        f"Provided bo_ws size ({bo_ws.size()}) is smaller than required ({self.workspace_bytes})"
                    )
                self.bo_ws = bo_ws
                self._owns_bo_ws = False
            else:
                self.bo_ws = self.harness.create_host_bo(self.workspace_bytes, 3)
                self._owns_bo_ws = True
            if "wpackets.bin" in self._reader.blobs:
                wp = self._reader.get_blob_memoryview("wpackets.bin")
                self.bo_wp = self.harness.create_host_bo(max(64, len(wp)), 4)
                self.bo_wp.write(wp, 0)
                self.bo_wp.sync(self.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            else:
                resolved_weights_path = None
                if weights_path is not None:
                    resolved_weights_path = Path(weights_path)
                elif self.path is not None:
                    weights_file = self.ge.get("weights_file") or f"{self.path.stem}.weights"
                    candidate = self.path.parent / weights_file
                    if candidate.exists():
                        resolved_weights_path = candidate
                    elif self.path.with_suffix(".weights").exists():
                        resolved_weights_path = self.path.with_suffix(".weights")
                if resolved_weights_path is None or not resolved_weights_path.exists():
                    raise FileNotFoundError(
                        f"Decoupled weights sidecar not found for {self.path}. "
                        f"Expected sidecar at {resolved_weights_path or '<container>.weights'} or pass weights_path."
                    )
                with open(resolved_weights_path, "rb") as f:
                    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                        expected_sha = self.ge.get("weights_sha256")
                        if expected_sha:
                            actual_sha = hashlib.sha256(mm).hexdigest()
                            if actual_sha != expected_sha:
                                raise ValueError(
                                    f"Decoupled weights checksum mismatch: expected {expected_sha}, got {actual_sha}"
                                )
                        self.bo_wp = self.harness.create_host_bo(max(64, len(mm)), 4)
                        self.bo_wp.write(mm, 0)
                        self.bo_wp.sync(self.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            # One-time workspace image: halo rings for every tensor (only if owning bo_ws).
            if getattr(self, "_owns_bo_ws", True):
                ws_init = halo_fill_image(self.ge)
                self.bo_ws.write(ws_init, 0)
                self.bo_ws.sync(self.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

            # Input staging: the image plane (one 8-channel block with its halo ring).
            # With ``map_workspace`` (the default) the plane is a view of the mapped
            # workspace buffer object, so staging writes straight into it and readback
            # reads the mapped tensors without a host copy. Buffer object writes, syncs
            # and reads cost < 0.06 ms per frame on Phoenix.
            self.input_placement = self.ge["placements"][self.ge["input_tensor"]]
            p = self.input_placement
            plane_shape = (p["height"] + 2 * p["halo"], p["width"] + 2 * p["halo"], 8)
            # ONE plane, which is what every _upload_input sync spans. np.prod is an ELEMENT
            # count; the sync wants bytes, and the two are equal only at one byte an element.
            self._input_dtype = placement_dtype(p)
            self._input_bytes = int(np.prod(plane_shape)) * self._input_dtype.itemsize
            self._ws_map = None
            if map_workspace:
                try:
                    mapped = np.frombuffer(self.bo_ws.map(), dtype=np.uint8)
                    if mapped.size >= self.workspace_bytes:
                        self._ws_map = mapped
                except Exception:  # noqa: BLE001 - mapping is an optimisation only
                    self._ws_map = None
            if self._ws_map is not None:
                base = p["base"]
                # _ws_map is a BYTE view of the workspace, so reinterpret before reshaping:
                # at two bytes an element the old reshape would have raised, which is the one
                # place this file failed loudly rather than quietly.
                self._input_plane = (self._ws_map[base:base + self._input_bytes]
                                     .view(self._input_dtype).reshape(plane_shape))
                self._input_plane[:] = ZP
            else:
                self._input_plane = np.full(plane_shape, ZP, dtype=self._input_dtype)
            self.last_dispatch_ms = 0.0

            # One XRT run object for every frame: its arguments (opcode, instructions,
            # workspace, packets) never change, so each frame only starts and awaits it.
            pyxrt = self.harness.pyxrt
            self._completed = getattr(getattr(pyxrt, "ert_cmd_state", None), "ERT_CMD_STATE_COMPLETED", None)
            self._runs = []
            try:
                for bo_instr, n_instr in self._npu_streams:
                    run = pyxrt.run(self.harness.kernel)
                    for i, arg in enumerate((3, bo_instr, n_instr, self.bo_ws, self.bo_wp)):
                        run.set_arg(i, arg)
                    self._runs.append(run)
            except Exception:  # noqa: BLE001 - fall back to one run per dispatch
                self._runs = []
            self._run = self._runs[0] if self._runs else None
            self._host_steps = [HostStep(self, seg) for seg in self.segments if seg["kind"] == "host"]
            self.last_host_ms = 0.0
            self.last_segment_ms = []
        except Exception:
            self.close()
            raise

    # ------------------------------------------------------------------ status
    @property
    def is_monolithic(self) -> bool:
        return True

    @property
    def stage_names(self):
        return [L["name"] for L in self.ge["layers"]]

    # ------------------------------------------------------------------ frame
    def _upload_input(self) -> None:
        base = self.input_placement["base"]
        if self._ws_map is None:
            self.bo_ws.write(self._input_plane, base)
        self.bo_ws.sync(self.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, self._input_bytes, base)

    def stage_quantized(self, chw: np.ndarray) -> None:
        """Write an already-quantized uint8 [C][H][W] input tensor into the input plane and upload it."""
        p = self.input_placement
        h = int(p["halo"])
        self._input_plane[h:h + p["height"], h:h + p["width"], :chw.shape[0]] = np.moveaxis(chw, 0, -1)
        self._upload_input()

    def _dispatch_stream(self, k: int, timeout_ms: int) -> None:
        if self._runs:
            self._runs[k].start()
            state = self._runs[k].wait(timeout_ms)
        else:
            bo_instr, n_instr = self._npu_streams[k]
            _, state = self.harness.dispatch_kernel(bo_instr, n_instr, self.bo_ws, self.bo_wp, timeout_ms=timeout_ms)
        if state != self._completed and str(state) != "ert_cmd_state.ERT_CMD_STATE_COMPLETED":
            where = f" (NPU segment {k})" if len(self._npu_streams) > 1 else ""
            raise RuntimeError(f"graph engine dispatch{where} ended in state {state}")

    def dispatch(self, timeout_ms: int = 10000, max_segments: Optional[int] = None) -> float:
        """Run the container's segments in order and return the NPU dispatch milliseconds.

        ``last_dispatch_ms`` counts NPU segments only, ``last_host_ms`` the host layers between them
        (syncs and ONNX Runtime), and ``last_segment_ms`` every segment in order.
        If ``max_segments`` is provided, dispatch stops after executing that many segments (supporting
        early-exit cascades).
        """
        seg_ms: List[float] = []
        npu_k = host_k = 0
        segs = self.segments if max_segments is None else self.segments[:max_segments]
        for seg in segs:
            t0 = time.perf_counter()
            if seg["kind"] == "npu":
                self._dispatch_stream(npu_k, timeout_ms)
                npu_k += 1
            else:
                self._host_steps[host_k].run()
                host_k += 1
            seg_ms.append((time.perf_counter() - t0) * 1e3)
        self.last_segment_ms = seg_ms
        self.last_host_ms = sum(ms for ms, seg in zip(seg_ms, segs) if seg["kind"] == "host")
        self.last_dispatch_ms = sum(seg_ms) - self.last_host_ms
        return self.last_dispatch_ms

    def set_host_constants(self, values: Dict[str, np.ndarray]) -> Dict[str, List[str]]:
        """Replace named float initializers in the host segments' models ({host segment name: names replaced}).

        Only host segments run ONNX Runtime, so this changes what the CPU computes between NPU segments and never the
        NPU program. YOLO-World v2 uses it for a vocabulary chosen at run time: its text guides are initializers of
        the attention regions. Every name must exist in some host segment.
        """
        replaced = {step.name: step.replace_constants(values) for step in self._host_steps}
        missing = sorted(set(values) - {n for names in replaced.values() for n in names})
        if missing:
            raise KeyError(f"{self.path}: no host segment has initializers {missing}")
        return {k: v for k, v in replaced.items() if v}

    def read_tensor(self, name: str, sync: bool = True) -> np.ndarray:
        """Debug helper: sync one tensor from the device and return uint8 [C][H][W]."""
        p = self.ge["placements"][name]
        h, w, halo = p["height"], p["width"], p["halo"]
        band = p.get("band_rows", 0)
        if band:
            # Band-packed: channel blocks interleave every `band` rows, so a block's rows are
            # NOT contiguous and the region spans every plane of every band, not blocks planes.
            # Reading it plane-major returns the right bytes in the wrong order.
            nbytes = region_bytes(p, p["planes"])     # halo is always 0 when banded
            if sync:
                self.bo_ws.sync(self.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE,
                                nbytes, p["base"])
            raw = np.frombuffer(self.bo_ws.read(nbytes, p["base"]), dtype=np.uint8)
            v = raw.view(placement_dtype(p)).reshape(h // band, p["planes"], band, w, 8)[:, :p["blocks"]]
            return np.ascontiguousarray(v.transpose(1, 4, 0, 2, 3)).reshape(
                p["blocks"] * 8, h, w)[:p["channels"]]
        nbytes = region_bytes(p)
        if sync:
            self.bo_ws.sync(self.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, nbytes, p["base"])
        raw = np.frombuffer(self.bo_ws.read(nbytes, p["base"]), dtype=np.uint8)
        planes = raw.view(placement_dtype(p)).reshape(p["blocks"], h + 2 * halo, w + 2 * halo, 8)[:, halo:halo + h, halo:halo + w, :]
        return np.transpose(planes, (0, 3, 1, 2)).reshape(p["blocks"] * 8, h, w)[:p["channels"]]

    def stage_tensor(self, name: str, data: np.ndarray, sync: bool = True):
        """Stages an intermediate or input uint8 tensor [C, H, W] into the workspace, optionally syncing to device."""
        if name not in self.ge["placements"]:
            raise KeyError(f"{self.path}: tensor {name!r} not in placements")
        p = self.ge["placements"][name]
        h, w, halo = p["height"], p["width"], p["halo"]
        blocks = p["blocks"]
        channels = p["channels"]
        if data.shape != (channels, h, w):
            raise ValueError(f"Expected shape ({channels}, {h}, {w}), got {data.shape}")
        nbytes = region_bytes(p)
        dt = placement_dtype(p)

        padded_c = blocks * 8          # CHANNELS a block, not a width in bytes
        if channels < padded_c:
            padded_data = np.pad(data, ((0, padded_c - channels), (0, 0), (0, 0)), mode="constant", constant_values=ZP)
        else:
            padded_data = data
        swizzled = padded_data.reshape(blocks, 8, h, w).transpose(0, 2, 3, 1)
        if halo > 0:
            plane = np.full((blocks, h + 2 * halo, w + 2 * halo, 8), p.get("halo_value", ZP), dtype=dt)
            plane[:, halo:halo + h, halo:halo + w, :] = swizzled
        else:
            plane = np.ascontiguousarray(swizzled, dtype=dt)

        if p.get("band_rows", 0):
            band = p["band_rows"]
            nb = h // band
            full = np.full((nb, p["planes"], band, w, 8), p.get("halo_value", ZP), dtype=dt)
            full[:, :blocks] = swizzled.reshape(blocks, nb, band, w, 8).transpose(1, 0, 2, 3, 4)
            plane = np.ascontiguousarray(full, dtype=dt)
            nbytes = plane.nbytes          # .size is an element count, equal only at one byte
        raw_bytes = plane.tobytes()
        if self._ws_map is not None:
            self._ws_map[p["base"]:p["base"] + nbytes] = np.frombuffer(raw_bytes, dtype=np.uint8)
        else:
            self.bo_ws.write(raw_bytes, p["base"])

        if sync:
            self.bo_ws.sync(self.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, nbytes, p["base"])

    # ------------------------------------------------------------------ lifetime
    def close(self):
        if self._closed:
            return
        self._closed = True
        self._run = None  # the runs hold references to the buffer objects
        self._runs = []
        self._host_steps = []
        self._npu_streams = []
        self._ws_map = None
        self._input_plane = None
        self.bo_ws = None
        self.bo_wp = None
        self.bo_instr_exec = None
        if hasattr(self, "_weights_mmap") and self._weights_mmap is not None:
            try:
                self._weights_mmap.close()
            except Exception:
                pass
            self._weights_mmap = None
        if hasattr(self, "_weights_file_handle") and self._weights_file_handle is not None:
            try:
                self._weights_file_handle.close()
            except Exception:
                pass
            self._weights_file_handle = None
        if getattr(self, "_owns_harness", True) and self.harness is not None:
            self.harness.close()
        self.harness = None
        if self._reader is not None:
            self._reader.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class GraphSession(EngineSession):
    """Session for ``engine == conv_engine_v1`` detect and pose containers (whole YOLOv8 or YOLOv8-pose on the NPU)."""

    def __init__(self, container_path: Union[str, Path], device_index: int = 0,
                 xclbin_cache_dir: Optional[Union[str, Path]] = None, map_workspace: bool = True,
                 weights_path: Optional[Union[str, Path]] = None,
                 bo_ws: Optional[Any] = None, harness: Optional[Any] = None, **_ignored):
        super().__init__(container_path, device_index=device_index, xclbin_cache_dir=xclbin_cache_dir,
                         map_workspace=map_workspace, weights_path=weights_path,
                         bo_ws=bo_ws, harness=harness, **_ignored)
        if self.task not in ("detect", "pose"):
            task = self.task
            self.close()
            opener = {"super_resolution": "DenseGraphSession", "segment": "DenseTensorSession",
                      "matte": "DenseTensorSession"}.get(task)
            raise ValueError(f"{self.path} is a {task} container; open it with "
                             f"{opener or 'the session for that task'}, or let "
                             "InferenceSession.from_file pick one")
        self.head_names = POSE_HEAD_NAMES if self.task == "pose" else HEAD_NAMES
        self.in_bytes = 3 * 640 * 640
        p = self.input_placement
        # With the mapped workspace, ``stage_image`` has the native preprocessor letterbox,
        # resize and quantize a camera frame straight into the input plane; the numpy lookup
        # and transpose of ``stage_input`` cost 3.5 ms.
        self._input_lut = input_lut(float(self.ignite_manifest["quant_scales"]["input_scale"]),
                                    int(self.ignite_manifest["quant_scales"].get("input_zero_point", ZP)))
        try:
            from ignite_xdna.pipelines.preprocess import FusedPreprocessor, blocks_class_max_int8, blocks_to_nchw_int8
            self._pre: Optional[Any] = FusedPreprocessor(imgsz=int(p["width"]))
            self._to_nchw = blocks_to_nchw_int8
            self._class_max = blocks_class_max_int8
        except Exception:  # noqa: BLE001 - native helpers are optional
            self._pre, self._to_nchw, self._class_max = None, None, None
        self._direct_ingress = bool(self._pre is not None and self._pre.has_plane_ingress)
        # Per-frame ingress call with everything but the frame bound once.
        if self._direct_ingress:
            import ctypes
            self._lut_c = np.ascontiguousarray(self._input_lut, dtype=np.uint8)
            self._c_top, self._c_left, self._c_scale = ctypes.c_int(), ctypes.c_int(), ctypes.c_float()
            self._c_refs = (ctypes.byref(self._c_top), ctypes.byref(self._c_left), ctypes.byref(self._c_scale))
            self._ingress_fn = self._pre.lib.fused_preprocess_bgr_to_c8_plane
        # Head readback: each head tensor is contiguous (halo 0); egress is int8 NCHW.
        self.heads_meta = self.ge["heads"]
        # out_bytes is a BYTE count - the manifest's egress_bytes, which threads itemsize - while
        # every head slice below indexes this buffer by ELEMENT (`off + c * height * width`).
        # The two coincide only while the egress is one byte an element, which int8 heads are.
        # A wider egress would allocate twice the elements and land every head at half its
        # declared offset, overlapping its neighbour. DetectHeadLayout.unpack refuses a non-int8
        # egress outright, which is what keeps that from being discovered in a detection.
        self._egress = np.zeros(self.out_bytes, dtype=np.int8)
        self._head_regions = []
        for name in self.head_names:
            hm = self.heads_meta[name]
            hp = self.ge["placements"][hm["tensor"]]
            nbytes = region_bytes(hp)          # heads are planned without a halo
            self._head_regions.append((name, hm, hp, hp["base"], nbytes))
        # Consolidated head sync spans: merge contiguous/overlapping regions
        # to reduce ioctl roundtrips from 6-9 down to 3.
        ranges = sorted((base, base + nbytes) for _, _, _, base, nbytes in self._head_regions)
        merged = []
        for b, e in ranges:
            if not merged or b > merged[-1][1]:
                merged.append([b, e])
            else:
                merged[-1][1] = max(merged[-1][1], e)
        self._head_sync_spans = [(b, e - b) for b, e in merged]
        self._raw_heads: Dict[str, np.ndarray] = {}
        self._raw_c8_views: Optional[Dict[str, np.ndarray]] = None
        # Per-anchor class-logit maxima of the class heads (int8, zero point 0), filled natively
        # during readback so the decoder's confidence prune does not scan the class tensors. A pose score
        # head has one channel, so the pose decoder prunes on it directly.
        self._cls_max: Dict[str, np.ndarray] = {
            name: np.empty(self.ge["placements"][self.heads_meta[name]["tensor"]]["height"]
                           * self.ge["placements"][self.heads_meta[name]["tensor"]]["width"], dtype=np.int8)
            for name in self.head_names if name.endswith("_cls") and self.task == "detect"}
        self._cls_max_valid = False
        self._head_status: Optional[HeadStatus] = None
        self._head_absence_warned = False
        self._head_views: Optional[Dict[str, np.ndarray]] = None  # int8 views of the persistent egress
        self._head_scales: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ status
    @property
    def head_status(self) -> HeadStatus:
        if self._head_status is None:
            self._head_status = resolve_head_layout(self.ignite_manifest, self.out_bytes, self.head_names)
        return self._head_status

    # ------------------------------------------------------------------ frame
    @property
    def direct_ingress(self) -> bool:
        """True when ``stage_image`` can quantize camera frames straight into the input plane."""
        return self._direct_ingress

    def stage_image(self, img_bgr: np.ndarray) -> Tuple[Tuple[int, int], float]:
        """Letterbox, resize and quantize a BGR frame into the input plane and upload it.

        One native pass writes ``lut[pixel]`` into channels 0..2 of the plane (a view
        of the mapped workspace buffer object when mapping is available), so no
        int8 tensor, lookup pass or transpose exists on the host. Returns the
        letterbox ``(pad, scale)`` for box decoding. The plane is byte-identical to
        ``stage_input`` of the preprocessor's int8 tensor for the same frame.
        """
        if not self._direct_ingress:
            raise RuntimeError("direct ingress needs the native preprocessor with plane ingress")
        p = self.input_placement
        if not img_bgr.flags["C_CONTIGUOUS"]:
            img_bgr = np.ascontiguousarray(img_bgr)
        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3 or img_bgr.dtype != np.uint8:
            raise ValueError(f"stage_image expects an HxWx3 uint8 BGR frame, got {img_bgr.shape} {img_bgr.dtype}")
        ret = self._ingress_fn(img_bgr.ctypes.data, img_bgr.shape[1], img_bgr.shape[0], img_bgr.strides[0],
                               self._input_plane.ctypes.data, int(p["width"]), int(p["height"]), int(p["halo"]),
                               self._lut_c.ctypes.data, *self._c_refs)
        if ret != 0:
            raise RuntimeError(f"fused_preprocess_bgr_to_c8_plane returned {ret}")
        pad, scale = (self._c_top.value, self._c_left.value), self._c_scale.value
        self._upload_input()
        return pad, scale

    def stage_input(self, input_tensor: Any) -> None:
        """Quantize the preprocessor's int8 (1,3,640,640) tensor into the input plane and upload it."""
        x = np.asarray(input_tensor)
        if x.dtype != np.int8:
            x = np.clip(np.asarray(x, dtype=np.int64), -128, 127).astype(np.int8)
        chw = x.reshape(3, self.input_placement["height"], self.input_placement["width"])
        # LUT index is value + 128; the int8 bit pattern viewed as uint8 is (value + 128) ^ 0x80.
        q = self._input_lut[chw.view(np.uint8) ^ 0x80]            # uint8 [3][H][W]
        h = self.input_placement["halo"]
        self._input_plane[h:-h or None, h:-h or None, :3] = np.moveaxis(q, 0, -1)
        self._upload_input()

    def read_heads(self, unswizzle: bool = True) -> np.ndarray:
        """Sync the head tensors back and assemble the int8 NCHW egress buffer (or raw C8 views)."""
        d = self.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        cls_ok = self._class_max is not None
        for base, nbytes in self._head_sync_spans:
            self.bo_ws.sync(d, nbytes, base)
        for name, hm, hp, base, nbytes in self._head_regions:
            if self._ws_map is not None:
                raw = self._ws_map[base:base + nbytes]
            else:
                raw = np.frombuffer(self.bo_ws.read(nbytes, base), dtype=np.uint8)
            self._raw_heads[name] = raw
            c = hm["channels"]
            if name in self._cls_max:
                cls_ok = cls_ok and self._class_max(raw, hp["blocks"], hp["height"], hp["width"], c,
                                                    self._cls_max[name])
            if not unswizzle:
                continue
            off = hm["egress_offset"]
            dst = self._egress[off:off + c * hp["height"] * hp["width"]]
            # uint8 with zero point 128 -> int8 with zero point 0 (flip the top bit), NHWC blocks -> NCHW
            if self._to_nchw is not None and self._to_nchw(raw, hp["blocks"], hp["height"], hp["width"], c, dst):
                continue
            blocked = raw.reshape(hp["blocks"], hp["height"], hp["width"], 8)
            chw = np.transpose(blocked, (0, 3, 1, 2)).reshape(hp["blocks"] * 8, hp["height"], hp["width"])
            dst[:] = (chw[:c] ^ 0x80).view(np.int8).reshape(-1)
        self._cls_max_valid = bool(cls_ok)
        return self._egress

    def run_yolo_monolithic(self, input_tensor: Any, unswizzle: bool = True, timeout_ms: int = 10000,
                            return_timestamps: bool = False, max_segments: Optional[int] = None):
        """Whole-network forward pass; returns the ``run_yolo_monolithic`` head dict of InferenceSession.

        ``input_tensor=None`` dispatches on the input plane already staged by ``stage_image``.
        ``unswizzle=False`` skips the NCHW egress transpose and provides raw channel-blocked heads.
        ``max_segments`` limits dispatch to the first N segments.
        """
        t0 = time.perf_counter()
        if input_tensor is not None:
            self.stage_input(input_tensor)
        t1 = time.perf_counter()
        self.dispatch(timeout_ms=timeout_ms, max_segments=max_segments)
        t2 = time.perf_counter()
        egress = self.read_heads(unswizzle=unswizzle)
        t3 = time.perf_counter()
        status = self.head_status
        if not status.present and not self._head_absence_warned:
            # Absent heads are a SOFT failure by design: every head comes back None, a caller
            # with a CPU oracle falls back to it, and the only symptom is a worse latency with
            # no stated cause. The reason is in out['head_status'] either way; this says it once
            # out loud so it is not conditional on somebody reading that key.
            self._head_absence_warned = True
            warnings.warn(f"{self.path}: no usable head layout, so every NPU head is None and a caller "
                          f"that falls back will look merely slow - {status.reason}",
                          RuntimeWarning, stacklevel=2)
        out: Dict[str, Any] = {name: None for name in self.head_names}
        if status.present:
            if unswizzle:
                if self._head_views is None:
                    # The egress buffer is allocated once, so its head views and scales are too.
                    self._head_views = status.layout.unpack(egress)
                    self._head_scales = status.layout.scales()
                out.update(self._head_views)
                out["scales"] = self._head_scales
            else:
                if self._raw_c8_views is None:
                    self._head_scales = status.layout.scales()
                    self._raw_c8_views = {}
                for name in self.head_names:
                    hm = self.heads_meta[name]
                    hp = self.ge["placements"][hm["tensor"]]
                    self._raw_c8_views[name] = self._raw_heads[name].reshape(hp["blocks"], hp["height"] * hp["width"], 8)
                out.update(self._raw_c8_views)
                out["scales"] = self._head_scales
            if self._cls_max_valid:
                out["cls_max"] = self._cls_max  # {p*_cls: int8 per-anchor class maxima}, see YoloDecoder
        out["heads_present"] = status.present
        out["head_status"] = status.reason
        out["raw_output"] = egress
        out["raw_heads"] = egress
        out["unswizzled"] = unswizzle
        # npu_ms is the NPU segments' dispatch; host_ms the host layers between them (0 without any).
        timestamps = {"stage_ms": (t1 - t0) * 1e3, "npu_ms": self.last_dispatch_ms, "host_ms": self.last_host_ms,
                      "readback_ms": (t3 - t2) * 1e3}
        if return_timestamps:
            return out, timestamps
        return out

    def run(self, input_tensor: Any, **kwargs):
        return self.run_yolo_monolithic(input_tensor, **kwargs)["raw_output"]

    # ------------------------------------------------------------------ lifetime
    def close(self):
        self._head_views = None
        super().close()


class DenseGraphSession(EngineSession):
    """Session for ``super_resolution`` graph containers: a dense image tensor comes back, not detect heads.

    The manifest's ``input_normalization`` (the float input is ``(pixel - mean) / divisor``) and the
    input quantization give a pixel lookup table; for SESR it is the identity, so staging copies the
    resized RGB pixels into the input plane. ``dense_output`` names the tail tensor, its quantization
    and the host transform (DepthToSpace, CRD); the image is ``clip((q - zp) * scale + mean)``.
    """

    def __init__(self, container_path: Union[str, Path], device_index: int = 0,
                 xclbin_cache_dir: Optional[Union[str, Path]] = None, map_workspace: bool = True, **_ignored):
        super().__init__(container_path, device_index=device_index, xclbin_cache_dir=xclbin_cache_dir,
                         map_workspace=map_workspace)
        try:
            self._init_dense()
        except Exception:
            self.close()
            raise

    def _init_dense(self) -> None:
        m = self.ignite_manifest
        if self.task != "super_resolution":
            raise ValueError(f"{self.path} is a {self.task} container; open it with GraphSession")
        self.dense = m["dense_output"]
        p = self.input_placement
        self.input_hw: Tuple[int, int] = (int(p["height"]), int(p["width"]))
        self.in_channels = int(m["input_shape"][1])
        self.scale = int(m.get("upscale", 1))
        norm = m.get("input_normalization", {"mean": 0.0, "divisor": 1.0})
        qs = m["quant_scales"]
        x = (np.arange(256, dtype=np.float64) - float(norm["mean"])) / float(norm["divisor"])
        self._input_lut = np.clip(np.round(x / float(qs["input_scale"])) + int(qs["input_zero_point"]),
                                  0, 255).astype(np.uint8)
        self._input_identity = bool(np.array_equal(self._input_lut, np.arange(256, dtype=np.uint8)))
        op = self.ge["placements"][self.dense["tensor"]]
        if op["halo"]:
            raise ValueError("the dense output tensor must be planned without a halo")
        self._out_base = int(op["base"])
        self._out_blocks = int(op["blocks"])
        self._out_hw = (int(op["height"]), int(op["width"]))
        self._out_region = region_bytes(op)
        transform = self.dense["transform"]
        if transform.get("op") != "depth_to_space" or transform.get("mode", "DCR") != "CRD":
            raise ValueError(f"unsupported dense output transform {transform}")
        self._bs = int(transform["blocksize"])
        self._out_channels = int(self.dense["channels"])
        self._image_channels = self._out_channels // (self._bs * self._bs)
        q = np.arange(256, dtype=np.float64)
        self._output_lut = np.clip((q - int(self.dense["zero_point"])) * float(self.dense["scale"])
                                   + float(norm["mean"]), 0, 255).astype(np.uint8)

    def stage_image(self, img_bgr: np.ndarray) -> None:
        """Resize a BGR frame to the network input (bilinear), write RGB input codes into the plane, upload."""
        ih, iw = self.input_hw
        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3 or img_bgr.dtype != np.uint8:
            raise ValueError(f"stage_image expects an HxWx3 uint8 BGR frame, got {img_bgr.shape} {img_bgr.dtype}")
        h = int(self.input_placement["halo"])
        try:
            from ignite_xdna.pipelines.preprocess import resize_bgr_to_c8_plane
            lut = None if self._input_identity else self._input_lut
            if resize_bgr_to_c8_plane(img_bgr, self._input_plane, iw, ih, h, lut):
                self._upload_input()
                return
        except Exception:
            pass
        import cv2
        src = img_bgr if img_bgr.shape[:2] == (ih, iw) else cv2.resize(img_bgr, (iw, ih), interpolation=cv2.INTER_LINEAR)
        rgb = src[:, :, ::-1]
        self._input_plane[h:h + ih, h:h + iw, :3] = rgb if self._input_identity else self._input_lut[rgb]
        self._upload_input()

    def read_output(self) -> np.ndarray:
        """Sync the dense output tensor back; uint8 [blocks][H][W][8] (a view of the mapped workspace)."""
        self.bo_ws.sync(self.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, self._out_region,
                        self._out_base)
        if self._ws_map is not None:
            raw = self._ws_map[self._out_base:self._out_base + self._out_region]
        else:
            raw = np.frombuffer(self.bo_ws.read(self._out_region, self._out_base), dtype=np.uint8)
        return raw.reshape(self._out_blocks, self._out_hw[0], self._out_hw[1], 8)

    def postprocess(self, blocks: np.ndarray) -> np.ndarray:
        """[blocks][H][W][8] output codes -> BGR uint8 image of (H * bs, W * bs): DepthToSpace (CRD) + dequantize.

        CRD: input channel ``k * bs * bs + i * bs + j`` is output pixel ``(y * bs + i, x * bs + j)`` of image
        channel ``k``. Each input channel is one lookup written straight into its stride-``bs`` slice of the
        output (channel order reversed, RGB -> BGR), so no whole-tensor transpose or copy exists.
        """
        oh, ow = self._out_hw
        bs, oc = self._bs, self._image_channels
        lut = self._output_lut
        image = np.empty((oh * bs, ow * bs, oc), dtype=np.uint8)
        if bs == 2 and oc == 3 and self._out_blocks == 2:
            try:
                from ignite_xdna.pipelines.preprocess import depth_to_space_crd_bgr
                if depth_to_space_crd_bgr(blocks, oh, ow, lut, image):
                    return image
            except Exception:
                pass
        for k in range(oc):
            for i in range(bs):
                for j in range(bs):
                    ch = (k * bs + i) * bs + j
                    image[i::bs, j::bs, oc - 1 - k] = lut[blocks[ch // 8, :, :, ch % 8]]
        return image

    def run(self, img_bgr: np.ndarray, timeout_ms: int = 10000) -> Tuple[np.ndarray, Dict[str, float]]:
        t0 = time.perf_counter()
        self.stage_image(img_bgr)
        t1 = time.perf_counter()
        self.dispatch(timeout_ms=timeout_ms)
        t2 = time.perf_counter()
        blocks = self.read_output()
        t3 = time.perf_counter()
        image = self.postprocess(blocks)
        t4 = time.perf_counter()
        return image, {"stage_ms": (t1 - t0) * 1e3, "npu_ms": (t2 - t1) * 1e3, "readback_ms": (t3 - t2) * 1e3,
                       "postprocess_ms": (t4 - t3) * 1e3}


def bf16_input_lut(norm: Dict[str, Any]) -> np.ndarray:
    """uint8 pixel -> uint16 bf16 pattern of ``(pixel - mean) / divisor``, rounded to nearest even.

    For SESR (mean 128, divisor 1) every entry is an integer in [-128, 127] and exact in bf16, so
    this is the float pipeline's own input (``npu/sesr.py``), not an approximation of it.
    """
    x = (np.arange(256, dtype=np.float64) - float(norm["mean"])) / float(norm["divisor"])
    return bf16_bits(to_bf16(x.astype(np.float32)))


def bf16_dense_image(blocks: np.ndarray, channels: int, blocksize: int, mean: float) -> np.ndarray:
    """``[blocks][H][W][8]`` bf16 patterns -> BGR uint8 image of ``(H * bs, W * bs)``.

    DepthToSpace (CRD: channel ``k * bs * bs + i * bs + j`` is pixel ``(y * bs + i, x * bs + j)`` of
    image channel ``k``), then ``clip(value + mean, 0, 255)`` TRUNCATED to uint8, in float32, with the
    channels reversed - exactly ``npu/sesr.py``'s ``postprocess``, which is what the float pipeline
    this is compared against uses. Truncation, not rounding, because that is what it does.

    A non-finite value in a real channel raises. The int8 engine cannot produce one; the bf16 engine
    produces one when a multiply-accumulate reads memory nothing wrote, or when the input lanes past
    the image's channels are not +0.0 (0 x NaN is NaN). Converting it to a pixel would hide exactly
    that fault.
    """
    nb, h, w, _ = blocks.shape
    bs = int(blocksize)
    oc = int(channels) // (bs * bs)
    chw = np.moveaxis(np.asarray(blocks, dtype=np.uint16), 3, 1).reshape(nb * 8, h, w)[:int(channels)]
    y = from_bf16_bits(chw)
    bad = int(np.count_nonzero(~np.isfinite(y)))
    if bad:
        raise RuntimeError(f"{bad} non-finite values in the dense output: a multiply-accumulate read memory "
                           "nothing wrote, or the input lanes past the image's channels are not +0.0")
    image = y.reshape(oc, bs, bs, h, w).transpose(3, 1, 4, 2, 0).reshape(h * bs, w * bs, oc)
    image = image + np.float32(mean)
    np.clip(image, 0.0, 255.0, out=image)
    return np.ascontiguousarray(image.astype(np.uint8)[:, :, ::-1])


class Bf16DenseGraphSession(DenseGraphSession):
    """``super_resolution`` on the bf16 engine (``conv_engine_bf16_v1``): W8A16, activations in real units.

    The same session as ``DenseGraphSession`` - one dispatch, the dense tail read back, DepthToSpace on
    the host - with the conversions a two-byte workspace needs and nothing else:

    * Ingress is a 256-entry uint16 table of bf16 patterns (``bf16_input_lut``) where int8's is a uint8
      table of quantization codes.
    * Egress is arithmetic (``bf16_dense_image``), because a bf16 pattern has 65,536 values where an
      int8 code has 256.
    * The input plane is +0.0 everywhere the image does not write. ``EngineSession`` fills it with the
      int8 zero point, which at two bytes is 0x0080, a bf16 denormal: finite, so no 0 x NaN fires and
      no emulator test sees it, but it is not what the compiler's workspace holds, and ``_upload_input``
      would carry it to the device in the halo ring and in lanes 3..7 with the first frame. It is
      overwritten here before anything is synced. The input is pinned at the workspace's base and never
      handed to another tensor, so what is zeroed once stays zero.

    IGNORED BY DESIGN: ``quant_scales``, ``input_dtype`` and ``dense_output``'s ``scale`` and
    ``zero_point``. They are int8 fields the compiler still writes into a bf16 manifest; nothing here
    reads them.

    Neither native fast path is used. ``resize_bgr_to_c8_plane`` and ``depth_to_space_crd_bgr`` are uint8
    routines; this numpy path is correct and slower, and what it costs is for a sitting to measure.
    """

    SUPPORTED_ELEMS = (ELEM_BF16,)

    def _init_dense(self) -> None:
        m = self.ignite_manifest
        if self.task != "super_resolution":
            raise ValueError(f"{self.path} is a {self.task} container; open it with GraphSession")
        self.dense = m["dense_output"]
        p = self.input_placement
        self.input_hw = (int(p["height"]), int(p["width"]))
        self.in_channels = int(m["input_shape"][1])
        if self.in_channels != 3 or int(p["blocks"]) != 1:
            raise ValueError(f"{self.path}: stage_image writes a three-channel image into one channel block, "
                             f"not {self.in_channels} channels in {p['blocks']}")
        self.scale = int(m.get("upscale", 1))
        norm = m.get("input_normalization", {"mean": 0.0, "divisor": 1.0})
        if float(norm["divisor"]) != 1.0:
            # The int8 session adds the mean back to its output and never multiplies by the divisor, so
            # it assumes the network answers in the input's centred pixel units. That holds for SESR. At
            # another divisor the manifest does not say what units the output is in, so refuse to guess.
            raise ValueError(f"{self.path}: input divisor {norm['divisor']}; the output's units are only "
                             "known at divisor 1")
        self._mean = float(norm["mean"])
        self._input_lut = bf16_input_lut(norm)
        op = self.ge["placements"][self.dense["tensor"]]
        if op["halo"]:
            raise ValueError("the dense output tensor must be planned without a halo")
        self._out_base = int(op["base"])
        self._out_blocks = int(op["blocks"])
        self._out_hw = (int(op["height"]), int(op["width"]))
        self._out_region = region_bytes(op)
        transform = self.dense["transform"]
        if transform.get("op") != "depth_to_space" or transform.get("mode", "DCR") != "CRD":
            raise ValueError(f"unsupported dense output transform {transform}")
        self._bs = int(transform["blocksize"])
        self._out_channels = int(self.dense["channels"])
        self._image_channels = self._out_channels // (self._bs * self._bs)
        self._input_plane[...] = 0

    def _upload_input(self) -> None:
        # The unmapped path hands pyxrt a byte view: the plane is uint16, and a write that counted its
        # elements as bytes would send half of it. The mapped path (the default) never writes.
        base = self.input_placement["base"]
        if self._ws_map is None:
            self.bo_ws.write(self._input_plane.view(np.uint8), base)
        self.bo_ws.sync(self.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, self._input_bytes, base)

    def stage_image(self, img_bgr: np.ndarray) -> None:
        """Resize a BGR frame to the network input (bilinear), write its RGB bf16 patterns, upload."""
        ih, iw = self.input_hw
        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3 or img_bgr.dtype != np.uint8:
            raise ValueError(f"stage_image expects an HxWx3 uint8 BGR frame, got {img_bgr.shape} {img_bgr.dtype}")
        import cv2
        h = int(self.input_placement["halo"])
        src = img_bgr if img_bgr.shape[:2] == (ih, iw) else cv2.resize(img_bgr, (iw, ih), interpolation=cv2.INTER_LINEAR)
        self._input_plane[h:h + ih, h:h + iw, :3] = self._input_lut[src[:, :, ::-1]]
        self._upload_input()

    def stage_quantized(self, chw: np.ndarray) -> None:
        """Write bf16 PATTERNS, uint16 ``[C][H][W]``, into the input plane and upload; lanes past C are +0.0.

        Patterns rather than values, as ``Bf16Workspace.write_tensor`` takes them: assigning float32 into
        the uint16 plane would truncate each value to an integer without a word of complaint.
        """
        chw = np.asarray(chw)
        if chw.dtype != np.uint16:
            raise TypeError(f"the input plane holds bf16 patterns; pass bf16_bits(values), not {chw.dtype}")
        p = self.input_placement
        h, c = int(p["halo"]), int(chw.shape[0])
        if chw.shape[1:] != self.input_hw or not 0 < c <= 8:
            raise ValueError(f"expected [C <= 8][{self.input_hw[0]}][{self.input_hw[1]}], got {chw.shape}")
        interior = self._input_plane[h:h + p["height"], h:h + p["width"], :]
        interior[..., :c] = np.moveaxis(chw, 0, -1)
        interior[..., c:] = 0
        self._upload_input()

    def stage_tensor(self, name: str, data: np.ndarray, sync: bool = True):
        raise NotImplementedError("EngineSession.stage_tensor pads with the int8 zero point, a bf16 denormal; "
                                  "stage the input with stage_image or stage_quantized")

    def read_output(self) -> np.ndarray:
        """Sync the dense output tensor back; uint16 bf16 patterns [blocks][H][W][8] (a view when mapped)."""
        self.bo_ws.sync(self.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, self._out_region,
                        self._out_base)
        if self._ws_map is not None:
            raw = self._ws_map[self._out_base:self._out_base + self._out_region]
        else:
            raw = np.frombuffer(self.bo_ws.read(self._out_region, self._out_base), dtype=np.uint8)
        return raw.view(np.uint16).reshape(self._out_blocks, self._out_hw[0], self._out_hw[1], 8)

    def postprocess(self, blocks: np.ndarray) -> np.ndarray:
        """[blocks][H][W][8] bf16 patterns -> BGR uint8 image of (H * bs, W * bs); see ``bf16_dense_image``."""
        return bf16_dense_image(blocks, self._out_channels, self._bs, self._mean)


def sr_session_class(manifest: Optional[Dict[str, Any]]) -> type:
    """The session a ``super_resolution`` container needs, by the activation width its placements declare.

    Anything but bf16 goes to ``DenseGraphSession``, whose admission then refuses a width it cannot read
    by name, so an unknown width is still refused loudly rather than routed somewhere plausible.
    """
    return Bf16DenseGraphSession if container_elem(manifest) == ELEM_BF16 else DenseGraphSession


def open_sr_session(container_path: Union[str, Path], device_index: int = 0, **kwargs) -> DenseGraphSession:
    """Open a ``super_resolution`` container in the session its width needs (see ``sr_session_class``)."""
    with IgniteModelReader(container_path) as reader:
        cls = sr_session_class(reader.manifest)
    return cls(container_path, device_index=device_index, **kwargs)


class ClassificationSession(EngineSession):
    """Session for ``classify`` graph containers: ImageNet-style classification logits come back.

    The model's weights and matrix multiplication are executed on the NPU as a 1x1 convolution
    (k=1, s=1, p=0) across 20x20 tile geometry, eliminating all CPU host segments.
    """

    def __init__(self, container_path: Union[str, Path], device_index: int = 0,
                 xclbin_cache_dir: Optional[Union[str, Path]] = None, map_workspace: bool = True, **_ignored):
        super().__init__(container_path, device_index=device_index, xclbin_cache_dir=xclbin_cache_dir,
                         map_workspace=map_workspace)
        try:
            self._init_classification()
        except Exception:
            self.close()
            raise

    def _init_classification(self) -> None:
        m = self.ignite_manifest
        if self.task != "classify":
            raise ValueError(f"{self.path} is a {self.task} container; open it with GraphSession")
        self.cls_meta = m["classification"]
        ip = self.input_placement
        self.input_hw: Tuple[int, int] = (int(ip["height"]), int(ip["width"]))
        self.in_channels = int(m["input_shape"][1])
        self.in_blocks = int(ip["blocks"])
        self.num_classes = int(self.cls_meta["channels"])
        self.scale = float(self.cls_meta["scale"])
        self.zero_point = int(self.cls_meta["zero_point"])

        qs = m["quant_scales"]
        self.input_scale = float(qs["input_scale"])
        self.input_zp = int(qs["input_zero_point"])

        # Input buffer: one whole region, in the placement's own element width
        h, w, halo = int(ip["height"]), int(ip["width"]), int(ip["halo"])
        self._cls_in_base = int(ip["base"])
        self._cls_in_bytes = region_bytes(ip, self.in_blocks)
        self._cls_in_shape = (self.in_blocks, h + 2 * halo, w + 2 * halo, 8)

        op = self.ge["placements"][self.cls_meta["tensor"]]
        self._out_base = int(op["base"])
        self._out_blocks = int(op["blocks"])
        self._out_hw = (int(op["height"]), int(op["width"]))
        self._out_region = region_bytes(op)

    def stage_pooled(self, pooled_features: np.ndarray) -> None:
        """Stage a pooled feature vector [Cin] or [1, Cin, 1, 1] into workspace at pixel (0, 0)."""
        x = np.asarray(pooled_features)
        x_flat = x.reshape(-1)
        if x_flat.size != self.in_channels:
            raise ValueError(f"expected {self.in_channels} features, got {x_flat.size}")

        if np.issubdtype(x_flat.dtype, np.floating):
            # Quantize float features
            q = np.clip(np.round(x_flat.astype(np.float64) / self.input_scale) + self.input_zp, 0, 255).astype(np.uint8)
        else:
            q = x_flat.astype(np.uint8)

        # Build blocks array initialized to ZP
        blocks_arr = np.full(self._cls_in_shape, ZP, dtype=np.uint8)
        ip = self.input_placement
        halo = int(ip["halo"])

        # Stage features at (y=0, x=0) across channel blocks
        for b in range(self.in_blocks):
            ch_start = b * 8
            ch_end = min(ch_start + 8, self.in_channels)
            blocks_arr[b, halo, halo, :ch_end - ch_start] = q[ch_start:ch_end]

        if self._ws_map is not None:
            self._ws_map[self._cls_in_base:self._cls_in_base + self._cls_in_bytes] = blocks_arr.reshape(-1)
        else:
            self.bo_ws.write(blocks_arr.reshape(-1), self._cls_in_base)
        self.bo_ws.sync(self.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
                        self._cls_in_bytes, self._cls_in_base)

    def stage_input(self, x: np.ndarray) -> None:
        """Stage input tensor (image or pooled features) into workspace."""
        arr = np.asarray(x)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[1:] == self.input_hw:
            if np.issubdtype(arr.dtype, np.floating):
                q = np.clip(np.round(arr.astype(np.float64) / self.input_scale) + self.input_zp, 0, 255).astype(np.uint8)
            else:
                q = arr.astype(np.uint8)
            if self.in_blocks == 1:
                super().stage_quantized(q)
            else:
                self.stage_quantized(q)
        else:
            self.stage_pooled(arr)

    def stage_quantized(self, chw: np.ndarray) -> None:
        """Stage an already-quantized uint8 [C, H, W] tensor into workspace."""
        c = chw.shape[0]
        ip = self.input_placement
        halo = int(ip["halo"])
        h, w = int(ip["height"]), int(ip["width"])
        blocks_arr = np.full(self._cls_in_shape, ZP, dtype=np.uint8)
        padded = np.full((self.in_blocks * 8, h, w), ZP, dtype=np.uint8)
        padded[:c] = chw
        for b in range(self.in_blocks):
            blocks_arr[b, halo:halo + h, halo:halo + w, :] = np.transpose(
                padded[b * 8:(b + 1) * 8], (1, 2, 0)
            )
        if self._ws_map is not None:
            self._ws_map[self._cls_in_base:self._cls_in_base + self._cls_in_bytes] = blocks_arr.reshape(-1)
        else:
            self.bo_ws.write(blocks_arr.reshape(-1), self._cls_in_base)
        self.bo_ws.sync(self.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
                        self._cls_in_bytes, self._cls_in_base)

    def read_logits(self) -> np.ndarray:
        """Sync output tensor from device, extract pixel (0, 0) and dequantize to float32 logits."""
        self.bo_ws.sync(self.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE,
                        self._out_region, self._out_base)
        if self._ws_map is not None:
            raw = self._ws_map[self._out_base:self._out_base + self._out_region]
        else:
            raw = np.frombuffer(self.bo_ws.read(self._out_region, self._out_base), dtype=np.uint8)
        blocks = raw.reshape(self._out_blocks, self._out_hw[0], self._out_hw[1], 8)
        raw_u8 = blocks[:, 0, 0, :].reshape(-1)[:self.num_classes]
        logits = (raw_u8.astype(np.float32) - float(self.zero_point)) * float(self.scale)
        return logits

    def run(self, input_data: np.ndarray, timeout_ms: int = 10000) -> Tuple[np.ndarray, Dict[str, float]]:
        t0 = time.perf_counter()
        self.stage_input(input_data)
        t1 = time.perf_counter()
        self.dispatch(timeout_ms=timeout_ms)
        t2 = time.perf_counter()
        logits = self.read_logits()
        t3 = time.perf_counter()
        return logits, {
            "stage_ms": (t1 - t0) * 1e3,
            "npu_ms": (t2 - t1) * 1e3,
            "readback_ms": (t3 - t2) * 1e3,
        }


class ComposedSession:
    """Chains multiple .ignite stage containers in a single persistent hardware context.

    Shares a unified workspace BO across all stages according to the Tensor Placement ABI,
    achieving 0 bytes host memory bounce between stages and sub-10 µs inter-stage dispatch.
    """

    def __init__(self, stage_paths: Sequence[Union[str, Path]], device_index: int = 0,
                 weights_paths: Optional[Sequence[Optional[Union[str, Path]]]] = None,
                 map_workspace: bool = True):
        self.stage_paths = [Path(p) for p in stage_paths]
        if not self.stage_paths:
            raise ValueError("At least one stage container path is required")
        self.device_index = device_index
        weights_paths = list(weights_paths) if weights_paths else [None] * len(self.stage_paths)

        self.stages: List[Any] = []
        self.harness = None
        self.bo_ws = None
        try:
            for p, w in zip(self.stage_paths, weights_paths):
                with IgniteModelReader(p) as r:
                    task = r.manifest.get("task", "detect")
                cls = GraphSession if task in ("detect", "pose") else EngineSession
                stage = cls(p, device_index=device_index, weights_path=w,
                            map_workspace=map_workspace,
                            bo_ws=self.bo_ws, harness=self.harness)
                if not self.stages:
                    self.harness = stage.harness
                    self.bo_ws = stage.bo_ws
                self.stages.append(stage)
        except Exception:
            self.close()
            raise

    def stage_input(self, input_tensor: Any) -> None:
        """Stage input into the first stage's workspace plane."""
        self.stages[0].stage_input(input_tensor)

    def read_heads(self, unswizzle: bool = True) -> Any:
        """Read heads from the final stage."""
        return self.stages[-1].read_heads(unswizzle=unswizzle)

    def dispatch(self, timeout_ms: int = 10000, max_stages: Optional[int] = None) -> float:
        """Dispatches stages sequentially in persistent context and returns total NPU milliseconds."""
        active = self.stages if max_stages is None else self.stages[:max_stages]
        total_ms = 0.0
        for stage in active:
            total_ms += stage.dispatch(timeout_ms=timeout_ms)
        return total_ms

    def run_yolo(self, input_tensor: Any, unswizzle: bool = True, timeout_ms: int = 10000) -> Any:
        """Runs full forward pass across all composed stages with zero host memory copies."""
        self.stage_input(input_tensor)
        self.dispatch(timeout_ms=timeout_ms)
        return self.read_heads(unswizzle=unswizzle)

    def close(self):
        for stage in reversed(self.stages):
            try:
                stage.close()
            except Exception:
                pass
        self.stages.clear()
        self.harness = None
        self.bo_ws = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


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
  into an upscaled BGR image.

A container whose manifest lists ``graph_engine.segments`` (YOLO11's C2PSA block on the host) runs
them in order in ``dispatch``: each NPU segment is its own instruction stream and XRT run over the
shared workspace, and each host segment is a ``HostStep`` that reads its input tensor from the
workspace, runs the layer's ONNX model on ONNX Runtime's CPU provider and writes the output tensor
back before the next NPU segment reads it.

All buffers are allocated once at construction.
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ignite_xdna.compiler.serializer import IgniteModelReader
from ignite_xdna.runtime.driver import XrtSiliconHarness, get_repo_root, setup_xrt_environment
from ignite_xdna.runtime.heads import HEAD_NAMES, POSE_HEAD_NAMES, HeadStatus, resolve_head_layout

ENGINE_NAME = "conv_engine_v1"
ZP = 128


def is_graph_container(manifest: Optional[Dict[str, Any]]) -> bool:
    return bool(manifest) and manifest.get("engine") == ENGINE_NAME


def _plane_view(buf: np.ndarray, p: Dict[str, Any], planes: Optional[int] = None) -> np.ndarray:
    h, w, halo = p["height"], p["width"], p["halo"]
    n = p["planes"] if planes is None else planes
    plane_bytes = (h + 2 * halo) * (w + 2 * halo) * 8
    return buf[p["base"]:p["base"] + n * plane_bytes].reshape(n, h + 2 * halo, w + 2 * halo, 8)


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
    h, w, halo = int(p["height"]), int(p["width"]), int(p["halo"])
    return int(p["base"]), int(p["blocks"]) * (h + 2 * halo) * (w + 2 * halo) * 8


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
        from ignite_xdna.pipelines.power import ort_session_options  # the power mode covers host segments too
        self.ort_session = ort.InferenceSession(blob, ort_session_options(), providers=["CPUExecutionProvider"])
        self.input_name = self.ort_session.get_inputs()[0].name
        placements = session.ge["placements"]
        self.pin, self.pout = placements[seg["input"]], placements[seg["output"]]
        self.in_base, self.in_bytes = _region(self.pin)
        self.out_base, self.out_bytes = _region(self.pout)
        self._x = np.empty((1, int(self.pin["channels"]), int(self.pin["height"]), int(self.pin["width"])),
                           dtype=np.uint8)
        self.last_ms = 0.0

    def run(self) -> None:
        s = self.session
        t0 = time.perf_counter()
        d = s.harness.pyxrt.xclBOSyncDirection
        p = self.pin
        h, w, halo, blocks = int(p["height"]), int(p["width"]), int(p["halo"]), int(p["blocks"])
        s.bo_ws.sync(d.XCL_BO_SYNC_BO_FROM_DEVICE, self.in_bytes, self.in_base)
        if s._ws_map is not None:
            raw = s._ws_map[self.in_base:self.in_base + self.in_bytes]
        else:
            raw = np.frombuffer(s.bo_ws.read(self.in_bytes, self.in_base), dtype=np.uint8)
        planes = raw.reshape(blocks, h + 2 * halo, w + 2 * halo, 8)[:, halo:halo + h, halo:halo + w, :]
        self._x[0] = np.transpose(planes, (0, 3, 1, 2)).reshape(blocks * 8, h, w)[:self._x.shape[1]]
        y = self.ort_session.run(None, {self.input_name: self._x})[0]
        q = self.pout
        oh, ow, ohalo, oblocks = int(q["height"]), int(q["width"]), int(q["halo"]), int(q["blocks"])
        if s._ws_map is not None:
            region = s._ws_map[self.out_base:self.out_base + self.out_bytes]
        else:
            region = np.frombuffer(s.bo_ws.read(self.out_bytes, self.out_base), dtype=np.uint8).copy()
        full = np.full((oblocks * 8, oh, ow), ZP, dtype=np.uint8)
        full[:y.shape[1]] = y[0]
        interior = region.reshape(oblocks, oh + 2 * ohalo, ow + 2 * ohalo, 8)[:, ohalo:ohalo + oh, ohalo:ohalo + ow, :]
        interior[:] = np.transpose(full.reshape(oblocks, 8, oh, ow), (0, 2, 3, 1))
        if s._ws_map is None:
            s.bo_ws.write(region, self.out_base)
        s.bo_ws.sync(d.XCL_BO_SYNC_BO_TO_DEVICE, self.out_bytes, self.out_base)
        self.last_ms = (time.perf_counter() - t0) * 1e3


class EngineSession:
    """The convolution engine and its buffers for one graph-engine container (see ``compiler/engine_compile.py``)."""

    def __init__(self, container_path: Union[str, Path], device_index: int = 0,
                 xclbin_cache_dir: Optional[Union[str, Path]] = None, map_workspace: bool = True, **_ignored):
        setup_xrt_environment()
        self.path = Path(container_path)
        self.device_index = device_index
        self._reader = IgniteModelReader(self.path)
        self.ignite_manifest: Dict[str, Any] = self._reader.manifest
        if not is_graph_container(self.ignite_manifest):
            raise ValueError(f"{self.path} is not a graph-engine container")
        self.ge: Dict[str, Any] = self.ignite_manifest["graph_engine"]
        self.task = self.ignite_manifest.get("task", "detect")
        self.monolithic_stages: Dict[str, Any] = {}
        self.out_bytes = int(self.ignite_manifest["egress_bytes"])
        self.num_cores = 16
        self.single_dispatch = True
        self._closed = False

        # pyxrt.xclbin needs a file: cache the blob by hash.
        cache_dir = Path(xclbin_cache_dir or get_repo_root() / "build" / "ignite_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        xclbin_bytes = self._reader.get_blob_bytes("engine.xclbin")
        sha = hashlib.sha256(xclbin_bytes).hexdigest()[:16]
        self.xclbin_path = cache_dir / f"engine_{sha}.xclbin"
        if not self.xclbin_path.exists() or self.xclbin_path.stat().st_size != len(xclbin_bytes):
            self.xclbin_path.write_bytes(xclbin_bytes)

        self.harness = XrtSiliconHarness(device_idx=device_index)
        self.harness.load_xclbin(str(self.xclbin_path), "MLIR_AIE")
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
        wp = self._reader.get_blob_memoryview("wpackets.bin")
        self.workspace_bytes = int(self.ge["workspace_bytes"])
        self.bo_ws = self.harness.create_host_bo(self.workspace_bytes, 3)
        self.bo_wp = self.harness.create_host_bo(max(64, len(wp)), 4)
        self.bo_wp.write(wp, 0)
        self.bo_wp.sync(self.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        # One-time workspace image: halo rings for every tensor.
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
        self._input_bytes = int(np.prod(plane_shape))
        self._ws_map: Optional[np.ndarray] = None
        if map_workspace:
            try:
                mapped = np.frombuffer(self.bo_ws.map(), dtype=np.uint8)
                if mapped.size >= self.workspace_bytes:
                    self._ws_map = mapped
            except Exception:  # noqa: BLE001 - mapping is an optimisation only
                self._ws_map = None
        if self._ws_map is not None:
            base = p["base"]
            self._input_plane = self._ws_map[base:base + self._input_bytes].reshape(plane_shape)
            self._input_plane[:] = ZP
        else:
            self._input_plane = np.full(plane_shape, ZP, dtype=np.uint8)
        self.last_dispatch_ms = 0.0

        # One XRT run object for every frame: its arguments (opcode, instructions,
        # workspace, packets) never change, so each frame only starts and awaits it.
        pyxrt = self.harness.pyxrt
        self._completed = getattr(getattr(pyxrt, "ert_cmd_state", None), "ERT_CMD_STATE_COMPLETED", None)
        self._runs: List[Any] = []
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
        self.last_segment_ms: List[float] = []

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

    def dispatch(self, timeout_ms: int = 10000) -> float:
        """Run the container's segments in order and return the NPU dispatch milliseconds.

        ``last_dispatch_ms`` counts NPU segments only, ``last_host_ms`` the host layers between them
        (syncs and ONNX Runtime), and ``last_segment_ms`` every segment in order.
        """
        seg_ms: List[float] = []
        npu_k = host_k = 0
        for seg in self.segments:
            t0 = time.perf_counter()
            if seg["kind"] == "npu":
                self._dispatch_stream(npu_k, timeout_ms)
                npu_k += 1
            else:
                self._host_steps[host_k].run()
                host_k += 1
            seg_ms.append((time.perf_counter() - t0) * 1e3)
        self.last_segment_ms = seg_ms
        self.last_host_ms = sum(ms for ms, seg in zip(seg_ms, self.segments) if seg["kind"] == "host")
        self.last_dispatch_ms = sum(seg_ms) - self.last_host_ms
        return self.last_dispatch_ms

    def read_tensor(self, name: str) -> np.ndarray:
        """Debug helper: sync one tensor from the device and return uint8 [C][H][W]."""
        p = self.ge["placements"][name]
        h, w, halo = p["height"], p["width"], p["halo"]
        nbytes = p["blocks"] * (h + 2 * halo) * (w + 2 * halo) * 8
        self.bo_ws.sync(self.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, nbytes, p["base"])
        raw = np.frombuffer(self.bo_ws.read(nbytes, p["base"]), dtype=np.uint8)
        planes = raw.reshape(p["blocks"], h + 2 * halo, w + 2 * halo, 8)[:, halo:halo + h, halo:halo + w, :]
        return np.transpose(planes, (0, 3, 1, 2)).reshape(p["blocks"] * 8, h, w)[:p["channels"]]

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
        if self.harness is not None:
            self.harness.close()
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
                 xclbin_cache_dir: Optional[Union[str, Path]] = None, map_workspace: bool = True, **_ignored):
        super().__init__(container_path, device_index=device_index, xclbin_cache_dir=xclbin_cache_dir,
                         map_workspace=map_workspace)
        if self.task not in ("detect", "pose"):
            task = self.task
            self.close()
            raise ValueError(f"{self.path} is a {task} container; open it with DenseGraphSession")
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
        self._egress = np.zeros(self.out_bytes, dtype=np.int8)
        self._head_regions = []
        for name in self.head_names:
            hm = self.heads_meta[name]
            hp = self.ge["placements"][hm["tensor"]]
            nbytes = hp["blocks"] * hp["height"] * hp["width"] * 8
            self._head_regions.append((name, hm, hp, hp["base"], nbytes))
        # Per-anchor class-logit maxima of the class heads (int8, zero point 0), filled natively
        # during readback so the decoder's confidence prune does not scan the class tensors. A pose score
        # head has one channel, so the pose decoder prunes on it directly.
        self._cls_max: Dict[str, np.ndarray] = {
            name: np.empty(self.ge["placements"][self.heads_meta[name]["tensor"]]["height"]
                           * self.ge["placements"][self.heads_meta[name]["tensor"]]["width"], dtype=np.int8)
            for name in self.head_names if name.endswith("_cls") and self.task == "detect"}
        self._cls_max_valid = False
        self._head_status: Optional[HeadStatus] = None
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

    def read_heads(self) -> np.ndarray:
        """Sync the head tensors back and assemble the int8 NCHW egress buffer."""
        d = self.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        cls_ok = self._class_max is not None
        for name, hm, hp, base, nbytes in self._head_regions:
            self.bo_ws.sync(d, nbytes, base)
            if self._ws_map is not None:
                raw = self._ws_map[base:base + nbytes]
            else:
                raw = np.frombuffer(self.bo_ws.read(nbytes, base), dtype=np.uint8)
            c = hm["channels"]
            off = hm["egress_offset"]
            dst = self._egress[off:off + c * hp["height"] * hp["width"]]
            if name in self._cls_max:
                cls_ok = cls_ok and self._class_max(raw, hp["blocks"], hp["height"], hp["width"], c,
                                                    self._cls_max[name])
            # uint8 with zero point 128 -> int8 with zero point 0 (flip the top bit), NHWC blocks -> NCHW
            if self._to_nchw is not None and self._to_nchw(raw, hp["blocks"], hp["height"], hp["width"], c, dst):
                continue
            blocked = raw.reshape(hp["blocks"], hp["height"], hp["width"], 8)
            chw = np.transpose(blocked, (0, 3, 1, 2)).reshape(hp["blocks"] * 8, hp["height"], hp["width"])
            dst[:] = (chw[:c] ^ 0x80).view(np.int8).reshape(-1)
        self._cls_max_valid = bool(cls_ok)
        return self._egress

    def run_yolo_monolithic(self, input_tensor: Any, unswizzle: bool = True, timeout_ms: int = 10000,
                            return_timestamps: bool = False):
        """Whole-network forward pass; returns the ``run_yolo_monolithic`` head dict of InferenceSession.

        ``input_tensor=None`` dispatches on the input plane already staged by ``stage_image``.
        """
        t0 = time.perf_counter()
        if input_tensor is not None:
            self.stage_input(input_tensor)
        t1 = time.perf_counter()
        self.dispatch(timeout_ms=timeout_ms)
        t2 = time.perf_counter()
        egress = self.read_heads()
        t3 = time.perf_counter()
        status = self.head_status
        out: Dict[str, Any] = {name: None for name in self.head_names}
        if status.present:
            if self._head_views is None:
                # The egress buffer is allocated once, so its head views and scales are too.
                self._head_views = status.layout.unpack(egress)
                self._head_scales = status.layout.scales()
            out.update(self._head_views)
            out["scales"] = self._head_scales
            if self._cls_max_valid:
                out["cls_max"] = self._cls_max  # {p*_cls: int8 per-anchor class maxima}, see YoloDecoder
        out["heads_present"] = status.present
        out["head_status"] = status.reason
        out["raw_output"] = egress
        out["raw_heads"] = egress
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
        self._out_region = self._out_blocks * self._out_hw[0] * self._out_hw[1] * 8
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
        import cv2
        ih, iw = self.input_hw
        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3 or img_bgr.dtype != np.uint8:
            raise ValueError(f"stage_image expects an HxWx3 uint8 BGR frame, got {img_bgr.shape} {img_bgr.dtype}")
        src = img_bgr if img_bgr.shape[:2] == (ih, iw) else cv2.resize(img_bgr, (iw, ih), interpolation=cv2.INTER_LINEAR)
        h = int(self.input_placement["halo"])
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

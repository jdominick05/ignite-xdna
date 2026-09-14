"""Execute a graph-engine ``.ignite`` container (whole YOLOv8n on the NPU).

The container's instruction stream drives the 16-core convolution engine
through every layer; activations live in one host-visible workspace buffer
in the channel-blocked layout the compiler planned. Per frame the session
stages the quantized image into the input tensor, dispatches once, reads the
six head tensors back and presents them through the ``head_layout`` contract
of ``runtime/heads.py`` as int8 NCHW views of an egress buffer, so
``YoloPipeline.predict_sync(use_oracle_for_boxes=False)`` decodes them with
``head_source == "npu"``. All buffers are allocated once at construction.
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ignite_xdna.compiler.serializer import IgniteModelReader
from ignite_xdna.runtime.driver import XrtSiliconHarness, get_repo_root, setup_xrt_environment
from ignite_xdna.runtime.heads import HEAD_NAMES, HeadStatus, resolve_head_layout

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


class GraphSession:
    """Session for ``engine == conv_engine_v1`` containers (see ``compiler/engine_compile.py``)."""

    def __init__(self, container_path: Union[str, Path], device_index: int = 0,
                 xclbin_cache_dir: Optional[Union[str, Path]] = None, map_workspace: bool = False, **_ignored):
        setup_xrt_environment()
        self.path = Path(container_path)
        self.device_index = device_index
        self._reader = IgniteModelReader(self.path)
        self.ignite_manifest: Dict[str, Any] = self._reader.manifest
        if not is_graph_container(self.ignite_manifest):
            raise ValueError(f"{self.path} is not a graph-engine container")
        self.ge: Dict[str, Any] = self.ignite_manifest["graph_engine"]
        self.monolithic_stages: Dict[str, Any] = {}
        self.out_bytes = int(self.ignite_manifest["egress_bytes"])
        self.in_bytes = 3 * 640 * 640
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
        self.bo_instr_exec, self.ninstr_exec = self.harness.create_instruction_bo_from_bytes(
            self._reader.get_blob_memoryview("insts.bin"))
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
        # With ``map_workspace`` the plane is written in place through bo.map()
        # and only synced; measured no faster than bo.write on Phoenix (3.5 vs
        # 3.0 ms per frame), so the default is the host copy.
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
        self._input_lut = input_lut(float(self.ignite_manifest["quant_scales"]["input_scale"]),
                                    int(self.ignite_manifest["quant_scales"].get("input_zero_point", ZP)))
        # Head readback: each head tensor is contiguous (halo 0); egress is int8 NCHW.
        self.heads_meta = self.ge["heads"]
        self._egress = np.zeros(self.out_bytes, dtype=np.int8)
        self._head_regions = []
        for name in HEAD_NAMES:
            hm = self.heads_meta[name]
            hp = self.ge["placements"][hm["tensor"]]
            nbytes = hp["blocks"] * hp["height"] * hp["width"] * 8
            self._head_regions.append((name, hm, hp, hp["base"], nbytes))
        self._head_status: Optional[HeadStatus] = None
        self.last_dispatch_ms = 0.0

    # ------------------------------------------------------------------ status
    @property
    def head_status(self) -> HeadStatus:
        if self._head_status is None:
            self._head_status = resolve_head_layout(self.ignite_manifest, self.out_bytes)
        return self._head_status

    @property
    def is_monolithic(self) -> bool:
        return True

    @property
    def stage_names(self):
        return [L["name"] for L in self.ge["layers"]]

    # ------------------------------------------------------------------ frame
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
        base = self.input_placement["base"]
        if self._ws_map is None:
            self.bo_ws.write(self._input_plane, base)
        self.bo_ws.sync(self.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, self._input_bytes, base)

    def read_heads(self) -> np.ndarray:
        """Sync the six head tensors back and assemble the int8 NCHW egress buffer."""
        d = self.harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        for name, hm, hp, base, nbytes in self._head_regions:
            self.bo_ws.sync(d, nbytes, base)
            if self._ws_map is not None:
                raw = self._ws_map[base:base + nbytes]
            else:
                raw = np.frombuffer(self.bo_ws.read(nbytes, base), dtype=np.uint8)
            blocked = raw.reshape(hp["blocks"], hp["height"], hp["width"], 8)
            chw = np.transpose(blocked, (0, 3, 1, 2)).reshape(hp["blocks"] * 8, hp["height"], hp["width"])
            c = hm["channels"]
            off = hm["egress_offset"]
            # uint8 with zero point 128 -> int8 with zero point 0 (flip the top bit)
            self._egress[off:off + c * hp["height"] * hp["width"]] = (chw[:c] ^ 0x80).view(np.int8).reshape(-1)
        return self._egress

    def dispatch(self, timeout_ms: int = 10000) -> float:
        t0 = time.perf_counter()
        run, state = self.harness.dispatch_kernel(self.bo_instr_exec, self.ninstr_exec, self.bo_ws, self.bo_wp,
                                                  timeout_ms=timeout_ms)
        if str(state) != "ert_cmd_state.ERT_CMD_STATE_COMPLETED":
            raise RuntimeError(f"graph engine dispatch ended in state {state}")
        self.last_dispatch_ms = (time.perf_counter() - t0) * 1e3
        return self.last_dispatch_ms

    def run_yolo_monolithic(self, input_tensor: Any, unswizzle: bool = True, timeout_ms: int = 10000,
                            return_timestamps: bool = False):
        """Whole-network forward pass; returns the ``run_yolo_monolithic`` head dict of InferenceSession."""
        t0 = time.perf_counter()
        self.stage_input(input_tensor)
        t1 = time.perf_counter()
        self.dispatch(timeout_ms=timeout_ms)
        t2 = time.perf_counter()
        egress = self.read_heads()
        t3 = time.perf_counter()
        status = self.head_status
        out: Dict[str, Any] = {name: None for name in HEAD_NAMES}
        if status.present:
            out.update(status.layout.unpack(egress))
            out["scales"] = status.layout.scales()
        out["heads_present"] = status.present
        out["head_status"] = status.reason
        out["raw_output"] = egress
        out["raw_heads"] = egress
        timestamps = {"stage_ms": (t1 - t0) * 1e3, "npu_ms": (t2 - t1) * 1e3, "readback_ms": (t3 - t2) * 1e3}
        if return_timestamps:
            return out, timestamps
        return out

    def run(self, input_tensor: Any, **kwargs):
        return self.run_yolo_monolithic(input_tensor, **kwargs)["raw_output"]

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

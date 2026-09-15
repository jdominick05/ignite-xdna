# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
src/ignite_xdna/runtime/heads.py

Detect- and pose-head egress layout for YOLOv8 containers.

The runtime never guesses how the device packs the six raw detection heads
(``p3_box … p5_cls``) into ``bo_out``. A container that emits them must say so
in its manifest, and the layout resolves only when every piece is declared:

  * ``output_shapes`` — the six 4-D head shapes (``ignite-compile`` writes these
    for every YOLO container, whether or not the stream computes them);
  * ``head_layout`` — per head, the byte ``offset`` of the int8 NCHW tensor
    inside the egress buffer plus the dequantization ``scale`` and
    ``zero_point`` of that head;
  * an egress buffer large enough to hold every declared range, with the six
    ranges disjoint.

Anything less resolves to ``HeadStatus(present=False)`` with the reason spelled
out, and the session returns ``None`` for each head instead of a tensor. The
shipped ``build/yolov8n.ignite`` has ``output_shapes`` (1,209,600 values) and a
4,096-byte egress, because every stage stream is the single-layer conv0
template (docs/LOW_LEVEL_AUDIT.md §1.5); it therefore resolves as absent.

Pose containers (``task == "pose"``, YOLOv8-pose) carry nine heads, ``POSE_HEAD_NAMES``: the six detect
heads with a single person class plus the keypoint heads ``p3_kpt … p5_kpt`` (51 channels each). The
resolver checks whichever names the manifest's ``task`` declares.

Unpacking is zero-copy: each head is a ``reshape`` view of the int8 egress
array. Dequantization (``(q - zero_point) * scale``) is left to the consumer so
that pruning can run in the int8 domain and only survivors are converted.
"""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

HEAD_NAMES: Tuple[str, ...] = ("p3_box", "p3_cls", "p4_box", "p4_cls", "p5_box", "p5_cls")
POSE_HEAD_NAMES: Tuple[str, ...] = HEAD_NAMES + ("p3_kpt", "p4_kpt", "p5_kpt")
HEAD_RANK = 4


def head_names_for(manifest: Optional[Mapping[str, Any]]) -> Tuple[str, ...]:
    """The heads a container's ``task`` declares: nine for ``pose``, the six detect heads otherwise."""
    return POSE_HEAD_NAMES if (manifest or {}).get("task") == "pose" else HEAD_NAMES


@dataclass(frozen=True)
class HeadSpec:
    """One raw head: int8 NCHW tensor at ``offset`` bytes into the egress buffer."""
    name: str
    shape: Tuple[int, ...]
    offset: int
    scale: float
    zero_point: int

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape))

    @property
    def end(self) -> int:
        return self.offset + self.nbytes


@dataclass(frozen=True)
class DetectHeadLayout:
    """Resolved placement of the heads inside one egress buffer."""
    heads: Tuple[HeadSpec, ...]

    @property
    def required_bytes(self) -> int:
        return max(h.end for h in self.heads)

    def spec(self, name: str) -> HeadSpec:
        for h in self.heads:
            if h.name == name:
                return h
        raise KeyError(name)

    def scales(self) -> Dict[str, Tuple[float, int]]:
        return {h.name: (h.scale, h.zero_point) for h in self.heads}

    def unpack(self, raw: np.ndarray) -> Dict[str, np.ndarray]:
        """Return the heads as int8 views of ``raw`` (no copy, no dequantization)."""
        flat = np.asarray(raw)
        if flat.dtype != np.int8:
            raise TypeError(f"egress must be int8, got {flat.dtype}")
        flat = flat.reshape(-1)
        if flat.size < self.required_bytes:
            raise ValueError(
                f"egress holds {flat.size} bytes but the declared head layout needs {self.required_bytes}"
            )
        return {h.name: flat[h.offset:h.end].reshape(h.shape) for h in self.heads}

    def dequantize(self, views: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """float32 ``(q - zero_point) * scale`` for every head in ``views``."""
        out: Dict[str, np.ndarray] = {}
        for h in self.heads:
            q = views[h.name]
            out[h.name] = (q.astype(np.float32) - np.float32(h.zero_point)) * np.float32(h.scale)
        return out

    def quantize(self, heads: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """int8 ``round(x / scale) + zero_point`` with saturation, per head."""
        out: Dict[str, np.ndarray] = {}
        for h in self.heads:
            x = np.asarray(heads[h.name], dtype=np.float32)
            q = np.rint(x / np.float32(h.scale)) + h.zero_point
            out[h.name] = np.clip(q, -128, 127).astype(np.int8)
        return out

    def pack(self, heads: Mapping[str, np.ndarray], egress_bytes: Optional[int] = None) -> np.ndarray:
        """Inverse of :meth:`unpack`: write int8 heads into a fresh egress buffer."""
        size = self.required_bytes if egress_bytes is None else int(egress_bytes)
        if size < self.required_bytes:
            raise ValueError(f"egress_bytes {size} < required {self.required_bytes}")
        raw = np.zeros(size, dtype=np.int8)
        for h in self.heads:
            q = np.asarray(heads[h.name])
            if q.dtype != np.int8 or tuple(q.shape) != tuple(h.shape):
                raise ValueError(f"{h.name}: expected int8 {h.shape}, got {q.dtype} {tuple(q.shape)}")
            raw[h.offset:h.end] = q.reshape(-1)
        return raw


@dataclass(frozen=True)
class HeadStatus:
    """Outcome of :func:`resolve_head_layout`; ``reason`` is human-readable either way."""
    present: bool
    reason: str
    declared_bytes: int
    egress_bytes: int
    layout: Optional[DetectHeadLayout] = None


def declared_head_bytes(manifest: Mapping[str, Any], names: Optional[Tuple[str, ...]] = None) -> int:
    """Total int8 bytes of the heads ``names`` (default: the manifest task's) in ``output_shapes``, 0 if any is
    missing."""
    names = head_names_for(manifest) if names is None else names
    shapes = manifest.get("output_shapes") or {}
    total = 0
    for name in names:
        shape = shapes.get(name)
        if not shape:
            return 0
        total += int(np.prod([int(d) for d in shape]))
    return total


def _absent(reason: str, declared: int, egress: int) -> HeadStatus:
    return HeadStatus(present=False, reason=reason, declared_bytes=declared, egress_bytes=egress)


def resolve_head_layout(manifest: Optional[Mapping[str, Any]], egress_bytes: int,
                        names: Optional[Tuple[str, ...]] = None) -> HeadStatus:
    """Decide whether ``egress_bytes`` of device output carry the raw heads ``names``; by default the manifest
    task's (six detect heads, nine for ``pose``).

    Fails closed: every check below must pass for ``present`` to be True.
    """
    manifest = manifest or {}
    names = head_names_for(manifest) if names is None else tuple(names)
    egress_bytes = int(egress_bytes)
    declared = declared_head_bytes(manifest, names)
    shapes = manifest.get("output_shapes") or {}

    if manifest.get("fused_dfl"):
        return _absent("container is fused-DFL: egress carries decoded boxes/scores, not raw heads",
                       declared, egress_bytes)
    if declared == 0:
        return _absent(f"manifest output_shapes does not name all {len(names)} heads {list(names)}", declared,
                       egress_bytes)
    for name in names:
        if len(shapes[name]) != HEAD_RANK:
            return _absent(f"output_shapes[{name}] is not a 4-D shape: {shapes[name]}", declared, egress_bytes)

    layout_meta = manifest.get("head_layout")
    if not isinstance(layout_meta, Mapping):
        return _absent(
            f"manifest declares no head_layout (per-head egress offset, scale, zero_point); "
            f"the runtime does not guess a device packing order. Heads need {declared} bytes, "
            f"egress is {egress_bytes} bytes", declared, egress_bytes)

    specs = []
    for name in names:
        entry = layout_meta.get(name)
        if not isinstance(entry, Mapping):
            return _absent(f"head_layout lacks an entry for {name}", declared, egress_bytes)
        try:
            offset = int(entry["offset"])
            scale = float(entry["scale"])
            zero_point = int(entry.get("zero_point", 0))
        except (KeyError, TypeError, ValueError) as ex:
            return _absent(f"head_layout[{name}] is malformed: {ex!r}", declared, egress_bytes)
        if offset < 0:
            return _absent(f"head_layout[{name}].offset is negative", declared, egress_bytes)
        if not (scale > 0.0) or not np.isfinite(scale):
            return _absent(f"head_layout[{name}].scale must be a positive finite float", declared, egress_bytes)
        if not (-128 <= zero_point <= 127):
            return _absent(f"head_layout[{name}].zero_point {zero_point} is outside int8", declared, egress_bytes)
        specs.append(HeadSpec(name=name, shape=tuple(int(d) for d in shapes[name]),
                              offset=offset, scale=scale, zero_point=zero_point))

    ordered = sorted(specs, key=lambda h: h.offset)
    for a, b in zip(ordered, ordered[1:]):
        if a.end > b.offset:
            return _absent(f"head_layout ranges overlap: {a.name} [{a.offset}, {a.end}) and "
                           f"{b.name} [{b.offset}, {b.end})", declared, egress_bytes)
    layout = DetectHeadLayout(heads=tuple(specs))
    if layout.required_bytes > egress_bytes:
        return _absent(f"head_layout needs {layout.required_bytes} bytes of egress but the session "
                       f"allocates {egress_bytes}", declared, egress_bytes, )
    return HeadStatus(present=True,
                      reason=f"{len(names)} heads declared in head_layout, {layout.required_bytes} of "
                             f"{egress_bytes} egress bytes",
                      declared_bytes=declared, egress_bytes=egress_bytes, layout=layout)


def contiguous_head_layout(shapes: Mapping[str, Any], scales: Mapping[str, Tuple[float, int]],
                           names: Tuple[str, ...] = HEAD_NAMES) -> Dict[str, Dict[str, Any]]:
    """Build a ``head_layout`` manifest entry that packs the heads back to back in
    ``HEAD_NAMES`` order. A convenience for tests and for a compiler that does
    emit heads — nothing in the runtime assumes this order."""
    out: Dict[str, Dict[str, Any]] = {}
    offset = 0
    for name in names:
        scale, zero_point = scales[name]
        out[name] = {"offset": offset, "scale": float(scale), "zero_point": int(zero_point)}
        offset += int(np.prod([int(d) for d in shapes[name]]))
    return out

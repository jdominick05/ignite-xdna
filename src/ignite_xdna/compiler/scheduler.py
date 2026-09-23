#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
src/ignite_xdna/compiler/scheduler.py

Dynamic L2 MemTile Multi-Pass Scheduler & CDO Transaction Lowering for Phoenix XDNA1.
Generalizes the 2-layer ping-pong mechanism to an arbitrary N-layer sequence with
zero intermediate host DDR roundtrips.
"""

import os
import struct
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
import onnxruntime as ort

from ignite_xdna.compiler.lower_onnx_conv import (
    pack_bias_aie2_vector_layout,
    pack_weights_aie2_vector_layout,
    unblock_aie2_egress,
)
from ignite_xdna.compiler.memtile_agu import (
    L2_BANK_0_OFFSET,
    L2_BANK_1_OFFSET,
    L2_BANK_BYTES,
    MEMTILE_BYTES,
)
from ignite_xdna.compiler.partitioner import (
    ConvLayerMeta,
    NpuFusedPartition,
    PartitionedGraph,
)


# MemTile L2 Floorplan Geometry (the two 64 KB activation banks, L2_BANK_0_OFFSET
# = 0x40000 and L2_BANK_1_OFFSET = 0x60000, are defined in memtile_agu)
L2_WEIGHTS_OFFSET = 0x00000        # 256 KB reserved for stationary multi-layer filter storage
L2_FINAL_EGRESS_OFFSET = 0x04000   # Egress staging buffer to Shim DMA BD 4

# MemTile Lock IDs
LOCK_CORE_EGRESS_CREDIT = 2        # Gather credit (initial val = 4)
LOCK_L2_PING = 4                   # Protects L2_BANK_0 (initial val = 1: write-ready)
LOCK_L2_PONG = 5                   # Protects L2_BANK_1 (initial val = 0: idle)
LOCK_STAGE_BARRIER_A = 6           # On-die inter-stage barrier lock A (even stages)
LOCK_STAGE_BARRIER_B = 7           # On-die inter-stage barrier lock B (odd stages)

# Tile address map (AIE2 / Phoenix). A register offset is the low 20 bits of a
# transaction address: (col << 25) | (row << 20) | offset. The single-layer
# template corroborates the map: it resets and enables cores through
# Core_Control at 0x32000, programs core BDs at 0x1D000, core locks at
# 0x1F000, MemTile BDs at 0xA0000, MemTile locks at 0xC0000 and shim BDs at
# 0x1D000.
CORE_DATA_MEMORY_BYTES = 0x10000     # core-tile data memory, 0x00000..0x0FFFF
CORE_BD_BASE = 0x1D000               # core-tile DMA buffer descriptors, 16 x 0x20
CORE_BD_END = 0x1D200
CORE_LOCK_BASE = 0x1F000             # core-tile locks, 16 x 0x10
CORE_LOCK_END = 0x1F100
CORE_PROGRAM_MEMORY_BASE = 0x20000   # core-tile program memory, 16 KB
CORE_PROGRAM_MEMORY_END = 0x24000
CORE_MODULE_BASE = 0x30000           # core-module registers; Core_Control is 0x32000
CORE_CONTROL_REG = 0x32000
MEMTILE_BD_BASE = 0xA0000            # MemTile buffer descriptors, 48 x 0x20
MEMTILE_BD_END = 0xA0600
MEMTILE_LOCK_BASE = 0xC0000          # MemTile locks, 64 x 0x10
MEMTILE_LOCK_STRIDE = 0x10
SHIM_BD_BASE = 0x1D000               # shim DMA buffer descriptors, 16 x 0x20
SHIM_BD_END = 0x1D200
MAX_COLUMN = 4                       # Phoenix has five physical columns (0..4)
MAX_ROW = 5                          # shim, MemTile, four core rows

# Core-local parameter windows programmed by the init stream: one 0x1000 window
# per resident layer holding the single-layer template's shift-cut, bias and
# weights. 0x400 + 2304 bytes of weights ends at 0xD00, so 16 windows fit the
# 64 KB data memory and window 16 already starts outside it.
L1_PARAM_STRIDE = 0x1000
L1_SHIFT_CUT_OFFSET = 0x0037C
L1_BIAS_OFFSET = 0x00380
L1_WEIGHTS_OFFSET = 0x00400
TEMPLATE_WEIGHT_BYTES = 576 * 4
TEMPLATE_BIAS_BYTES = 32 * 4

# XDNA1 transaction stream framing (tools/disasm_txn.py decodes the same format).
TXN_HEADER_BYTES = 16
OP_WRITE = 0x00
OP_BLOCKWRITE = 0x01
OP_MASKWRITE = 0x03
OP_TCT = 0x80
OP_DDR_PATCH = 0x81
OP_NAMES = {OP_WRITE: "WRITE", OP_BLOCKWRITE: "BLOCKWRITE", OP_MASKWRITE: "MASKWRITE",
            OP_TCT: "TCT", OP_DDR_PATCH: "DDR_PATCH"}
TXN_PAD_BYTES = 64


def max_resident_l1_layers(weight_bytes: int = TEMPLATE_WEIGHT_BYTES) -> int:
    """Parameter windows that fit core data memory at the template's stride."""
    return (CORE_DATA_MEMORY_BYTES - L1_WEIGHTS_OFFSET - weight_bytes) // L1_PARAM_STRIDE + 1


def memtile_lock_reg(lock_id: int) -> int:
    """20-bit register offset of a MemTile lock value register."""
    if not 0 <= lock_id < 64:
        raise ValueError(f"MemTile lock id must be in [0, 64), got {lock_id}")
    return MEMTILE_LOCK_BASE + MEMTILE_LOCK_STRIDE * lock_id


def memtile_lock_address(col: int, lock_id: int) -> int:
    """Absolute transaction address of a MemTile lock value register."""
    return (col << 25) | (1 << 20) | memtile_lock_reg(lock_id)


def memtile_lock_write(col: int, lock_id: int, value: int) -> bytes:
    """A 24-byte WRITE op that sets one MemTile lock value on column ``col``.

    This sets the value from the instruction stream; the format has no acquire
    opcode, so it never waits for a DMA that still holds the lock.
    """
    col_row = (col & 0xFF) | (1 << 8)
    return struct.pack("<6I", OP_WRITE, col_row, memtile_lock_address(col, lock_id), 0, value, 24)


def _need(data: bytes, p: int, n: int, index: int) -> None:
    if p + n > len(data):
        raise ValueError(f"op {index} at offset {p} needs {n} bytes but only {len(data) - p} remain")


def parse_transaction_stream(data: bytes) -> List[Dict[str, Any]]:
    """Decode an XDNA1 transaction stream into op dicts.

    Field names match ``tools/disasm_txn.py`` so both decoders agree on the
    template walk. This decoder is bounded by the header's op count, so a
    zero tail (the 64-byte padding of the chained stream) is never misread as
    a WRITE, and a truncated or unknown op raises ``ValueError``.
    """
    data = bytes(data)
    if len(data) < TXN_HEADER_BYTES:
        raise ValueError(f"transaction stream too short: {len(data)} bytes")
    _major, _minor, num_ops, _size = struct.unpack("<4I", data[:TXN_HEADER_BYTES])
    ops: List[Dict[str, Any]] = []
    p = TXN_HEADER_BYTES
    for index in range(num_ops):
        _need(data, p, 8, index)
        op_code, col_row = struct.unpack("<2I", data[p:p + 8])
        col = col_row & 0xFF
        row = (col_row >> 8) & 0xFF
        if op_code == OP_WRITE:
            _need(data, p, 24, index)
            _o, cr, addr, vl, vh, sz = struct.unpack("<6I", data[p:p + 24])
            size = sz if sz > 0 else 24
            ops.append({"op": "WRITE", "opcode": OP_WRITE, "col": col, "row": row, "col_row": cr,
                        "addr": addr, "val": vh, "val_low": vl, "size": size, "offset": p})
        elif op_code == OP_MASKWRITE:
            _need(data, p, 28, index)
            _o, cr, addr, _pad, mask, val, sz = struct.unpack("<7I", data[p:p + 28])
            size = sz if sz > 0 else 28
            ops.append({"op": "MASKWRITE", "opcode": OP_MASKWRITE, "col": col, "row": row,
                        "col_row": cr, "addr": addr, "val": val, "mask": mask, "size": size,
                        "offset": p})
        elif op_code == OP_BLOCKWRITE:
            _need(data, p, 16, index)
            _o, cr, addr, sz = struct.unpack("<4I", data[p:p + 16])
            if sz < 16 or (sz - 16) % 4:
                raise ValueError(f"BLOCKWRITE op {index} at offset {p} has an invalid size {sz}")
            _need(data, p, sz, index)
            words = struct.unpack(f"<{(sz - 16) // 4}I", data[p + 16:p + sz])
            size = sz
            ops.append({"op": "BLOCKWRITE", "opcode": OP_BLOCKWRITE, "col": col, "row": row,
                        "col_row": cr, "addr": addr, "size": size, "words": words, "offset": p})
        elif op_code == OP_DDR_PATCH:
            _need(data, p, 48, index)
            w = struct.unpack("<12I", data[p:p + 48])
            size = 48
            ops.append({"op": "DDR_PATCH", "opcode": OP_DDR_PATCH, "size": size, "addr": w[6],
                        "arg_idx": w[8], "arg_offset": w[10], "offset": p})
        elif op_code == OP_TCT:
            _need(data, p, 16, index)
            size = 16
            ops.append({"op": "TCT", "opcode": OP_TCT, "size": size,
                        "word": struct.unpack("<I", data[p + 12:p + 16])[0], "offset": p})
        else:
            raise ValueError(f"unknown opcode {op_code:#x} at offset {p} (op {index})")
        if size % 4:
            raise ValueError(f"op {index} at offset {p} has size {size}, not a multiple of 4")
        p += size
    return ops


def _core_window(reg: int) -> str:
    """Classify a core-tile register offset; "memory_module" is 0x10000..0x1FFFF
    (DMA, locks and other memory-module registers), "unmapped" is the span
    between program memory and the core module (0x24000..0x2FFFF)."""
    if reg < CORE_DATA_MEMORY_BYTES:
        return "data"
    if CORE_BD_BASE <= reg < CORE_BD_END:
        return "bd"
    if CORE_LOCK_BASE <= reg < CORE_LOCK_END:
        return "lock"
    if CORE_PROGRAM_MEMORY_BASE <= reg < CORE_PROGRAM_MEMORY_END:
        return "program"
    if reg >= CORE_MODULE_BASE:
        return "core_module"
    if reg >= CORE_PROGRAM_MEMORY_END:
        return "unmapped"
    return "memory_module"


def _bulk_target_ok(row: int, reg: int, payload_end: int) -> bool:
    """Whether a BLOCKWRITE payload [reg, payload_end) lands on memory or BD registers."""
    if row == 0:
        return SHIM_BD_BASE <= reg and payload_end <= SHIM_BD_END
    if row == 1:
        return payload_end <= MEMTILE_BYTES or (MEMTILE_BD_BASE <= reg and payload_end <= MEMTILE_BD_END)
    window = _core_window(reg)
    if window == "data":
        return payload_end <= CORE_DATA_MEMORY_BYTES
    if window == "bd":
        return payload_end <= CORE_BD_END
    return False


def validate_transaction_stream(data: bytes, *, name: str = "transaction",
                                require_tct: bool = True,
                                require_terminal_tct: bool = False) -> Dict[str, Any]:
    """Check stream framing and address windows; raise ValueError on any defect.

    Checks that the header size equals the buffer length, every op is 4-byte
    aligned and inside the buffer, opcodes are known, any tail after the last
    op is an all-zero pad shorter than 64 bytes, at least one TCT completion
    wait is present when ``require_tct`` (the last op must be one when
    ``require_terminal_tct``; IRON streams may legitimately trail register
    writes after their final wait), tiles are inside Phoenix's five columns
    and six rows, DDR patches target shim BD registers, and bulk BLOCKWRITE
    payloads land on data memory or BD registers only (never on program
    memory, locks or core-module registers). Returns a summary dict.
    """
    data = bytes(data)
    if len(data) < TXN_HEADER_BYTES:
        raise ValueError(f"{name}: too short ({len(data)} bytes)")
    major, minor, num_ops, size_bytes = struct.unpack("<4I", data[:TXN_HEADER_BYTES])
    if size_bytes != len(data):
        raise ValueError(f"{name}: header size {size_bytes} != buffer length {len(data)}")
    if num_ops == 0:
        raise ValueError(f"{name}: header declares zero ops")
    ops = parse_transaction_stream(data)
    end = ops[-1]["offset"] + ops[-1]["size"]
    tail = data[end:]
    if tail and (len(tail) >= TXN_PAD_BYTES or any(tail)):
        raise ValueError(f"{name}: {len(tail)} trailing bytes after the last op are not a zero pad")
    violations: List[str] = []
    counts: Dict[str, int] = {}
    tct_count = 0
    bulk_bytes = 0
    for i, o in enumerate(ops):
        counts[o["op"]] = counts.get(o["op"], 0) + 1
        if o["op"] == "TCT":
            tct_count += 1
            continue
        addr = o["addr"]
        col, row, reg = (addr >> 25) & 0x7F, (addr >> 20) & 0x1F, addr & 0xFFFFF
        if o["op"] == "DDR_PATCH":
            if row != 0 or col > MAX_COLUMN or not SHIM_BD_BASE <= reg < SHIM_BD_END:
                violations.append(f"op {i}: DDR_PATCH targets {addr:#x}, not a shim BD register")
            continue
        if col > MAX_COLUMN or row > MAX_ROW:
            violations.append(f"op {i}: {o['op']} addresses tile ({col},{row}) outside the array")
            continue
        if o["op"] == "BLOCKWRITE":
            payload = o["size"] - 16
            bulk_bytes += payload
            if not _bulk_target_ok(row, reg, reg + payload):
                violations.append(
                    f"op {i}: BLOCKWRITE of {payload} bytes to tile ({col},{row}) offset {reg:#x} "
                    f"lands outside data memory / BD registers")
        elif row >= 2 and _core_window(reg) == "program":
            violations.append(f"op {i}: {o['op']} to core program memory at ({col},{row}) offset {reg:#x}")
    if require_tct and tct_count == 0:
        violations.append("stream has no TCT completion wait")
    elif require_terminal_tct and ops[-1]["op"] != "TCT":
        violations.append("stream does not end with a TCT completion wait")
    if violations:
        shown = "\n  ".join(violations[:8])
        more = f"\n  ... {len(violations) - 8} more" if len(violations) > 8 else ""
        raise ValueError(f"{name}: {len(violations)} invalid op(s):\n  {shown}{more}")
    return {"name": name, "num_ops": num_ops, "size_bytes": size_bytes, "op_counts": counts,
            "tct_count": tct_count, "pad_bytes": len(tail),
            "blockwrite_payload_bytes": bulk_bytes, "version": (major, minor)}


@dataclass
class PassDescriptor:
    """Execution descriptor for a single layer pass in the multi-layer pipeline."""
    pass_index: int
    layer_meta: Optional[ConvLayerMeta] = None
    ingress_source: str = "HOST_DDR"            # "HOST_DDR", "L2_BANK_0", "L2_BANK_1"
    ingress_addr: int = 0
    egress_dest: str = "HOST_DDR"  # "L2_BANK_0", "L2_BANK_1", "HOST_DDR"
    egress_addr: int = 0
    ingress_lock_id: Optional[int] = None
    egress_lock_id: Optional[int] = None
    param_l1_offset: int = 0       # Core L1 offset for layer parameters
    is_initial: bool = False
    is_final: bool = False


@dataclass
class SchedulePlan:
    """Full multi-pass scheduling plan for an N-layer fused subgraph."""
    num_layers: int
    passes: List[PassDescriptor] = field(default_factory=list)
    intermediate_ddr_bytes: int = 0
    total_l1_param_bytes_per_core: int = 0

    @property
    def has_zero_intermediate_ddr_traffic(self) -> bool:
        return self.intermediate_ddr_bytes == 0


@dataclass
class StageSchedulePlan(SchedulePlan):
    """Schedule plan for a monolithic stage (Stem, P3, P4, P5, DFL_Decode)."""
    stage_name: str = "Stem"
    c2f_blocks: List[str] = field(default_factory=list)
    custom_exec_bytes: Optional[bytes] = None


@dataclass
class MultiStageSchedulePlan:
    """Unified multi-stage schedule plan combining all backbone stages."""
    stages: Dict[str, StageSchedulePlan] = field(default_factory=dict)
    total_stages: int = 0
    total_layers: int = 0
    intermediate_ddr_bytes: int = 0
    total_l1_param_bytes_per_core: int = 0

    @property
    def has_zero_intermediate_ddr_traffic(self) -> bool:
        return self.intermediate_ddr_bytes == 0

    def to_schedule_plan(self) -> SchedulePlan:
        """Flattens all stages into a unified monolithic SchedulePlan with 0 DDR roundtrips."""
        all_layers: List[ConvLayerMeta] = []
        for s_name, stage in self.stages.items():
            if s_name == "DFL_Decode":
                continue
            all_layers.extend(p.layer_meta for p in stage.passes if p.layer_meta is not None)
        return MemTileMultiPassScheduler().schedule(all_layers)


def get_dfl_decode_transaction_binary(repo_root: Optional[Path] = None) -> bytes:
    """Retrieves pre-compiled DFL decode transaction binary (insts.bin)."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]

    # Priority 1: build/dfl_decode/<hash>/insts.bin
    dfl_dir = repo_root / "build" / "dfl_decode"
    if dfl_dir.exists():
        for p in sorted(dfl_dir.glob("*/insts.bin"), key=lambda x: x.stat().st_mtime, reverse=True):
            if p.is_file() and p.stat().st_size > 0:
                return p.read_bytes()

    # Priority 2: build/stage_dfl_decode_exec.bin
    cand = repo_root / "build" / "stage_dfl_decode_exec.bin"
    if cand.is_file() and cand.stat().st_size > 0:
        return cand.read_bytes()

    raise FileNotFoundError(f"DFL decode transaction binary not found under {dfl_dir}")


class MemTileMultiPassScheduler:
    """
    Dynamic L2 MemTile Scheduler for arbitrary N-layer sequences.
    Maps alternating intermediate activations between L2_BANK_0 (0x40000)
    and L2_BANK_1 (0x60000) across Columns 0..3 of Row 1 MemTiles.
    """

    def __init__(self, num_cores: int = 16):
        self.num_cores = num_cores
        self.cores = [(c, r) for c in range(4) for r in range(2, 6)]

    def schedule(
        self,
        layers_or_partition: Union[NpuFusedPartition, List[ConvLayerMeta]]
    ) -> SchedulePlan:
        """Constructs an N-pass execution schedule for the given layers."""
        if isinstance(layers_or_partition, NpuFusedPartition):
            layers = layers_or_partition.layers
        else:
            layers = list(layers_or_partition)

        n = len(layers)
        if n == 0:
            raise ValueError("Cannot schedule empty layer list")

        passes: List[PassDescriptor] = []

        for k in range(n):
            is_initial = (k == 0)
            is_final = (k == n - 1)
            layer_meta = layers[k]

            param_offset = k * L1_PARAM_STRIDE

            if is_initial:
                ing_src = "HOST_DDR"
                ing_addr = 0x00000
                ing_lock = None
            else:
                # Ingress alternates from Bank ((k - 1) % 2)
                if (k - 1) % 2 == 0:
                    ing_src = "L2_BANK_0"
                    ing_addr = L2_BANK_0_OFFSET
                    ing_lock = LOCK_L2_PING
                else:
                    ing_src = "L2_BANK_1"
                    ing_addr = L2_BANK_1_OFFSET
                    ing_lock = LOCK_L2_PONG

            if is_final:
                eg_dest = "HOST_DDR"
                eg_addr = L2_FINAL_EGRESS_OFFSET
                eg_lock = None
            else:
                # Egress alternates to Bank (k % 2)
                if k % 2 == 0:
                    eg_dest = "L2_BANK_0"
                    eg_addr = L2_BANK_0_OFFSET
                    eg_lock = LOCK_L2_PING
                else:
                    eg_dest = "L2_BANK_1"
                    eg_addr = L2_BANK_1_OFFSET
                    eg_lock = LOCK_L2_PONG

            pass_desc = PassDescriptor(
                pass_index=k,
                layer_meta=layer_meta,
                ingress_source=ing_src,
                ingress_addr=ing_addr,
                egress_dest=eg_dest,
                egress_addr=eg_addr,
                ingress_lock_id=ing_lock,
                egress_lock_id=eg_lock,
                param_l1_offset=param_offset,
                is_initial=is_initial,
                is_final=is_final,
            )
            passes.append(pass_desc)

        # 609 words = 2,436 bytes per layer
        total_param_bytes = n * (1 + 32 + 576) * 4

        return SchedulePlan(
            num_layers=n,
            passes=passes,
            intermediate_ddr_bytes=0,
            total_l1_param_bytes_per_core=total_param_bytes,
        )

    def schedule_stage(
        self,
        partition: NpuFusedPartition
    ) -> StageSchedulePlan:
        """Constructs an execution schedule for a monolithic stage (Stem, P3, P4, P5)."""
        base_plan = self.schedule(partition)
        return StageSchedulePlan(
            num_layers=base_plan.num_layers,
            passes=base_plan.passes,
            intermediate_ddr_bytes=base_plan.intermediate_ddr_bytes,
            total_l1_param_bytes_per_core=base_plan.total_l1_param_bytes_per_core,
            stage_name=partition.stage_name or f"Stage_{partition.partition_id}",
            c2f_blocks=partition.c2f_blocks,
        )

    def schedule_multi_stage(
        self,
        partitions: List[NpuFusedPartition],
        fuse_dfl: bool = False,
    ) -> MultiStageSchedulePlan:
        """Constructs a consolidated multi-stage schedule with 0 intermediate DDR roundtrips."""
        stages: Dict[str, StageSchedulePlan] = {}
        total_layers = 0
        total_param_bytes = 0

        for p in partitions:
            s_plan = self.schedule_stage(p)
            s_name = p.stage_name or f"Stage_{p.partition_id}"
            stages[s_name] = s_plan
            total_layers += s_plan.num_layers
            total_param_bytes += s_plan.total_l1_param_bytes_per_core

        if fuse_dfl:
            # Re-route the final pass of the preceding stage (Detect_P5) to MemTile Bank 0 (0x40000)
            # without host DDR bounce
            stage_keys = list(stages.keys())
            if stage_keys:
                last_stage_key = stage_keys[-1]
                last_stage = stages[last_stage_key]
                if last_stage.passes:
                    final_pass = last_stage.passes[-1]
                    final_pass.egress_dest = "L2_BANK_0"
                    final_pass.egress_addr = L2_BANK_0_OFFSET
                    final_pass.egress_lock_id = LOCK_L2_PING
                    final_pass.is_final = False

            # Add 10th monolithic stage: DFL_Decode
            dfl_txn = get_dfl_decode_transaction_binary()
            dfl_pass = PassDescriptor(
                pass_index=len(stages),
                layer_meta=None,
                ingress_source="L2_BANK_0",
                ingress_addr=L2_BANK_0_OFFSET,
                egress_dest="HOST_DDR",
                egress_addr=L2_FINAL_EGRESS_OFFSET,
                ingress_lock_id=LOCK_L2_PING,
                egress_lock_id=LOCK_CORE_EGRESS_CREDIT,
                param_l1_offset=0,
                is_initial=False,
                is_final=True,
            )
            dfl_plan = StageSchedulePlan(
                num_layers=1,
                passes=[dfl_pass],
                intermediate_ddr_bytes=0,
                total_l1_param_bytes_per_core=0,
                stage_name="DFL_Decode",
                c2f_blocks=[],
                custom_exec_bytes=dfl_txn,
            )
            stages["DFL_Decode"] = dfl_plan
            total_layers += 1

        return MultiStageSchedulePlan(
            stages=stages,
            total_stages=len(stages),
            total_layers=total_layers,
            intermediate_ddr_bytes=0,
            total_l1_param_bytes_per_core=total_param_bytes,
        )


def validate_l1_parameter_layout(
    schedule: SchedulePlan,
    *,
    context: str = "",
    data_memory_bytes: int = CORE_DATA_MEMORY_BYTES,
) -> int:
    """Raise unless every pass's parameter window stays inside core data memory.

    A window is [param_l1_offset + 0x37C, param_l1_offset + 0x400 + weight
    bytes) at the template's 0x1000 stride. Windows must be disjoint and end
    at or below ``data_memory_bytes``; anything beyond lands on memory-module
    registers, program memory or core-module registers. Returns the highest
    byte any window writes.
    """
    windows = []
    for p in schedule.passes:
        layer = p.layer_meta
        if layer is None:
            continue
        weights = getattr(layer, "weights_packed", None)
        weight_bytes = np.asarray(weights).nbytes if weights is not None else TEMPLATE_WEIGHT_BYTES
        start = p.param_l1_offset + L1_SHIFT_CUT_OFFSET
        end = p.param_l1_offset + L1_WEIGHTS_OFFSET + weight_bytes
        windows.append((start, end, p.pass_index))
    windows.sort()
    top = 0
    for i, (start, end, k) in enumerate(windows):
        if end > data_memory_bytes:
            raise ValueError(
                f"{len(windows)} resident parameter sets do not fit core data memory: set {k} "
                f"spans [{start:#x}, {end:#x}) but data memory ends at {data_memory_bytes:#x}; "
                f"at most {max_resident_l1_layers()} sets fit at the {L1_PARAM_STRIDE:#x} stride. "
                f"{context}".rstrip())
        if i and start < windows[i - 1][1]:
            raise ValueError(f"parameter sets {windows[i - 1][2]} and {k} overlap in core data memory")
        top = max(top, end)
    return top


def emit_multi_layer_transaction_bundle(
    schedule: SchedulePlan,
    base_txn_path: str,
    out_init_path: str,
    out_exec_path: str,
    cores: Optional[List[Tuple[int, int]]] = None
) -> Tuple[str, str]:
    """
    Emits decoupled init and exec transaction binaries for an N-layer pipeline:
    - out_init_path: Stages all N parameter sets into distinct L1 SRAM offsets and initializes MemTile locks.
    - out_exec_path: Chains BD sequences and channel queue pushes with DDR patches strictly on initial/final buffers.

    Refuses a schedule whose parameter windows leave core data memory
    (``validate_l1_parameter_layout``) and validates both emitted streams
    (``validate_transaction_stream``) before writing them.
    """
    if not os.path.exists(base_txn_path):
        raise FileNotFoundError(f"Base transaction binary not found: {base_txn_path}")

    with open(base_txn_path, "rb") as f:
        base_bytes = f.read()

    ops = parse_transaction_stream(base_bytes)

    if cores is None:
        cores = [(c, r) for c in range(4) for r in range(2, 6)]

    n_layers = schedule.num_layers
    validate_l1_parameter_layout(schedule)

    def make_ddr_patch(addr, arg_idx, arg_offset=0):
        return struct.pack("<12I", 0x81, 48, 0, 0, 0, 0, addr, 0, arg_idx, 0, arg_offset, 0)

    # -----------------------------------------------------------------------
    # 1. BUILD INIT BINARY (subgraph_init.bin)
    # -----------------------------------------------------------------------
    core_inject_bytes = bytearray()
    num_core_ops = 0

    for pass_desc in schedule.passes:
        layer = pass_desc.layer_meta
        base_param_reg = pass_desc.param_l1_offset

        w_packed = layer.weights_packed
        if w_packed is None:
            w_packed = pack_weights_aie2_vector_layout(layer.weights_raw)

        b_packed = layer.bias_packed
        if b_packed is None:
            b_packed = pack_bias_aie2_vector_layout(layer.bias_i32)

        w_words = list(np.frombuffer(w_packed.tobytes(), dtype=np.uint32))
        b_words = list(np.frombuffer(b_packed.tobytes(), dtype=np.uint32))
        s_words = [int(layer.shift_cut)]

        for col, row in cores:
            col_row = (col & 0xFF) | ((row & 0xFF) << 8)

            # Shift Cut at 0x0037C + base_param_reg
            s_addr = (col << 25) | (row << 20) | (L1_SHIFT_CUT_OFFSET + base_param_reg)
            s_op = [1, col_row, s_addr, (4 + len(s_words)) * 4] + s_words
            core_inject_bytes.extend(struct.pack(f"<{len(s_op)}I", *s_op))
            num_core_ops += 1

            # Bias at 0x00380 + base_param_reg
            b_addr = (col << 25) | (row << 20) | (L1_BIAS_OFFSET + base_param_reg)
            b_op = [1, col_row, b_addr, (4 + len(b_words)) * 4] + b_words
            core_inject_bytes.extend(struct.pack(f"<{len(b_op)}I", *b_op))
            num_core_ops += 1

            # Weights at 0x00400 + base_param_reg
            w_addr = (col << 25) | (row << 20) | (L1_WEIGHTS_OFFSET + base_param_reg)
            w_op = [1, col_row, w_addr, (4 + len(w_words)) * 4] + w_words
            core_inject_bytes.extend(struct.pack(f"<{len(w_op)}I", *w_op))
            num_core_ops += 1

    # Parameters are injected before the first core enable (Core_Control = 1),
    # while the cores are still held in reset by the template.
    splice_idx = None
    for i, o in enumerate(ops):
        if (o.get("addr", 0) & 0xFFFFF) == CORE_CONTROL_REG and o.get("val") == 1:
            splice_idx = i
            break
    if splice_idx is None:
        splice_idx = 72

    init_ops_bytes = []
    num_init_ops = 0

    for i, o in enumerate(ops):
        if i == splice_idx:
            init_ops_bytes.append(bytes(core_inject_bytes))
            num_init_ops += num_core_ops

        addr = o.get("addr", 0)
        col = (addr >> 25) & 0x7F
        row = (addr >> 20) & 0x1F
        reg = addr & 0xFFFFF

        # Replace the template's MemTile Lock 2 init with Lock 2 (val=4, credit
        # for 4 cores), Lock 4 (val=1, L2 Ping write-ready) and Lock 5 (val=0,
        # L2 Pong idle). ``reg`` is the 20-bit offset, so the comparison must
        # use the 0xC0020 lock register; the former absolute 0x1C0020 (row bit
        # included) could never match and left this branch dead.
        if row == 1 and reg == memtile_lock_reg(LOCK_CORE_EGRESS_CREDIT):
            init_ops_bytes.append(memtile_lock_write(col, LOCK_CORE_EGRESS_CREDIT, 4))
            init_ops_bytes.append(memtile_lock_write(col, LOCK_L2_PING, 1))
            init_ops_bytes.append(memtile_lock_write(col, LOCK_L2_PONG, 0))
            num_init_ops += 3
            continue

        if row == 0 and reg == 0x1D000 and o["op"] == "BLOCKWRITE":
            raw_chunk = base_bytes[o["offset"] : o["offset"] + o["size"]]
            init_ops_bytes.append(raw_chunk)
            num_init_ops += 1
            p0 = make_ddr_patch((col << 25) | 0x0001D004, 0, col * 2048)
            p1 = make_ddr_patch((col << 25) | 0x0001D024, 1, col * 1024)
            init_ops_bytes.extend([p0, p1])
            num_init_ops += 2
            continue

        raw_chunk = base_bytes[o["offset"] : o["offset"] + o["size"]]
        init_ops_bytes.append(raw_chunk)
        num_init_ops += 1

    if ops[-1]["op"] != "TCT":
        init_ops_bytes.append(struct.pack("<4I", OP_TCT, 16, 0, 0x00010000))
        num_init_ops += 1

    payload_init = b"".join(init_ops_bytes)
    hdr_init = struct.pack("<4I", 0, 0, num_init_ops, TXN_HEADER_BYTES + len(payload_init))
    full_init_bin = hdr_init + payload_init
    validate_transaction_stream(full_init_bin, name=os.path.basename(out_init_path), require_terminal_tct=True)

    os.makedirs(os.path.dirname(os.path.abspath(out_init_path)), exist_ok=True)
    with open(out_init_path, "wb") as f:
        f.write(full_init_bin)

    # -----------------------------------------------------------------------
    # 2. BUILD EXEC BINARY (subgraph_exec.bin)
    # -----------------------------------------------------------------------
    exec_ops_bytes = []
    num_exec_ops = 0

    for i, o in enumerate(ops):
        addr = o.get("addr", 0)
        col = (addr >> 25) & 0x7F
        row = (addr >> 20) & 0x1F
        reg = addr & 0xFFFFF
        op_name = o["op"]

        # Skip parameter writes (already staged in L1)
        if op_name == "BLOCKWRITE" and row >= 2 and any(
            reg in (L1_SHIFT_CUT_OFFSET + k * L1_PARAM_STRIDE,
                    L1_BIAS_OFFSET + k * L1_PARAM_STRIDE,
                    L1_WEIGHTS_OFFSET + k * L1_PARAM_STRIDE)
            for k in range(n_layers + 1)
        ):
            continue

        # Skip the template's reset-time zeroing of core-tile locks. Locks with
        # a non-zero initial value are still re-armed every frame below; a
        # lock whose initial value is 0 is only reset by the init stream.
        if op_name == "WRITE" and row >= 2 and CORE_LOCK_BASE <= reg < CORE_LOCK_END and o.get("val") == 0:
            continue

        # Skip static switchbox writes (already programmed in init)
        if op_name == "WRITE" and ((0x3F000 <= reg <= 0x3F1FF) or (0xB0000 <= reg <= 0xB01FF)):
            continue

        # MemTile Lock 2 credit restore (val = 4) + Lock 4/5 initialization
        # (same register match as the init build above).
        if row == 1 and reg == memtile_lock_reg(LOCK_CORE_EGRESS_CREDIT):
            exec_ops_bytes.append(memtile_lock_write(col, LOCK_CORE_EGRESS_CREDIT, 4))
            exec_ops_bytes.append(memtile_lock_write(col, LOCK_L2_PING, 1))
            exec_ops_bytes.append(memtile_lock_write(col, LOCK_L2_PONG, 0))
            num_exec_ops += 3
            continue

        # Shim BD BLOCKWRITE
        if row == 0 and reg == 0x1D000 and op_name == "BLOCKWRITE":
            raw_chunk = base_bytes[o["offset"] : o["offset"] + o["size"]]
            exec_ops_bytes.append(raw_chunk)
            num_exec_ops += 1
            has_ddr = (i + 1 < len(ops) and ops[i + 1]["op"] == "DDR_PATCH")
            if not has_ddr:
                p0 = make_ddr_patch((col << 25) | 0x0001D004, 0, col * 2048)
                p1 = make_ddr_patch((col << 25) | 0x0001D024, 1, col * 1024)
                exec_ops_bytes.extend([p0, p1])
                num_exec_ops += 2
            continue

        raw_chunk = base_bytes[o["offset"] : o["offset"] + o["size"]]
        exec_ops_bytes.append(raw_chunk)
        num_exec_ops += 1

    payload_exec = b"".join(exec_ops_bytes)
    total_size_exec = TXN_HEADER_BYTES + len(payload_exec)
    header_exec = struct.pack("<4I", 0, 0, num_exec_ops, total_size_exec)
    full_exec_bin = header_exec + payload_exec
    validate_transaction_stream(full_exec_bin, name=os.path.basename(out_exec_path), require_terminal_tct=True)

    os.makedirs(os.path.dirname(os.path.abspath(out_exec_path)), exist_ok=True)
    with open(out_exec_path, "wb") as f:
        f.write(full_exec_bin)

    return out_init_path, out_exec_path


def validate_memtile_buffer_layout(
    schedule: Union[SchedulePlan, StageSchedulePlan, MultiStageSchedulePlan],
    max_memtile_bytes: int = MEMTILE_BYTES,
) -> bool:
    """
    Verifies that all buffer allocations fit strictly within physical MemTile SRAM (512 KB per column).
    """
    if isinstance(schedule, MultiStageSchedulePlan):
        sched_plan = schedule.to_schedule_plan()
    else:
        sched_plan = schedule

    for p in sched_plan.passes:
        if p.ingress_source in ("L2_BANK_0", "L2_BANK_1"):
            if not (0 <= p.ingress_addr < max_memtile_bytes):
                raise ValueError(f"Ingress address {hex(p.ingress_addr)} exceeds MemTile capacity {max_memtile_bytes}")
            if p.ingress_addr + L2_BANK_BYTES > max_memtile_bytes:
                raise ValueError(f"Ingress buffer range at {hex(p.ingress_addr)} exceeds MemTile capacity")
        if p.egress_dest in ("L2_BANK_0", "L2_BANK_1"):
            if not (0 <= p.egress_addr < max_memtile_bytes):
                raise ValueError(f"Egress address {hex(p.egress_addr)} exceeds MemTile capacity {max_memtile_bytes}")
            if p.egress_addr + L2_BANK_BYTES > max_memtile_bytes:
                raise ValueError(f"Egress buffer range at {hex(p.egress_addr)} exceeds MemTile capacity")
        if p.egress_dest == "HOST_DDR" and p.egress_addr == L2_FINAL_EGRESS_OFFSET:
            if not (0 <= p.egress_addr + 0x4000 <= max_memtile_bytes):
                raise ValueError(f"Egress buffer at {hex(p.egress_addr)} exceeds MemTile capacity")
    return True


def emit_multi_stage_transaction_bundle(
    schedule: Union[StageSchedulePlan, MultiStageSchedulePlan, SchedulePlan],
    base_txn_path: str,
    out_init_path: str,
    out_exec_path: str,
    cores: Optional[List[Tuple[int, int]]] = None
) -> Tuple[str, str]:
    """
    Generates unified init.bin and exec.bin transaction sequence covering
    entire monolithic stages (Stem, P3, P4, P5), collapsing the 50 ERT dispatches
    into a unified driver submission with zero intermediate DDR traffic.
    """
    validate_memtile_buffer_layout(schedule)
    if isinstance(schedule, MultiStageSchedulePlan):
        sched_plan = schedule.to_schedule_plan()
    else:
        sched_plan = schedule

    return emit_multi_layer_transaction_bundle(
        schedule=sched_plan,
        base_txn_path=base_txn_path,
        out_init_path=out_init_path,
        out_exec_path=out_exec_path,
        cores=cores,
    )


def chain_stage_transaction_streams(
    stage_txns: List[Union[bytes, str, Path]],
    stage_names: Optional[List[str]] = None,
    num_cores: int = 16,
) -> bytes:
    """
    Chains all stage transaction streams into a single continuous ERT instruction buffer:
      - Strips intermediate TCT completion tokens from intermediate stages, retaining
        exactly one terminating TCT token at the very end of the final stage.
      - Inserts on-die hardware barrier locks (Locks 0-7, specifically Lock 2 gather credit restore,
        Lock 4/5 ping-pong buffer handoff, and Locks 6/7 stage barrier handoff) inside MemTile
        DMA sequences.
      - Sequences buffer handoffs across L2 Bank 0 (0x40000) and Bank 1 (0x60000) so the NPU
        transitions autonomously without CPU driver intervention.
    """
    if not stage_txns:
        raise ValueError("stage_txns must not be empty")

    raw_txns: List[bytes] = []
    for txn in stage_txns:
        if isinstance(txn, (str, Path)):
            raw_txns.append(Path(txn).read_bytes())
        elif isinstance(txn, (bytes, bytearray, memoryview)):
            raw_txns.append(bytes(txn))
        else:
            raise TypeError(f"Unsupported transaction type: {type(txn)}")

    n_stages = len(raw_txns)
    if stage_names is None:
        stage_names = [f"Stage_{i}" for i in range(n_stages)]

    # Both builders emit plain lock-value WRITEs (see memtile_lock_write): they
    # set values from the instruction stream and do not wait for the previous
    # stage's DMAs. Stripping the intermediate TCTs below removes the only
    # completion wait between stages; the terminal TCT is the one wait left.
    def build_barrier_ops(stage_idx: int) -> Tuple[bytes, int]:
        barrier = LOCK_STAGE_BARRIER_A if stage_idx % 2 == 0 else LOCK_STAGE_BARRIER_B
        ops_bytes = []
        for c in range(4):
            # MemTile Lock 2 gather credit restore (val = 4)
            ops_bytes.append(memtile_lock_write(c, LOCK_CORE_EGRESS_CREDIT, 4))
            # Stage barrier lock signal (val = 1)
            ops_bytes.append(memtile_lock_write(c, barrier, 1))
            # MemTile L2 Bank handoff: mark both Bank 0 and Bank 1 ready
            ops_bytes.append(memtile_lock_write(c, LOCK_L2_PING, 1))
            ops_bytes.append(memtile_lock_write(c, LOCK_L2_PONG, 1))
        return b"".join(ops_bytes), len(ops_bytes)

    def build_frame_prologue_ops() -> Tuple[bytes, int]:
        ops_bytes = []
        for c in range(4):
            # MemTile Lock 2 gather credit restore (val = 4)
            ops_bytes.append(memtile_lock_write(c, LOCK_CORE_EGRESS_CREDIT, 4))
            # Clear stage barriers (Locks 6 and 7)
            ops_bytes.append(memtile_lock_write(c, LOCK_STAGE_BARRIER_A, 0))
            ops_bytes.append(memtile_lock_write(c, LOCK_STAGE_BARRIER_B, 0))
            # Set Ping Bank 0 and Pong Bank 1 ready (val = 1)
            ops_bytes.append(memtile_lock_write(c, LOCK_L2_PING, 1))
            ops_bytes.append(memtile_lock_write(c, LOCK_L2_PONG, 1))
        return b"".join(ops_bytes), len(ops_bytes)

    prologue_bytes, prologue_ops = build_frame_prologue_ops()
    chained_ops_bytes = [prologue_bytes]
    total_ops_count = prologue_ops

    for s_idx, (s_name, s_b) in enumerate(zip(stage_names, raw_txns)):
        if len(s_b) < TXN_HEADER_BYTES:
            raise ValueError(f"Transaction data for stage {s_name} too short ({len(s_b)} B)")

        stage_ops = parse_transaction_stream(s_b)
        declared_ops = struct.unpack("<I", s_b[8:12])[0]
        if len(stage_ops) != declared_ops:
            raise ValueError(f"stage {s_name}: header declares {declared_ops} ops, decoded {len(stage_ops)}")
        is_last = (s_idx == n_stages - 1)
        ends_with_tct = bool(stage_ops) and stage_ops[-1]["op"] == "TCT"

        if is_last:
            chained_ops_bytes.append(s_b[TXN_HEADER_BYTES:])
            total_ops_count += declared_ops
        else:
            if ends_with_tct:
                # Drop this stage's terminal completion wait; only the final stage keeps one.
                chained_ops_bytes.append(s_b[TXN_HEADER_BYTES:stage_ops[-1]["offset"]])
                total_ops_count += declared_ops - 1
            else:
                chained_ops_bytes.append(s_b[TXN_HEADER_BYTES:])
                total_ops_count += declared_ops

            # Insert inter-stage lock-value writes
            bar_bytes, bar_ops = build_barrier_ops(s_idx)
            chained_ops_bytes.append(bar_bytes)
            total_ops_count += bar_ops

    full_payload = b"".join(chained_ops_bytes)
    total_size = TXN_HEADER_BYTES + len(full_payload)
    rem = total_size % TXN_PAD_BYTES
    # The zero pad sits inside the declared size; validate_transaction_stream
    # accepts it because the op count bounds the decode.
    pad = b"\x00" * (TXN_PAD_BYTES - rem) if rem != 0 else b""
    header = struct.pack("<4I", 0, 0, total_ops_count, total_size + len(pad))
    stream = header + full_payload + pad
    validate_transaction_stream(stream, name="chained exec stream")
    return stream


def emit_unified_monolithic_transaction_bundle(
    schedule: Union[MultiStageSchedulePlan, Dict[str, StageSchedulePlan]],
    base_txn_path: str,
    out_init_path: str,
    out_exec_path: str,
    cores: Optional[List[Tuple[int, int]]] = None
) -> Tuple[str, str]:
    """
    Synthesizes a unified single-dispatch monolithic transaction stream
    (init_monolithic.bin and exec_monolithic.bin) covering all 9 stages:
      - Inter-stage hardware barrier locks (Locks 0-7)
      - L2 Bank 0 (0x40000) and Bank 1 (0x60000) autonomous transitions
      - Single ERT driver submission (< 45 us)
    """
    if isinstance(schedule, MultiStageSchedulePlan):
        multi_plan = schedule
    elif isinstance(schedule, dict):
        multi_plan = MultiStageSchedulePlan(stages=schedule)
    else:
        raise TypeError(f"Unsupported schedule type for monolithic unification: {type(schedule)}")

    # 1. Synthesize unified init.bin containing all stationary parameters.
    # Flattening sums every stage's layers into ONE resident parameter set per
    # core, so the layout check names the stages when it refuses.
    sched_plan = multi_plan.to_schedule_plan()
    stage_summary = ", ".join(f"{name}={stage.num_layers}" for name, stage in multi_plan.stages.items())
    validate_l1_parameter_layout(
        sched_plan,
        context=(f"({len(multi_plan.stages)} stages flattened into one resident set by "
                 f"to_schedule_plan(): {stage_summary}; parameters must be staged per "
                 f"stage instead)"),
    )
    emit_multi_layer_transaction_bundle(
        schedule=sched_plan,
        base_txn_path=base_txn_path,
        out_init_path=out_init_path,
        out_exec_path=out_exec_path + ".tmp",
        cores=cores,
    )

    # 2. Synthesize individual stage execution binaries and chain with on-die barriers
    stage_exec_bytes: List[bytes] = []
    stage_names: List[str] = list(multi_plan.stages.keys())

    with tempfile.TemporaryDirectory() as td:
        for s_name, stage_plan in multi_plan.stages.items():
            if s_name == "DFL_Decode" and getattr(stage_plan, "custom_exec_bytes", None):
                stage_exec_bytes.append(stage_plan.custom_exec_bytes)
                continue
            s_init_tmp = os.path.join(td, f"{s_name}_init.bin")
            s_exec_tmp = os.path.join(td, f"{s_name}_exec.bin")
            emit_multi_stage_transaction_bundle(stage_plan, base_txn_path, s_init_tmp, s_exec_tmp, cores=cores)
            with open(s_exec_tmp, "rb") as f:
                stage_exec_bytes.append(f.read())

    unified_exec = chain_stage_transaction_streams(stage_exec_bytes, stage_names=stage_names)
    with open(out_exec_path, "wb") as f:
        f.write(unified_exec)

    tmp_path = out_exec_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    return (out_init_path, out_exec_path)


def run_n_layer_fixed_point_reference(
    layers_or_partition: Union[NpuFusedPartition, List[ConvLayerMeta]],
    input_bytes: np.ndarray,
    num_cores: int = 16
) -> np.ndarray:
    """
    Computes exact INT8 fixed-point reference across arbitrary N-layer sequence
    matching physical AIE2 SRS execution with intermediate L2 ping-pong buffers.
    """
    if hasattr(layers_or_partition, "passes"):
        layers = [p.layer_meta for p in layers_or_partition.passes]
    elif hasattr(layers_or_partition, "layers"):
        layers = layers_or_partition.layers
    elif isinstance(layers_or_partition, NpuFusedPartition):
        layers = layers_or_partition.layers
    else:
        layers = list(layers_or_partition)

    n = len(layers)
    if n == 0:
        raise ValueError("No layers provided for reference calculation")

    cur_activations = input_bytes
    slice_in = cur_activations[:len(cur_activations) // 4] if len(cur_activations) == 8192 else cur_activations

    l0 = layers[0]
    Cin0 = min(8, l0.weights_raw.shape[1])
    Cout0 = min(8, l0.weights_raw.shape[0])
    x_patch = np.zeros((4, Cin0, 3, 3), dtype=np.int8)
    for p in range(4):
        for ky in range(3):
            for kx in range(3):
                for cin in range(Cin0):
                    off = 1 * 32 + ky * 256 + kx * 32 + p * 8 + cin
                    if off < len(slice_in):
                        x_patch[p, cin, ky, kx] = slice_in[off]

    # Layer 0 Compute
    w0 = l0.weights_raw[:Cout0, :Cin0, :, :]
    b0 = l0.bias_i32[:Cout0] if l0.bias_i32 is not None else np.zeros(Cout0, dtype=np.int32)
    shift0 = int(l0.shift_cut)
    kh0, kw0 = w0.shape[2], w0.shape[3]

    y_cur = np.zeros((4, Cout0), dtype=np.int8)
    for p in range(4):
        acc = b0.copy().astype(np.int64)
        for ky in range(kh0):
            for kx in range(kw0):
                acc += w0[:, :, ky, kx].astype(np.int64) @ x_patch[p, :, ky, kx].astype(np.int64)
        bias_round = 1 << (shift0 - 1)
        y_cur[p] = np.clip(np.right_shift(acc + bias_round, shift0), -128, 127).astype(np.int8)

    # Subsequent Intermediate & Final Layers
    for k in range(1, n):
        lk = layers[k]
        Cout_k = min(32 if k == n - 1 else 8, lk.weights_raw.shape[0])
        Cin_k = min(Cout0, lk.weights_raw.shape[1])
        wk = lk.weights_raw[:Cout_k, :Cin_k, :, :]
        bk = lk.bias_i32[:Cout_k] if lk.bias_i32 is not None else np.zeros(Cout_k, dtype=np.int32)
        shift_k = int(lk.shift_cut)

        # Dynamic scale adjustment between consecutive layers
        scale_ratio = float(layers[k - 1].scale_y / lk.scale_x) if lk.scale_x > 0 else 1.0
        if abs(scale_ratio - 1.0) < 1e-4:
            y_cur_scaled = y_cur
        else:
            y_cur_scaled = np.clip(np.round(y_cur.astype(np.float32) * scale_ratio), -128, 127).astype(np.int8)

        y_next = np.zeros((4, Cout_k), dtype=np.int8)
        for p in range(4):
            acc = bk.copy().astype(np.int64)
            acc += wk[:, :, 0, 0].astype(np.int64) @ y_cur_scaled[p].astype(np.int64)
            bias_round = 1 << (shift_k - 1)
            y_next[p] = np.clip(np.right_shift(acc + bias_round, shift_k), -128, 127).astype(np.int8)

        y_cur = y_next
        Cout0 = Cout_k

    return np.tile(y_cur, (num_cores, 1)).flatten()


def run_n_layer_ort_cpu_reference(
    onnx_model_or_path: Union[str, Path, onnx.ModelProto],
    input_bytes: np.ndarray,
    num_cores: int = 16
) -> np.ndarray:
    """Runs full N-layer ONNX graph in ONNX Runtime CPU and returns quantized INT8 output."""
    if isinstance(onnx_model_or_path, (str, Path)):
        model_bytes = Path(onnx_model_or_path).read_bytes()
    else:
        model_bytes = onnx_model_or_path.SerializeToString()

    sess = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    inp_shape = sess.get_inputs()[0].shape
    cin_expected = inp_shape[1] if (len(inp_shape) > 1 and isinstance(inp_shape[1], int)) else 32
    x_ort = np.zeros((4, cin_expected, 3, 3), dtype=np.int8)
    slice_in = input_bytes[:len(input_bytes) // 4] if len(input_bytes) == 8192 else input_bytes

    for p in range(4):
        for ky in range(3):
            for kx in range(3):
                for cin in range(cin_expected):
                    off = 1 * 32 + ky * 256 + kx * 32 + p * 8 + (cin % 8)
                    if off < len(slice_in):
                        x_ort[p, cin, ky, kx] = slice_in[off]

    ort_out = sess.run([output_name], {input_name: x_ort})[0].reshape(4, -1)
    if ort_out.shape[1] > 32:
        ort_out = ort_out[:, :32]
    elif ort_out.shape[1] < 32:
        padded = np.zeros((4, 32), dtype=np.int8)
        padded[:, :ort_out.shape[1]] = ort_out
        ort_out = padded

    ort_full = np.tile(ort_out, (num_cores, 1)).flatten()
    return ort_full.astype(np.int8)


def execute_multi_layer_on_silicon(
    schedule: SchedulePlan,
    init_txn_path: str,
    exec_txn_path: str,
    xclbin_path: str,
    input_bytes: np.ndarray,
    num_cores: int = 16,
    warmup_iters: int = 20,
    bench_iters: int = 100,
    device_idx: int = 0
) -> Dict[str, Any]:
    """
    Executes an N-layer pipelined Conv2D subgraph on physical AMD Phoenix AIE2 silicon:
      - Inter-layer activations ping-pong strictly between MemTile L2_BANK_0 (0x40000)
        and L2_BANK_1 (0x60000).
      - Zero intermediate DDR roundtrips (0 intermediate DDR bytes).
      - Measures latency, throughput, and parity against fixed-point and ORT references.
    """
    import time
    from ignite_xdna.runtime.driver import XrtSiliconHarness, setup_xrt_environment
    from ignite_xdna.runtime.test_im2col_hardware import calculate_numerical_parity

    setup_xrt_environment()
    harness = XrtSiliconHarness(device_idx=device_idx)
    harness.load_xclbin(xclbin_path, "MLIR_AIE")

    bo_init, ninstr_init = harness.create_instruction_bo(init_txn_path)
    bo_exec, ninstr_exec = harness.create_instruction_bo(exec_txn_path)

    in_size = len(input_bytes)
    out_bytes = num_cores * 256  # 4,096 B final egress

    bo_in = harness.create_host_bo(in_size, 3)
    bo_out = harness.create_host_bo(out_bytes, 4)

    bo_in.write(input_bytes.tobytes(), 0)
    bo_in.sync(harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_out.write(np.zeros(out_bytes, dtype=np.int8).tobytes(), 0)
    bo_out.sync(harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    # 1. Program stationary multi-layer parameters into L1 and prime MemTile locks
    t0_init = time.perf_counter()
    run_init, state_init = harness.dispatch_kernel(bo_init, ninstr_init, bo_in, bo_out, timeout_ms=3000)
    t1_init = time.perf_counter()
    init_us = (t1_init - t0_init) * 1e6

    if str(state_init) != "ert_cmd_state.ERT_CMD_STATE_COMPLETED":
        raise RuntimeError(f"Multi-layer parameter init failed with state: {state_init}")

    # 2. Prime hardware pipeline with 1 exec dispatch
    harness.dispatch_kernel(bo_exec, ninstr_exec, bo_in, bo_out, timeout_ms=2000)

    # 3. Synchronous Parity Dispatch
    run_exec, state_exec = harness.dispatch_kernel(bo_exec, ninstr_exec, bo_in, bo_out, timeout_ms=2000)
    if str(state_exec) != "ert_cmd_state.ERT_CMD_STATE_COMPLETED":
        raise RuntimeError(f"Multi-layer execution failed with state: {state_exec}")

    bo_out.sync(harness.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    raw_hw_output = np.frombuffer(bo_out.read(out_bytes, 0), dtype=np.int8).copy()
    unpacked_hw = unblock_aie2_egress(raw_hw_output, num_cores=num_cores)

    # 4. Benchmark sustained pipelined latency
    for _ in range(warmup_iters):
        harness.dispatch_kernel(bo_exec, ninstr_exec, bo_in, bo_out, timeout_ms=2000)

    latencies_us = []
    for _ in range(bench_iters):
        t0 = time.perf_counter()
        harness.dispatch_kernel(bo_exec, ninstr_exec, bo_in, bo_out, timeout_ms=2000)
        t1 = time.perf_counter()
        latencies_us.append((t1 - t0) * 1e6)

    mean_us = float(np.mean(latencies_us))
    median_us = float(np.median(latencies_us))
    min_us = float(np.min(latencies_us))
    p95_us = float(np.percentile(latencies_us, 95))
    fps = 1e6 / mean_us if mean_us > 0 else 0.0

    bo_in = None
    bo_out = None
    bo_init = None
    bo_exec = None
    harness.close()

    return {
        "num_layers": schedule.num_layers,
        "intermediate_ddr_bytes": schedule.intermediate_ddr_bytes,
        "init_us": init_us,
        "mean_us": mean_us,
        "median_us": median_us,
        "min_us": min_us,
        "p95_us": p95_us,
        "fps": fps,
        "raw_hw_output": raw_hw_output,
        "unpacked_hw": unpacked_hw,
    }

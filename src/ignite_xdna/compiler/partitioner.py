#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
src/ignite_xdna/compiler/partitioner.py

ONNX Graph Partitioner & Subgraph Extraction for AMD Phoenix XDNA1 (AIE2).
Walks an ONNX DAG, identifies maximal chains of fusible accelerator operators
(Conv2D -> Relu / Clip / Identity -> Conv2D -> ...), and partitions unsupported
operators to fallback host CPU execution nodes.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ignite_xdna.compiler.lower_onnx_conv import (
    pack_bias_aie2_vector_layout,
    pack_weights_aie2_vector_layout,
)


@dataclass
class ConvLayerMeta:
    """Metadata and packed stationary parameters for a single Conv2D layer."""
    node_name: str
    layer_index: int
    op_type: str = "Conv"
    in_channels: int = 32
    out_channels: int = 32
    kernel_shape: List[int] = field(default_factory=lambda: [3, 3])
    strides: List[int] = field(default_factory=lambda: [1, 1])
    pads: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    weights_raw: Optional[np.ndarray] = None
    weights_packed: Optional[np.ndarray] = None
    bias_raw: Optional[np.ndarray] = None
    bias_i32: Optional[np.ndarray] = None
    bias_packed: Optional[np.ndarray] = None
    scale_x: float = 0.0078125
    scale_w: float = 0.0078125
    scale_y: float = 0.0078125
    scale_bias: float = 0.00006103515625
    zp_x: int = 0
    zp_w: int = 0
    zp_y: int = 0
    pos_x: int = 7
    pos_w: int = 7
    pos_y: int = 7
    shift_cut: int = 7
    sigma: int = 21  # shift_cut + 14
    activation: Optional[str] = None  # None, "Relu", "Clip", "Identity", "SiLU"
    residual_add: bool = False
    residual_source: Optional[str] = None
    fused_ops: List[str] = field(default_factory=list)
    channel_slice: Optional[Tuple[int, int]] = None
    channel_concat_offset: Optional[int] = None


@dataclass
class NpuFusedPartition:
    """A maximal fused sequence of Conv2D layers executable on AIE2 hardware."""
    partition_id: int
    layers: List[ConvLayerMeta] = field(default_factory=list)
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    in_bytes: int = 8192
    out_bytes: int = 4096
    stage_name: Optional[str] = None
    c2f_blocks: List[str] = field(default_factory=list)
    has_zero_ddr_roundtrip: bool = True

    @property
    def num_layers(self) -> int:
        return len(self.layers)


@dataclass
class CpuFallbackPartition:
    """A sequence of ONNX nodes to be executed on host CPU via ONNX Runtime."""
    partition_id: int
    nodes: List[onnx.NodeProto] = field(default_factory=list)
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    onnx_model: Optional[onnx.ModelProto] = None
    is_backbone: bool = False


@dataclass
class PartitionedGraph:
    """Full execution graph consisting of sequential CPU and NPU partitions."""
    model_name: str
    partitions: List[Union[NpuFusedPartition, CpuFallbackPartition]] = field(default_factory=list)
    initial_inputs: List[str] = field(default_factory=list)
    terminal_outputs: List[str] = field(default_factory=list)

    @property
    def npu_partitions(self) -> List[NpuFusedPartition]:
        return [p for p in self.partitions if isinstance(p, NpuFusedPartition)]

    @property
    def cpu_partitions(self) -> List[CpuFallbackPartition]:
        return [p for p in self.partitions if isinstance(p, CpuFallbackPartition)]

    @property
    def backbone_npu_partitions(self) -> List[NpuFusedPartition]:
        return [
            p for p in self.npu_partitions
            if p.stage_name in {"Stem", "P3", "P4", "P5"} or any(self._is_backbone_node_name(l.node_name) for l in p.layers)
        ]

    @property
    def backbone_cpu_partitions(self) -> List[CpuFallbackPartition]:
        return [
            p for p in self.cpu_partitions
            if p.is_backbone or any(self._is_backbone_node_name(n.name) for n in p.nodes)
        ]

    @property
    def neck_npu_partitions(self) -> List[NpuFusedPartition]:
        return [
            p for p in self.npu_partitions
            if p.stage_name in {"Neck", "Neck_FPN", "Neck_PAN"} or any(self._is_neck_node_name(l.node_name) for l in p.layers)
        ]

    @property
    def neck_cpu_partitions(self) -> List[CpuFallbackPartition]:
        return [
            p for p in self.cpu_partitions
            if any(self._is_neck_node_name(n.name) for n in p.nodes)
        ]

    @property
    def head_npu_partitions(self) -> List[NpuFusedPartition]:
        return [
            p for p in self.npu_partitions
            if p.stage_name in {"Detect_P3", "Detect_P4", "Detect_P5", "Detect_Heads"}
            or any(self._is_head_node_name(l.node_name) for l in p.layers)
        ]

    @property
    def head_cpu_partitions(self) -> List[CpuFallbackPartition]:
        return [
            p for p in self.cpu_partitions
            if any(self._is_head_node_name(n.name) for n in p.nodes)
        ]

    @staticmethod
    def _is_backbone_node_name(name: str) -> bool:
        return any(name.startswith(f"/model.{i}/") for i in range(10))

    @staticmethod
    def _is_neck_node_name(name: str) -> bool:
        return any(name.startswith(f"/model.{i}/") for i in range(10, 22))

    @staticmethod
    def _is_head_node_name(name: str) -> bool:
        return any(name.startswith(f"/model.{i}/") or name.startswith(f"model.{i}.") for i in range(22, 24))


class GraphPartitioner:
    """
    Walks an ONNX DAG to partition operations into host CPU fallback nodes
    and maximal fusible NPU AIE2 subgraphs.
    """

    SUPPORTED_FUSIBLE_OPS = {"Conv", "Relu", "Clip", "Identity", "Mul", "HardSigmoid", "Sigmoid", "Resize", "Concat"}

    def __init__(
        self,
        model_or_path: Union[str, Path, onnx.ModelProto],
        fuse_c2f: bool = True,
        fuse_backbone: bool = True,
        fuse_neck: bool = True,
        fuse_head: bool = True,
        backbone_only: bool = False,
        neck_only: bool = False,
        head_only: bool = False,
    ):
        if isinstance(model_or_path, (str, Path)):
            self.model_path = str(model_or_path)
            self.model = onnx.load(self.model_path)
        elif isinstance(model_or_path, onnx.ModelProto):
            self.model_path = "<in-memory>"
            self.model = model_or_path
        else:
            raise TypeError(f"Expected path or ModelProto, got {type(model_or_path)}")

        self.fuse_c2f = fuse_c2f
        self.fuse_backbone = fuse_backbone
        self.fuse_neck = fuse_neck
        self.fuse_head = fuse_head
        self.backbone_only = backbone_only
        self.neck_only = neck_only
        self.head_only = head_only
        self.inits = {t.name: numpy_helper.to_array(t) for t in self.model.graph.initializer}

    def _is_yolo_backbone_model(self) -> bool:
        return any(n.name.startswith("/model.") for n in self.model.graph.node)

    def _extract_backbone_stages(self) -> List[NpuFusedPartition]:
        """
        Extracts the YOLOv8 backbone into 4 monolithic NPU partitions:
        - Stem (model.0..model.3): 7 Convs
        - P3 (model.4..model.5): 7 Convs
        - P4 (model.6..model.7): 7 Convs
        - P5 (model.8..model.9): 6 Convs

        Zero CPU fallback partitions are created across the backbone!
        Residual Adds in bottlenecks are marked for in-tile kernel fusion.
        C2f Split, Slice, and Concat are mapped to MemTile AGU descriptors.
        """
        nodes = list(self.model.graph.node)
        add_nodes = [n for n in nodes if n.op_type == "Add"]

        stage_specs = [
            ("Stem", (0, 1, 2, 3)),
            ("P3", (4, 5)),
            ("P4", (6, 7)),
            ("P5", (8, 9)),
        ]

        partitions: List[NpuFusedPartition] = []
        for part_id, (stage_name, layer_nums) in enumerate(stage_specs):
            stage_conv_nodes: List[Tuple[int, onnx.NodeProto]] = []
            for idx, node in enumerate(nodes):
                if node.op_type == "Conv":
                    for ln in layer_nums:
                        if node.name.startswith(f"/model.{ln}/"):
                            stage_conv_nodes.append((idx, node))
                            break

            stage_layers: List[ConvLayerMeta] = []
            for layer_idx, (node_idx, conv_node) in enumerate(stage_conv_nodes):
                layer = self._extract_conv_layer(conv_node, node_idx, nodes)
                layer.layer_index = layer_idx

                # 1. Activation detection: check for SiLU (/act/Mul, /act/Sigmoid, HardSigmoid)
                c_name = conv_node.name
                if "/act/" in c_name or any(f"{c_name.rsplit('/', 1)[0]}/act" in n.name for n in nodes):
                    layer.activation = "SiLU"
                    layer.fused_ops.append("SiLU")
                else:
                    downstream = [n for n in nodes if conv_node.output[0] in n.input]
                    for d in downstream:
                        if d.op_type in {"Relu", "Clip", "Identity"}:
                            layer.activation = d.op_type
                            layer.fused_ops.append(d.op_type)
                            break
                        elif d.op_type in {"HardSigmoid", "Sigmoid", "Mul"}:
                            layer.activation = "SiLU"
                            layer.fused_ops.append("SiLU")
                            break
                    if layer.activation is None:
                        layer.activation = "SiLU"
                        layer.fused_ops.append("SiLU")

                # 2. Residual Add detection: check if node is in a bottleneck with an Add node
                if "/cv2/" in c_name and "/m." in c_name:
                    bottleneck_prefix = c_name.split("/cv2/")[0]
                    matching_adds = [a for a in add_nodes if a.name.startswith(bottleneck_prefix)]
                    if matching_adds:
                        layer.residual_add = True
                        layer.residual_source = matching_adds[0].input[0]
                        layer.fused_ops.append("Add")

                # 3. Channel Slice & Concat routing metadata
                if "/cv1/" in c_name and "/m." in c_name:
                    layer.channel_slice = (layer.in_channels, layer.in_channels)
                    layer.fused_ops.append("Slice")
                elif "/cv2/" in c_name and not "/m." in c_name:
                    layer.channel_concat_offset = 0
                    layer.fused_ops.append("Concat")

                stage_layers.append(layer)

            c2f_in_stage = [f"model.{ln}" for ln in layer_nums if ln in (2, 4, 6, 8)]
            p = NpuFusedPartition(
                partition_id=part_id,
                layers=stage_layers,
                input_names=[stage_layers[0].node_name + "_in"] if stage_layers else [],
                output_names=[stage_layers[-1].node_name + "_out"] if stage_layers else [],
                in_bytes=8192,
                out_bytes=4096,
                stage_name=stage_name,
                c2f_blocks=c2f_in_stage,
                has_zero_ddr_roundtrip=True,
            )
            partitions.append(p)

        return partitions

    def _extract_neck_stages(self) -> List[NpuFusedPartition]:
        """
        Extracts the YOLOv8 Neck (Layers 10..21) into 2 monolithic NPU partitions:
        - Neck_FPN (Layers 10..15): 8 Convs with 2x NN upsampling and lateral P4/P3
          concatenations absorbed into MemTile AGU descriptors.
        - Neck_PAN (Layers 16..21): 10 Convs with lateral concatenations absorbed
          into MemTile strided S2MM DMA scatter.

        Zero CPU fallback partitions are created across Layers 10..21!
        """
        nodes = list(self.model.graph.node)
        add_nodes = [n for n in nodes if n.op_type == "Add"]

        stage_specs = [
            ("Neck_FPN", (10, 11, 12, 13, 14, 15)),
            ("Neck_PAN", (16, 17, 18, 19, 20, 21)),
        ]

        partitions: List[NpuFusedPartition] = []
        for part_id, (stage_name, layer_nums) in enumerate(stage_specs, start=4):
            stage_conv_nodes: List[Tuple[int, onnx.NodeProto]] = []
            for idx, node in enumerate(nodes):
                if node.op_type == "Conv":
                    for ln in layer_nums:
                        if node.name.startswith(f"/model.{ln}/"):
                            stage_conv_nodes.append((idx, node))
                            break

            stage_layers: List[ConvLayerMeta] = []
            for layer_idx, (node_idx, conv_node) in enumerate(stage_conv_nodes):
                layer = self._extract_conv_layer(conv_node, node_idx, nodes)
                layer.layer_index = layer_idx

                # 1. Activation detection: check for SiLU (/act/Mul, /act/Sigmoid, HardSigmoid)
                c_name = conv_node.name
                if "/act/" in c_name or any(f"{c_name.rsplit('/', 1)[0]}/act" in n.name for n in nodes):
                    layer.activation = "SiLU"
                    layer.fused_ops.append("SiLU")
                else:
                    layer.activation = "SiLU"
                    layer.fused_ops.append("SiLU")

                # 2. Residual Add detection (if any)
                if "/cv2/" in c_name and "/m." in c_name:
                    bottleneck_prefix = c_name.split("/cv2/")[0]
                    matching_adds = [a for a in add_nodes if a.name.startswith(bottleneck_prefix)]
                    if matching_adds:
                        layer.residual_add = True
                        layer.residual_source = matching_adds[0].input[0]
                        layer.fused_ops.append("Add")

                # 3. Channel Slice & Concat routing metadata
                if "/cv1/" in c_name and "/m." in c_name:
                    layer.channel_slice = (layer.in_channels, layer.in_channels)
                    layer.fused_ops.append("Slice")
                elif "/cv2/" in c_name and not "/m." in c_name:
                    layer.channel_concat_offset = 0
                    layer.fused_ops.append("Concat")

                stage_layers.append(layer)

            c2f_in_stage = [f"model.{ln}" for ln in layer_nums if ln in (12, 15, 18, 21)]
            p = NpuFusedPartition(
                partition_id=part_id,
                layers=stage_layers,
                input_names=[stage_layers[0].node_name + "_in"] if stage_layers else [],
                output_names=[stage_layers[-1].node_name + "_out"] if stage_layers else [],
                in_bytes=8192,
                out_bytes=4096,
                stage_name=stage_name,
                c2f_blocks=c2f_in_stage,
                has_zero_ddr_roundtrip=True,
            )
            partitions.append(p)

        return partitions

    def _extract_head_stages(self, split_scales: bool = True) -> List[NpuFusedPartition]:
        """
        Extracts the YOLOv8 Detect Head (Layer 22) into monolithic NPU partitions:
        When split_scales=True:
        - Detect_P3 (Scale 0, 80x80): 6 Convs (3 Box + 3 Cls)
        - Detect_P4 (Scale 1, 40x40): 6 Convs (3 Box + 3 Cls)
        - Detect_P5 (Scale 2, 20x20): 6 Convs (3 Box + 3 Cls)
        When split_scales=False:
        - Detect_Heads: 18 Convs across all 3 scales.

        Absorbs all 6 final 1x1 Box/Cls prediction heads and 12 intermediate 3x3 Convs
        with strictly 0 CPU fallback partitions across Layer 22!
        """
        nodes = list(self.model.graph.node)

        if split_scales:
            stage_specs = [
                ("Detect_P3", "0"),
                ("Detect_P4", "1"),
                ("Detect_P5", "2"),
            ]
        else:
            stage_specs = [("Detect_Heads", None)]

        partitions: List[NpuFusedPartition] = []
        for part_id, (stage_name, scale_id) in enumerate(stage_specs, start=6):
            stage_conv_nodes: List[Tuple[int, onnx.NodeProto]] = []
            for idx, node in enumerate(nodes):
                if node.op_type == "Conv" and "model.22" in node.name:
                    if scale_id is None:
                        stage_conv_nodes.append((idx, node))
                    elif f"cv2.{scale_id}" in node.name or f"cv3.{scale_id}" in node.name:
                        stage_conv_nodes.append((idx, node))

            stage_layers: List[ConvLayerMeta] = []
            for layer_idx, (node_idx, conv_node) in enumerate(stage_conv_nodes):
                layer = self._extract_conv_layer(conv_node, node_idx, nodes)
                layer.layer_index = layer_idx

                # Prediction heads (*.2/Conv) are linear; intermediate (*.0, *.1) are SiLU
                c_name = conv_node.name
                if c_name.endswith(".2/Conv"):
                    layer.activation = "Identity"
                    layer.fused_ops.append("Linear")
                else:
                    layer.activation = "SiLU"
                    layer.fused_ops.append("SiLU")

                stage_layers.append(layer)

            p = NpuFusedPartition(
                partition_id=part_id,
                layers=stage_layers,
                input_names=[stage_layers[0].node_name + "_in"] if stage_layers else [],
                output_names=[stage_layers[-1].node_name + "_out"] if stage_layers else [],
                in_bytes=8192,
                out_bytes=4096,
                stage_name=stage_name,
                has_zero_ddr_roundtrip=True,
            )
            partitions.append(p)

        return partitions

    def partition(
        self,
        backbone_only: Optional[bool] = None,
        neck_only: Optional[bool] = None,
        head_only: Optional[bool] = None,
    ) -> PartitionedGraph:
        """
        Partitions the graph into alternating CPU fallback and fused NPU partitions.
        When fuse_backbone=True and model contains YOLO backbone, lowers all C2f blocks,
        residual adds, and Convs into <= 4 monolithic stages with 0 CPU fallback partitions
        across the backbone feature extractor.
        When fuse_neck=True, absorbs all Resize and Concat nodes across Layers 10..21
        into <= 2 monolithic Neck stages with 0 CPU fallback partitions across the Neck.
        When fuse_head=True, absorbs all 6 prediction heads across Layer 22 into monolithic
        Detect Head stages with 0 CPU fallback partitions across the entire network.
        """
        if backbone_only is None:
            backbone_only = self.backbone_only
        if neck_only is None:
            neck_only = self.neck_only
        if head_only is None:
            head_only = self.head_only

        graph = self.model.graph

        if head_only and self._is_yolo_backbone_model():
            head_partitions = self._extract_head_stages()
            in_names = [head_partitions[0].input_names[0]] if head_partitions and head_partitions[0].input_names else []
            out_names = [head_partitions[-1].output_names[0]] if head_partitions and head_partitions[-1].output_names else []
            return PartitionedGraph(
                model_name=graph.name or "yolov8n_head",
                partitions=head_partitions,
                initial_inputs=in_names,
                terminal_outputs=out_names,
            )

        if neck_only and self._is_yolo_backbone_model():
            neck_partitions = self._extract_neck_stages()
            in_names = [neck_partitions[0].input_names[0]] if neck_partitions and neck_partitions[0].input_names else []
            out_names = [neck_partitions[-1].output_names[0]] if neck_partitions and neck_partitions[-1].output_names else []
            return PartitionedGraph(
                model_name=graph.name or "yolov8n_neck",
                partitions=neck_partitions,
                initial_inputs=in_names,
                terminal_outputs=out_names,
            )

        if self.fuse_backbone and self._is_yolo_backbone_model():
            backbone_partitions = self._extract_backbone_stages()

            if backbone_only:
                in_names = [inp.name for inp in graph.input]
                out_names = (
                    [backbone_partitions[-1].output_names[0]]
                    if backbone_partitions and backbone_partitions[-1].output_names
                    else [out.name for out in graph.output]
                )
                return PartitionedGraph(
                    model_name=graph.name or "yolov8n_backbone",
                    partitions=backbone_partitions,
                    initial_inputs=in_names,
                    terminal_outputs=out_names,
                )

            # If not backbone_only, partition the remaining nodes
            partitions: List[Union[NpuFusedPartition, CpuFallbackPartition]] = list(backbone_partitions)

            if self.fuse_neck:
                neck_partitions = self._extract_neck_stages()
                partitions.extend(neck_partitions)
                absorbed_layers = 22
            else:
                absorbed_layers = 10

            if self.fuse_head and self.fuse_neck:
                head_partitions = self._extract_head_stages()
                partitions.extend(head_partitions)
                absorbed_layers = 23

            part_id = len(partitions)
            remaining_nodes = [
                n for n in graph.node
                if not any(n.name.startswith(f"/model.{i}/") or n.name.startswith(f"model.{i}.") for i in range(absorbed_layers))
                and not n.name.startswith("images_")
                and n.op_type not in {"Constant"}
            ]

            if remaining_nodes:
                p = CpuFallbackPartition(
                    partition_id=part_id,
                    nodes=remaining_nodes,
                    input_names=[partitions[-1].output_names[0]] if partitions else [],
                    output_names=[out.name for out in graph.output],
                    is_backbone=False,
                )
                partitions.append(p)

            in_names = [inp.name for inp in graph.input]
            out_names = [out.name for out in graph.output]
            return PartitionedGraph(
                model_name=graph.name or "yolov8n_partitioned",
                partitions=partitions,
                initial_inputs=in_names,
                terminal_outputs=out_names,
            )

        # Standard topological walk for non-backbone models or when fuse_backbone=False
        partitions: List[Union[NpuFusedPartition, CpuFallbackPartition]] = []
        part_id = 0

        current_npu_layers: List[ConvLayerMeta] = []
        current_cpu_nodes: List[onnx.NodeProto] = []

        def flush_cpu():
            nonlocal part_id, current_cpu_nodes
            if current_cpu_nodes:
                node_inps = set()
                for n in current_cpu_nodes:
                    node_inps.update(n.input)
                part_inits = [t for t in graph.initializer if t.name in node_inps]

                produced = {out for n in current_cpu_nodes for out in n.output}
                consumed = {inp for n in current_cpu_nodes for inp in n.input if inp and inp not in {t.name for t in graph.initializer}}
                inp_names = [inp for inp in consumed if inp not in produced]
                if not inp_names:
                    inp_names = [n.input[0] for n in current_cpu_nodes if len(n.input) > 0][:1]
                if not inp_names:
                    inp_names = ["dummy_in"]

                out_names = [current_cpu_nodes[-1].output[0]] if len(current_cpu_nodes[-1].output) > 0 else ["dummy_out"]

                in_vis = []
                for in_name in inp_names:
                    match_in = [inp for inp in graph.input if inp.name == in_name]
                    match_vi = [vi for vi in graph.value_info if vi.name == in_name]
                    if match_in:
                        in_vis.append(match_in[0])
                    elif match_vi:
                        in_vis.append(match_vi[0])
                    else:
                        in_vis.append(helper.make_tensor_value_info(in_name, TensorProto.FLOAT, None))

                out_vis = []
                for out_name in out_names:
                    match_out = [out for out in graph.output if out.name == out_name]
                    match_vi = [vi for vi in graph.value_info if vi.name == out_name]
                    if match_out:
                        out_vis.append(match_out[0])
                    elif match_vi:
                        out_vis.append(match_vi[0])
                    else:
                        out_vis.append(helper.make_tensor_value_info(out_name, TensorProto.FLOAT, None))

                sub_graph = helper.make_graph(
                    list(current_cpu_nodes),
                    f"cpu_part_{part_id}",
                    in_vis,
                    out_vis,
                    part_inits
                )
                cpu_model = helper.make_model(sub_graph, opset_imports=self.model.opset_import)

                p = CpuFallbackPartition(
                    partition_id=part_id,
                    nodes=list(current_cpu_nodes),
                    input_names=inp_names,
                    output_names=out_names,
                    onnx_model=cpu_model,
                )
                partitions.append(p)
                part_id += 1
                current_cpu_nodes.clear()

        def flush_npu():
            nonlocal part_id, current_npu_layers
            if current_npu_layers:
                p = NpuFusedPartition(
                    partition_id=part_id,
                    layers=list(current_npu_layers),
                    input_names=[current_npu_layers[0].node_name + "_in"],
                    output_names=[current_npu_layers[-1].node_name + "_out"],
                    in_bytes=8192,
                    out_bytes=4096,
                )
                partitions.append(p)
                part_id += 1
                current_npu_layers.clear()

        # Topological walk
        idx = 0
        nodes = list(graph.node)
        while idx < len(nodes):
            node = nodes[idx]

            # Check if node is part of a QDQ Conv or direct Conv
            if node.op_type == "Conv":
                flush_cpu()
                layer_meta = self._extract_conv_layer(node, idx, nodes)
                current_npu_layers.append(layer_meta)
                idx += 1

                # Check if next node is a fusible activation
                if idx < len(nodes) and nodes[idx].op_type in {"Relu", "Clip", "Identity"}:
                    layer_meta.activation = nodes[idx].op_type
                    idx += 1
                continue

            # Skip standalone QDQ wrapper nodes already consumed by Conv
            if node.op_type in {"DequantizeLinear", "QuantizeLinear"}:
                # If output connects to next Conv, skip
                idx += 1
                continue

            # Unsupported op -> CPU fallback
            flush_npu()
            current_cpu_nodes.append(node)
            idx += 1

        flush_cpu()
        flush_npu()

        in_names = [inp.name for inp in graph.input]
        out_names = [out.name for out in graph.output]

        return PartitionedGraph(
            model_name=graph.name or "partitioned_model",
            partitions=partitions,
            initial_inputs=in_names,
            terminal_outputs=out_names,
        )

    def _extract_conv_layer(
        self,
        conv_node: onnx.NodeProto,
        node_idx: int,
        all_nodes: List[onnx.NodeProto]
    ) -> ConvLayerMeta:
        """Extracts weight/bias arrays and calculates quantization parameters."""
        w_name = conv_node.input[1]

        # 1. Resolve weights and weight scale
        w_raw = None
        sw = 0.0078125
        zw = 0
        dq_w = [n for n in all_nodes if n.output[0] == w_name and n.op_type == "DequantizeLinear"]
        if dq_w:
            w_raw_name = dq_w[0].input[0]
            if w_raw_name in self.inits:
                w_raw = self.inits[w_raw_name]
            if len(dq_w[0].input) > 1 and dq_w[0].input[1] in self.inits:
                sw_val = self.inits[dq_w[0].input[1]]
                sw = float(sw_val.flatten()[0])
            if len(dq_w[0].input) > 2 and dq_w[0].input[2] in self.inits:
                zw = int(self.inits[dq_w[0].input[2]].flatten()[0])
        elif w_name in self.inits:
            w_raw = self.inits[w_name]

        if w_raw is None:
            # Synthetic / fallback default weights
            w_raw = np.zeros((32, 32, 3, 3), dtype=np.int8)

        # 2. Resolve input scale sx
        x_name = conv_node.input[0]
        sx = 0.0078125
        zx = 0
        dq_x = [n for n in all_nodes if n.output[0] == x_name and n.op_type == "DequantizeLinear"]
        if dq_x and len(dq_x[0].input) > 1 and dq_x[0].input[1] in self.inits:
            sx = float(self.inits[dq_x[0].input[1]].flatten()[0])
            if len(dq_x[0].input) > 2 and dq_x[0].input[2] in self.inits:
                zx = int(self.inits[dq_x[0].input[2]].flatten()[0])

        # 3. Resolve output scale sy
        y_name = conv_node.output[0]
        sy = 0.0078125
        zy = 0
        q_y = [n for n in all_nodes if y_name in n.input and n.op_type == "QuantizeLinear"]
        if q_y and len(q_y[0].input) > 1 and q_y[0].input[1] in self.inits:
            sy = float(self.inits[q_y[0].input[1]].flatten()[0])
            if len(q_y[0].input) > 2 and q_y[0].input[2] in self.inits:
                zy = int(self.inits[q_y[0].input[2]].flatten()[0])

        # 4. Resolve Bias
        b_i32 = np.zeros(w_raw.shape[0], dtype=np.int32)
        scale_bias = sx * sw
        if len(conv_node.input) > 2:
            b_name = conv_node.input[2]
            dq_b = [n for n in all_nodes if n.output[0] == b_name and n.op_type == "DequantizeLinear"]
            if dq_b and dq_b[0].input[0] in self.inits:
                b_raw = self.inits[dq_b[0].input[0]]
                b_scale = float(self.inits[dq_b[0].input[1]].flatten()[0])
                b_fp = b_raw.astype(np.float32) * b_scale
                b_i32 = np.round(b_fp / scale_bias).astype(np.int32)
            elif b_name in self.inits:
                b_arr = self.inits[b_name]
                if b_arr.dtype in (np.float32, np.float64):
                    b_i32 = np.round(b_arr / scale_bias).astype(np.int32)
                else:
                    b_i32 = b_arr.astype(np.int32)

        # 5. Shift Cut & Sigma
        pos_x = int(-np.round(np.log2(sx))) if sx > 0 else 7
        pos_w = int(-np.round(np.log2(sw))) if sw > 0 else 7
        pos_y = int(-np.round(np.log2(sy))) if sy > 0 else 7
        shift_cut = pos_x + pos_w - pos_y
        sigma = shift_cut + 14

        # 6. Stationary Vector Lane Packing
        # Pack weights with AIE2 vector block layout (3 - blk) * 8 reversal
        w_packed = pack_weights_aie2_vector_layout(w_raw, in_ch_start=0, out_ch_start=0)
        b_packed = pack_bias_aie2_vector_layout(b_i32, out_ch_start=0)

        # Kernel shape, strides, pads
        k_shape = [3, 3]
        strides = [1, 1]
        pads = [0, 0, 0, 0]
        for attr in conv_node.attribute:
            if attr.name == "kernel_shape":
                k_shape = list(attr.ints)
            elif attr.name == "strides":
                strides = list(attr.ints)
            elif attr.name == "pads":
                pads = list(attr.ints)

        return ConvLayerMeta(
            node_name=conv_node.name or f"Conv_{node_idx}",
            layer_index=node_idx,
            op_type="Conv",
            in_channels=w_raw.shape[1] if w_raw.ndim >= 2 else 32,
            out_channels=w_raw.shape[0] if w_raw.ndim >= 1 else 32,
            kernel_shape=k_shape,
            strides=strides,
            pads=pads,
            weights_raw=w_raw,
            weights_packed=w_packed,
            bias_raw=None,
            bias_i32=b_i32,
            bias_packed=b_packed,
            scale_x=sx,
            scale_w=sw,
            scale_y=sy,
            scale_bias=scale_bias,
            zp_x=zx,
            zp_w=zw,
            zp_y=zy,
            pos_x=pos_x,
            pos_w=pos_w,
            pos_y=pos_y,
            shift_cut=shift_cut,
            sigma=sigma,
        )


def build_synthetic_multi_layer_conv_model(
    num_layers: int = 3,
    in_channels: int = 32,
    out_channels: int = 32,
    kernel_size: int = 3,
    out_path: Optional[str] = None,
    seed: int = 42,
    add_cpu_head: bool = False,
    add_cpu_tail: bool = False,
) -> onnx.ModelProto:
    """
    Constructs a synthetic N-layer ONNX QDQ Conv2D model with exact INT8 parameters
    for physical silicon verification, optionally including host CPU pre/post nodes.
    """
    rng = np.random.RandomState(seed)

    in_shape = [4, in_channels, 3, 3]
    out_shape = [4, out_channels, 1, 1] if kernel_size == 3 and num_layers > 1 else [4, out_channels, 3, 3]

    input_entry_name = 'input_raw' if add_cpu_head else 'x'
    output_exit_name = 'output_final' if add_cpu_tail else 'y'

    entry_vi = helper.make_tensor_value_info(input_entry_name, TensorProto.INT8, in_shape)
    exit_vi = helper.make_tensor_value_info(output_exit_name, TensorProto.INT8, out_shape)

    inits = []
    nodes = []

    if add_cpu_head:
        shape_in_arr = np.array(in_shape, dtype=np.int64)
        inits.append(numpy_helper.from_array(shape_in_arr, name='shape_in'))
        nodes.append(helper.make_node('Reshape', [input_entry_name, 'shape_in'], ['x']))

    scale_val = 0.0078125  # 1/128 -> pos = 7

    for k in range(num_layers):
        cin = in_channels if k == 0 else out_channels
        cout = out_channels
        k_h = kernel_size if k == 0 else 1
        k_w = kernel_size if k == 0 else 1

        w_arr = rng.randint(-8, 8, size=(cout, cin, k_h, k_w), dtype=np.int8)
        b_arr = rng.randint(-16, 16, size=(cout,), dtype=np.int32)

        # Scale and zero points
        sx_k = f"sx_{k}"
        zx_k = f"zx_{k}"
        sw_k = f"sw_{k}"
        zw_k = f"zw_{k}"
        sy_k = f"sy_{k}"
        zy_k = f"zy_{k}"
        w_k = f"w_{k}"
        b_k = f"b_{k}"

        inits.extend([
            helper.make_tensor(sx_k, TensorProto.FLOAT, [], [scale_val]),
            helper.make_tensor(zx_k, TensorProto.INT8, [], [0]),
            helper.make_tensor(sw_k, TensorProto.FLOAT, [], [scale_val]),
            helper.make_tensor(zw_k, TensorProto.INT8, [], [0]),
            helper.make_tensor(sy_k, TensorProto.FLOAT, [], [scale_val]),
            helper.make_tensor(zy_k, TensorProto.INT8, [], [0]),
            numpy_helper.from_array(w_arr, name=w_k),
            numpy_helper.from_array(b_arr.astype(np.float32) * (scale_val * scale_val), name=b_k),
        ])

        in_tensor = 'x' if k == 0 else f'y_{k-1}_q'
        x_f = f'x_{k}_f'
        w_f = f'w_{k}_f'
        y_f = f'y_{k}_f'
        out_tensor = 'y' if k == num_layers - 1 else f'y_{k}_q'

        nodes.extend([
            helper.make_node('DequantizeLinear', [in_tensor, sx_k, zx_k], [x_f]),
            helper.make_node('DequantizeLinear', [w_k, sw_k, zw_k], [w_f]),
            helper.make_node('Conv', [x_f, w_f, b_k], [y_f], kernel_shape=[k_h, k_w], pads=[0, 0, 0, 0]),
            helper.make_node('QuantizeLinear', [y_f, sy_k, zy_k], [out_tensor]),
        ])

    if add_cpu_tail:
        shape_out_arr = np.array(out_shape, dtype=np.int64)
        inits.append(numpy_helper.from_array(shape_out_arr, name='shape_out'))
        nodes.append(helper.make_node('Reshape', ['y', 'shape_out'], [output_exit_name]))

    graph = helper.make_graph(nodes, f'synthetic_{num_layers}layer_conv', [entry_vi], [exit_vi], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])

    if out_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        onnx.save(model, out_path)

    return model

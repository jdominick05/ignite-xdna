#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
    activation: Optional[str] = None  # None, "Relu", "Clip", "Identity"


@dataclass
class NpuFusedPartition:
    """A maximal fused sequence of Conv2D layers executable on AIE2 hardware."""
    partition_id: int
    layers: List[ConvLayerMeta] = field(default_factory=list)
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    in_bytes: int = 8192
    out_bytes: int = 4096

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


class GraphPartitioner:
    """
    Walks an ONNX DAG to partition operations into host CPU fallback nodes
    and maximal fusible NPU AIE2 subgraphs.
    """

    SUPPORTED_FUSIBLE_OPS = {"Conv", "Relu", "Clip", "Identity"}

    def __init__(self, model_or_path: Union[str, Path, onnx.ModelProto]):
        if isinstance(model_or_path, (str, Path)):
            self.model_path = str(model_or_path)
            self.model = onnx.load(self.model_path)
        elif isinstance(model_or_path, onnx.ModelProto):
            self.model_path = "<in-memory>"
            self.model = model_or_path
        else:
            raise TypeError(f"Expected path or ModelProto, got {type(model_or_path)}")

        self.inits = {t.name: numpy_helper.to_array(t) for t in self.model.graph.initializer}

    def partition(self) -> PartitionedGraph:
        """
        Partitions the graph into alternating CPU fallback and fused NPU partitions.
        """
        graph = self.model.graph
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

                inp_names = [current_cpu_nodes[0].input[0]]
                out_names = [current_cpu_nodes[-1].output[0]]

                in_vis = [inp for inp in graph.input if inp.name in inp_names]
                if not in_vis:
                    in_vis = [helper.make_tensor_value_info(inp_names[0], TensorProto.INT8, None)]
                out_vis = [out for out in graph.output if out.name in out_names]
                if not out_vis:
                    out_vis = [helper.make_tensor_value_info(out_names[0], TensorProto.INT8, None)]

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

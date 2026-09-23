#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
src/ignite_xdna/quantization/engine.py

Post-Training Quantization (PTQ) Orchestration Engine for AMD Phoenix XDNA1 (AIE2).
Integrates Cross-Layer Equalization (CLE), High-Bias Absorption, and Adaptive Rounding
(AdaRound) into an end-to-end standalone Python pipeline without external Docker/Vitis-AI dependencies.

Produces:
  1. Quantized ONNX model with equalized dynamic ranges and AdaRound integer weights.
  2. JSON scale sidecar (scales.json) ready for direct ingestion by tools/ignite_compile.py.
"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import numpy as np
import onnx
from onnx import helper, numpy_helper

from ignite_xdna.quantization.cle import cross_layer_equalize, CleReport
from ignite_xdna.quantization.adaround import adaround_optimize, FastFinetuneConfig, AdaRoundReport
from ignite_xdna.quantization.calib import CalibrationDataset, collect_calibration_activations


# YOLOv8 raw detection head convolution outputs (matching AIE2 hardware compilation target)
HEAD_OUTS = [
    '/model.22/cv2.0/cv2.0.2/Conv_output_0',
    '/model.22/cv2.1/cv2.1.2/Conv_output_0',
    '/model.22/cv2.2/cv2.2.2/Conv_output_0',
    '/model.22/cv3.0/cv3.0.2/Conv_output_0',
    '/model.22/cv3.1/cv3.1.2/Conv_output_0',
    '/model.22/cv3.2/cv3.2.2/Conv_output_0',
]


@dataclass
class QuantizationConfig:
    """Configuration options for standalone PTQ pipeline."""
    use_cle: bool = True
    use_adaround: bool = True
    num_calib: int = 128
    adaround_iterations: int = 1000
    adaround_lr: float = 0.05
    append_bias: bool = True
    absorb_high_bias: bool = True
    high_bias_threshold: float = 0.2
    input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640)
    input_scale: float = 1.0 / 256.0
    input_zero_point: int = -128
    input_dtype: str = "int8"


class PTQEngine:
    """
    Main Post-Training Quantization Engine for YOLOv8 and vision networks.
    Coordinates graph extraction, CLE dynamic range equalization, AdaRound
    continuous relaxation, integer locking, and scale JSON emission.
    """

    def __init__(self, config: Optional[QuantizationConfig] = None):
        self.config = config or QuantizationConfig()

    def prepare_model(self, model_path: Union[str, Path]) -> onnx.ModelProto:
        """
        Loads ONNX model. If the model contains the full DFL/anchor-decode tail,
        extracts the 6 raw detection conv outputs matching AIE2 compilation targets.
        """
        model_path = Path(model_path)
        model = onnx.load(str(model_path))

        produced = {o for n in model.graph.node for o in n.output}
        is_full_yolo = all(h in produced for h in HEAD_OUTS) and any(
            n.op_type in ("Reshape", "Softmax", "Transpose") for n in model.graph.node
        )

        if is_full_yolo:
            # Extract cut head subgraph
            input_name = model.graph.input[0].name
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                onnx.utils.extract_model(str(model_path), tmp_path, [input_name], HEAD_OUTS)
                cut_model = onnx.load(tmp_path)
                return cut_model
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        return model

    def quantize(
        self,
        model_path: Union[str, Path],
        calib_data_dir: Union[str, Path],
        output_onnx_path: Union[str, Path],
        output_scales_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        Executes end-to-end PTQ pipeline and saves artifacts.

        Args:
            model_path: Path to float ONNX model (yolov8n.onnx or yolov8n_cut.onnx).
            calib_data_dir: Path to calibration dataset directory (e.g. data/coco128/).
            output_onnx_path: Destination path for quantized ONNX model.
            output_scales_path: Destination path for quantization scales JSON.

        Returns:
            Dictionary containing execution summary, metrics, and artifact paths.
        """
        t0 = time.perf_counter()
        cfg = self.config

        print(f"[*] Loading and preparing model: {model_path}")
        model = self.prepare_model(model_path)
        print(f"    Graph contains {len(model.graph.node)} nodes and {len(model.graph.initializer)} initializers.")

        cle_report: Optional[CleReport] = None
        if cfg.use_cle:
            print("[*] Running Cross-Layer Equalization (CLE) & High-Bias Absorption...")
            model, cle_report = cross_layer_equalize(
                model,
                append_bias=cfg.append_bias,
                absorb_high_bias=cfg.absorb_high_bias,
                high_bias_threshold=cfg.high_bias_threshold,
            )
            print(f"    [OK] Equalized {cle_report.num_pairs_equalized} Conv pairs across graph.")
            print(f"    [OK] Absorbed {cle_report.num_biases_absorbed} high-residual biases to prevent SRS saturation.")

        adaround_report: Optional[AdaRoundReport] = None
        if cfg.use_adaround:
            print(f"[*] Ingesting calibration data from: {calib_data_dir}")
            calib_dataset = CalibrationDataset(
                data_dir=calib_data_dir,
                max_samples=cfg.num_calib,
                img_size=cfg.input_shape[2],
            )
            print(f"    Loaded {len(calib_dataset)} calibration images.")

            print("[*] Collecting intermediate layer activations via ONNX Runtime...")
            calib_activations = collect_calibration_activations(
                model,
                calib_dataset,
                max_samples=min(16, len(calib_dataset)),
            )

            print(f"[*] Running AdaRound optimization ({cfg.adaround_iterations} iters/layer, cosine beta annealing)...")
            ada_cfg = FastFinetuneConfig(
                data_size=len(calib_dataset),
                num_iterations=cfg.adaround_iterations,
                lr=cfg.adaround_lr,
            )
            model, adaround_report = adaround_optimize(
                model,
                calib_data=calib_activations,
                config=ada_cfg,
            )
            print(f"    [OK] Optimized and locked integer weights across {adaround_report.layers_optimized} layers.")
            print(f"    [OK] Completed {adaround_report.total_iterations} iterations in {adaround_report.elapsed_seconds:.2f}s.")

        # Compute per-layer symmetric scales and fixed-point positions
        scales_dict: Dict[str, Any] = {
            "producer": "Ignite-PTQ",
            "version": "1.0.0",
            "input_scale": cfg.input_scale,
            "input_zero_point": cfg.input_zero_point,
            "input_dtype": cfg.input_dtype,
            "cle_enabled": cfg.use_cle,
            "adaround_enabled": cfg.use_adaround,
            "scales": {},
            "positions": {},
        }

        inits = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
        for conv in model.graph.node:
            if conv.op_type == "Conv" and len(conv.input) > 1:
                w_name = conv.input[1]
                if w_name in inits:
                    w = inits[w_name]
                    max_abs = float(np.max(np.abs(w)))
                    sw = max(max_abs / 127.0, 1e-7)
                    scales_dict["scales"][w_name] = sw
                    # Position pos_w = -round(log2(sw))
                    pos_w = int(-round(math.log2(sw))) if sw > 0 else 7
                    scales_dict["positions"][w_name] = pos_w

        # Ensure output directories exist
        out_onnx = Path(output_onnx_path).resolve()
        out_scales = Path(output_scales_path).resolve()
        out_onnx.parent.mkdir(parents=True, exist_ok=True)
        out_scales.parent.mkdir(parents=True, exist_ok=True)

        # Save quantized ONNX model
        onnx.save(model, str(out_onnx))
        print(f"[+] Saved quantized ONNX model: {out_onnx} ({out_onnx.stat().st_size / (1024*1024):.2f} MB)")

        # Save scales JSON
        with open(out_scales, "w") as f:
            json.dump(scales_dict, f, indent=2)
        print(f"[+] Saved quantization scales JSON: {out_scales}")

        elapsed = time.perf_counter() - t0
        summary = {
            "status": "SUCCESS",
            "model_path": str(out_onnx),
            "scales_path": str(out_scales),
            "elapsed_seconds": round(elapsed, 2),
            "cle_pairs": cle_report.num_pairs_equalized if cle_report else 0,
            "cle_biases_absorbed": cle_report.num_biases_absorbed if cle_report else 0,
            "adaround_layers": adaround_report.layers_optimized if adaround_report else 0,
        }
        print(f"[+] Quantization Complete in {elapsed:.2f}s!")
        return summary


def quantize_model(
    model_path: Union[str, Path],
    calib_data_dir: Union[str, Path],
    output_onnx_path: Union[str, Path],
    output_scales_path: Union[str, Path],
    config: Optional[QuantizationConfig] = None,
) -> Dict[str, Any]:
    """Convenience function executing PTQEngine quantization."""
    engine = PTQEngine(config=config)
    return engine.quantize(
        model_path=model_path,
        calib_data_dir=calib_data_dir,
        output_onnx_path=output_onnx_path,
        output_scales_path=output_scales_path,
    )

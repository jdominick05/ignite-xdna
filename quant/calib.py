"""Exact-sample MinMSE calibration for folded, static ResNet classifiers.

Activations use float16 storage followed by float32 arithmetic, matching the
audited All-mode producer contract. One tensor is loaded at a time; no histogram
approximation or reference position table is used. See DESIGN.md section 2.2.
"""
from dataclasses import asdict, replace
from pathlib import Path
import shutil
import tempfile

import numpy as np
import onnx
import onnxruntime as ort

from .graph import Graph
from .pow2 import TensorQ, qrange, scale2pos, sqerr
from .qdq import prunable_tensors, quantizable_tensors
from .sources import ImageFolderSource


def choose_pow2_minmse(x: np.ndarray, dtype: str) -> tuple[int, dict]:
    """Five candidates around symmetric min/max, first minimum wins ties."""
    value = np.asarray(x, dtype=np.float32)
    if not value.size or not np.all(np.isfinite(value)):
        raise ValueError("MinMSE requires nonempty finite float32 samples")
    low, high = qrange(dtype)
    zp = 128 if dtype == "uint8" else 0
    vmin, vmax = value.min(), value.max()
    extent = np.minimum(max(abs(vmin), abs(vmax)), np.finfo(np.float32).max / 2)
    # Subtraction happens in float32, division in float64, then scale is float32.
    span = np.float32(np.float32(extent) - np.float32(-extent))
    scale = np.float32(np.float64(span) / (high - low))
    if scale < np.finfo(np.float32).tiny:
        if dtype == "uint8":
            raise ValueError("Degenerate activation range gives vendor zp0; outside the supported zp128 dialect")
        scale = np.float32(1)
    base = scale2pos(scale)
    candidates = list(range(base - 1, base + 4))
    if min(candidates) < -127 or max(candidates) > 127:
        raise ValueError("MinMSE candidate range exceeds supported positions")
    errors = [sqerr(value, pos, zp, dtype) for pos in candidates]
    if not all(np.isfinite(errors)):
        raise ValueError("MinMSE float32 error accumulation overflowed")
    chosen = candidates[min(range(len(errors)), key=errors.__getitem__)]
    return chosen, {"count": int(value.size), "vmin": float(vmin), "vmax": float(vmax),
                    "minmax_pos": base, "candidate_positions": candidates,
                    "candidate_sqerr": errors, "minmse_pos": chosen}


def collect_and_choose(g: Graph, source: ImageFolderSource, scratch: Path) -> tuple[dict[str, TensorQ], dict]:
    """Run the float graph once per image, spool samples, choose all positions.

    Conv/Add outputs consumed only by a Relu receive a temporary QDQ pair that emit
    removes, so their positions never reach the file; they are neither spooled nor
    searched. The reference producer spools them anyway (All mode); the chosen
    positions are unaffected because nothing downstream reads them.
    """
    all_acts, weights, sharing = quantizable_tensors(g)
    pruned = prunable_tensors(g)
    acts = [name for name in all_acts if name not in pruned]
    g.infer_shapes()
    inputs = list(g.model.graph.input)
    if len(inputs) != 1 or inputs[0].name != source.input_name:
        raise ValueError("Calibration supports one input matching the source input name")
    shapes = {name: g.value_shape(name) for name in acts}
    if any(shape is None or any(d <= 0 for d in shape) for shape in shapes.values()):
        raise ValueError("Calibration requires inferred, fully static tensor shapes")
    if any(g.value_dtype(name) != onnx.TensorProto.FLOAT for name in acts):
        raise ValueError("Calibration requires float32 activations")
    estimated_bytes = sum(int(np.prod(shape)) * 2 for shape in shapes.values()) * len(source)
    scratch = Path(scratch).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(scratch).free
    if free_bytes < estimated_bytes + 2 * 1024**3:
        raise OSError(f"Need {estimated_bytes} bytes of sample storage plus 2 GiB reserve; {free_bytes} free")
    augmented = onnx.ModelProto()
    augmented.CopyFrom(g.model)
    infos = {v.name: v for v in (*g.model.graph.input, *g.model.graph.value_info, *g.model.graph.output)}
    del augmented.graph.output[:]
    augmented.graph.output.extend(infos[name] for name in acts)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(augmented.SerializeToString(), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    report = {"store": "exact_float16", "mode": "All", "images": len(source),
              "listing": [p.as_posix() for p in source.listing()],
              "spooled_tensors": len(acts), "skipped_prunable_tensors": len(pruned),
              "sample_bytes": estimated_bytes, "free_bytes_before": free_bytes,
              "ort_version": ort.__version__, "graph_optimization": "ORT_DISABLE_ALL",
              "tensors": {}, "sharing": sharing}
    positions = {}
    with tempfile.TemporaryDirectory(prefix="owned-calib-", dir=scratch) as directory:
        root = Path(directory).resolve()
        if not root.is_relative_to(scratch):
            raise ValueError("Calibration temporary directory escaped its requested root")
        paths = {name: root / f"tensor_{i}.f16" for i, name in enumerate(acts)}
        print(f"CALIBRATION images={len(source)} tensors={len(acts)} skipped_prunable={len(pruned)} "
              f"sample_bytes={estimated_bytes}", flush=True)
        count = 0
        for count, image in enumerate(source, 1):
            if image.dtype != np.float32 or image.shape != shapes[source.input_name]:
                raise ValueError("Preprocessing must return the model's float32 batch-1 shape")
            outputs = session.run(acts, {source.input_name: image})
            for name, output in zip(acts, outputs, strict=True):
                if output.shape != shapes[name]:
                    raise ValueError(f"Activation shape changed for {name}")
                with np.errstate(over="ignore"):
                    samples = output.astype(np.float16)
                if not np.all(np.isfinite(samples)):
                    raise ValueError(f"Nonfinite float16 calibration samples for {name}")
                with paths[name].open("ab") as stream:
                    samples.tofile(stream)
            if count % 16 == 0 or count == len(source):
                print(f"COLLECT {count}/{len(source)}", flush=True)
        if count != len(source):
            raise ValueError("Calibration source length changed during collection")
        del session, outputs, samples
        for i, name in enumerate(acts, 1):
            samples = np.fromfile(paths[name], dtype=np.float16).astype(np.float32)
            if samples.size != int(np.prod(shapes[name])) * count:
                raise ValueError(f"Incomplete calibration spool for {name}")
            pos, stats = choose_pow2_minmse(samples, "uint8")
            positions[name] = TensorQ(name, "uint8", pos, 128, "activation_minmse")
            report["tensors"][name] = stats
            del samples
            if i % 16 == 0 or i == len(acts):
                print(f"CHOOSE {i}/{len(acts)}", flush=True)
    for name in weights:
        pos, stats = choose_pow2_minmse(g.initializer(name), "int8")
        positions[name] = TensorQ(name, "int8", pos, 0, "initializer_minmse")
        report["tensors"][name] = stats
    for name, provider in sharing.items():
        positions[name] = replace(positions[provider], name=name, source="shared:" + provider)
    report["emission_positions"] = {name: asdict(tq) for name, tq in positions.items()}
    return positions, report

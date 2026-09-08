"""Measure one ResNet quantization mutation on CPU and the real XDNA1 NPU.

Use scripts/quant-probe.sh for environment activation, variant logs and isolation.
The image slice measures output agreement, not dataset accuracy.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from npu.ep_report import read_report
from npu.paths import CACHE_DIR, RESNET_CACHE_KEY
from npu.session import build_session, clear_cache, resolve_xclbin
from quant.graph import Graph
from quant.probe import NOTES, mutate, output_difference
from quant.quantize import file_hash
from quant.sources import ImageFolderSource


def run_outputs(session, images, warmup):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    for _ in range(warmup):
        session.run([output_name], {input_name: images[0]})
    outputs, timings = [], []
    for image in images:
        start = time.perf_counter()
        output = session.run([output_name], {input_name: image})[0]
        timings.append((time.perf_counter() - start) * 1000)
        outputs.append(output.copy())
    result = np.concatenate(outputs, axis=0)
    if not np.all(np.isfinite(result)):
        raise ValueError("Model produced nonfinite outputs")
    return result, {"mean": float(np.mean(timings)), "median": float(np.median(timings)),
                    "p95": float(np.percentile(timings, 95)), "runs": len(timings), "warmup": warmup}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("models/resnet50_own_nocle_c64.onnx"))
    parser.add_argument("--mutation", choices=NOTES, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=Path("data/eval"))
    parser.add_argument("--cfg", type=Path, default=Path("models/preprocess_config.json"))
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--full-eval", action="store_true", help="Use every labeled image and report full-set accuracy")
    parser.add_argument("--cpu-only", action="store_true", help="Validate optimized/unoptimized CPU outputs without using the NPU")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--baseline-result", type=Path)
    args = parser.parse_args()
    sidecar = Path(str(args.out) + ".probe.json")
    arrays_path = Path(str(args.out) + ".outputs.npz")
    ep_copy = Path(str(args.out) + ".ep.json")
    if any(p.exists() for p in (args.out, sidecar, arrays_path, ep_copy)):
        parser.error("Choose a new output stem; model and evidence files are never overwritten")
    if args.n < 1 or args.warmup < 0:
        parser.error("--n must be positive and --warmup nonnegative")
    g, mutation = mutate(Graph.load(args.base), args.mutation)
    g.save(args.out)
    config = json.loads(args.cfg.read_text(encoding="utf-8"))
    source = ImageFolderSource(args.images, config, None if args.full_eval else args.n, g.model.graph.input[0].name)
    images = list(source)
    labels = None
    if args.full_eval:
        label_table = json.loads((args.images / "labels.json").read_text(encoding="utf-8"))
        labels = np.array([label_table[p.name] for p in source.listing()], dtype=np.int64)
    if any(image.shape != g.value_shape(source.input_name) for image in images):
        raise ValueError("Preprocessing shape differs from graph input")
    report = {"status": "prepared", "machine": "Desktop 2 / Ryzen 7 8700G",
              "mutation": asdict(mutation), "base_sha256": file_hash(args.base),
              "model_sha256": file_hash(args.out), "ort_version": ort.__version__,
              "images": len(images), "scope": "full labeled evaluation dataset" if args.full_eval else "diagnostic output agreement; not dataset accuracy",
              "listing": [p.as_posix() for p in source.listing()], "preprocess": config,
              "inputs_sha256": hashlib.sha256(b"".join(i.tobytes() for i in images)).hexdigest(),
              "cache_key": RESNET_CACHE_KEY, "fresh": True}

    def save():
        sidecar.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def accuracy(outputs):
        if labels is None:
            return None
        pred5 = np.argsort(outputs, axis=1)[:, ::-1][:, :5]
        return {"images": len(labels), "top1_percent": float(100 * np.mean(pred5[:, 0] == labels)),
                "top5_percent": float(100 * np.mean(np.any(pred5 == labels[:, None], axis=1)))}

    print("PROBE_PREPARED", json.dumps(report, indent=2), flush=True)
    save()
    try:
        cpu_base = build_session(args.base, "cpu", RESNET_CACHE_KEY, log_severity=3)
        base_out, _ = run_outputs(cpu_base, images, args.warmup)
        del cpu_base
        cpu = build_session(args.out, "cpu", RESNET_CACHE_KEY, log_severity=3)
        cpu_out, report["cpu_latency_ms"] = run_outputs(cpu, images, args.warmup)
        del cpu
        report["cpu_vs_baseline_cpu"] = output_difference(cpu_out, base_out)
        # INT32-bias QDQ can trigger an optimizer rewrite with different results.
        # Keep the unfused ONNX computation as a separate numerical reference.
        def unoptimized(path):
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            options.log_severity_level = 3
            session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
            return run_outputs(session, images, args.warmup)[0]
        raw_cpu = unoptimized(args.out)
        raw_base = raw_cpu if report["model_sha256"] == report["base_sha256"] else unoptimized(args.base)
        report["optimized_cpu_vs_unoptimized_cpu"] = output_difference(cpu_out, raw_cpu)
        report["unoptimized_cpu_vs_baseline_unoptimized_cpu"] = output_difference(raw_cpu, raw_base)
        if labels is not None:
            report["cpu_accuracy"] = accuracy(cpu_out)
            report["baseline_cpu_accuracy"] = accuracy(base_out)
        report["status"] = "cpu_validated"
        print("CPU_COMPARISON", json.dumps(report["cpu_vs_baseline_cpu"]), flush=True)
        save()
    except Exception:
        report["status"] = "cpu_failed"
        report["error"] = traceback.format_exc()
        save()
        print("PROBE_RESULT", json.dumps(report, indent=2), flush=True)
        raise SystemExit(1)
    if args.cpu_only:
        report["status"] = "cpu_only_completed"
        save()
        print("PROBE_RESULT", json.dumps(report, indent=2), flush=True)
        return
    witness = subprocess.run(["C:/Windows/System32/AMD/xrt-smi.exe", "examine", "-r", "aie-partitions"],
                              capture_output=True, text=True, check=True)
    print("PRE_NPU_CONTEXT_WITNESS\n" + witness.stdout, flush=True)
    report["context_witness"] = witness.stdout
    if "No hardware contexts running" not in witness.stdout:
        report["status"] = "busy_device"
        save()
        raise SystemExit(3)
    xclbin = Path(resolve_xclbin())
    report["xclbin_sha256"] = file_hash(xclbin)
    report["xclbin"] = str(xclbin)
    cache_path = (CACHE_DIR / RESNET_CACHE_KEY).resolve()
    if cache_path.parent != CACHE_DIR.resolve():
        raise ValueError("Cache path escaped the worktree")
    clear_cache(RESNET_CACHE_KEY)
    report["status"] = "npu_build_started"
    save()
    try:
        npu = build_session(args.out, "npu", RESNET_CACHE_KEY, str(xclbin), log_severity=3)
        ep = read_report(cache_path / "vitisai_ep_report.json")
        ep_copy.write_bytes(ep.path.read_bytes())
        report["ep"] = ep.summary()
        print("EP_PLACEMENT", json.dumps(report["ep"], indent=2), flush=True)
        npu_out, report["npu_latency_ms"] = run_outputs(npu, images, args.warmup)
        report["npu_vs_cpu"] = output_difference(npu_out, cpu_out)
        report["npu_vs_unoptimized_cpu"] = output_difference(npu_out, raw_cpu)
        if labels is not None:
            report["npu_accuracy"] = accuracy(npu_out)
        if args.baseline_result:
            baseline = json.loads(args.baseline_result.read_text(encoding="utf-8"))
            if any(baseline[key] != report[key] for key in ("base_sha256", "inputs_sha256", "xclbin_sha256")):
                raise ValueError("Baseline differs in model, inputs or xclbin")
            base_arrays = Path(str(args.baseline_result).removesuffix(".probe.json") + ".outputs.npz")
            with np.load(base_arrays) as archive:
                report["npu_vs_baseline_npu"] = output_difference(npu_out, archive["npu"])
        np.savez(arrays_path, cpu=cpu_out, npu=npu_out, base_cpu=base_out, cpu_unoptimized=raw_cpu)
        report["status"] = "completed"
        del npu
    except Exception:
        report["status"] = "npu_failed"
        report["error"] = traceback.format_exc()
    save()
    print("PROBE_RESULT", json.dumps(report, indent=2), flush=True)
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

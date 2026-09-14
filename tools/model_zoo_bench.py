#!/usr/bin/env python3
"""Model-zoo latency suite: Ignition's live_ignition.py over a list of models, one JSON file per suite,
and the comparison tables in docs/MODEL_ZOO_BENCHMARKS.md.

    python tools/model_zoo_bench.py run --suite onnx-cpu                  # 300 frames per ONNX model on CPU
    python tools/model_zoo_bench.py run --suite npu --frames 500          # the .ignite containers on Device 0
    python tools/model_zoo_bench.py report                                # rewrite the tables from the JSON

Every model runs in its own live_ignition.py process (--headless --frames N --json), so each
number is a cold process with its own warm-up. The runner records the return code, keeps the
process log under results/model_zoo/, scrubs the profile path to C:\\Users\\<user>, and exits 1
if any run failed. The npu suite reads `xrt-smi examine -r aie-partitions` before every run
and stores what it said: a foreign hardware context makes the number contention, not a result.

The source defaults to assets/bus.jpg: a still image gives every model the same frame, so the
latency does not depend on what the camera sees (YOLO decode cost grows with detections).
Pass --source 0 for the webcam. Only runs actually executed are written; nothing is estimated.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "MODEL_ZOO_BENCHMARKS.md"
LOG_DIR = ROOT / "results" / "model_zoo"
XRT_SMI = Path(r"C:\Windows\System32\AMD\xrt-smi.exe")

SUITES: Dict[str, Dict[str, Any]] = {
    "onnx-cpu": {
        "json": ROOT / "results" / "benchmarks_onnx_cpu.json",
        "frames": 300,
        "models": ["models/yolov8s_cut_xint8.onnx", "models/yolo11n_no_c2psa_cut_xint8.onnx",
                   "models/sesr_m7_xint8.onnx", "models/resnet50_xint8_c64.onnx"],
    },
    "npu": {
        "json": ROOT / "results" / "benchmarks_npu_silicon.json",
        "frames": 500,
        "models": ["build/yolov8s.ignite", "build/sesr_m7.ignite"],
    },
}

_PROFILE = re.compile(r"C:[\\/]+Users[\\/]+(?!<user>)[^\\/\s\"']+", re.IGNORECASE)
_ROOTS: List[Any] = []   # (compiled pattern, label): checkout roots replaced by a label in every string


def set_roots(pairs) -> None:
    """Checkout paths (either slash direction, any case) to print as labels such as <ignite-xdna>."""
    _ROOTS.clear()
    for path, label in pairs:
        parts = [re.escape(p) for p in re.split(r"[\\/]+", str(Path(path).resolve())) if p]
        _ROOTS.append((re.compile(r"[\\/]+".join(parts), re.IGNORECASE), label))


def scrub(value: Any) -> Any:
    """Checkout roots to labels, then the Windows profile directory to C:\\Users\\<user>, in every string."""
    if isinstance(value, str):
        for pattern, label in _ROOTS:
            value = pattern.sub(label, value)
        return _PROFILE.sub(lambda m: "C:\\Users\\<user>" if "\\" in m.group(0) else "C:/Users/<user>", value)
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value


def find_ignition(explicit: Optional[str]) -> Path:
    candidates = [Path(explicit)] if explicit else []
    if os.environ.get("IGNITION_ROOT"):
        candidates.append(Path(os.environ["IGNITION_ROOT"]))
    candidates += [p / "Ignition" for p in ROOT.parents]
    for c in candidates:
        if (c / "live_ignition.py").is_file():
            return c
    raise FileNotFoundError("Ignition checkout with live_ignition.py not found; pass --ignition")


def git_head(path: Path) -> str:
    try:
        # "-dirty" when tracked files differ from the commit, so a number is never credited to code it did not run
        return subprocess.run(["git", "-C", str(path), "describe", "--always", "--dirty", "--abbrev=7"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance only
        return ""


def xrt_partitions() -> str:
    if not XRT_SMI.is_file():
        return "xrt-smi not found"
    res = subprocess.run([str(XRT_SMI), "examine", "-r", "aie-partitions"], capture_output=True, text=True, timeout=60)
    return (res.stdout + res.stderr).strip()


def run_suite(args: argparse.Namespace) -> int:
    suite = SUITES[args.suite]
    ignition = find_ignition(args.ignition)
    frames = args.frames or suite["frames"]
    source = args.source or str(ROOT / "assets" / "bus.jpg")
    models = args.models or suite["models"]
    set_roots([(ignition, "<Ignition>"), (ROOT, "<ignite-xdna>")])
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    runs: List[Dict[str, Any]] = []
    for rel in models:
        model = (ROOT / rel) if not Path(rel).is_absolute() else Path(rel)
        stem = model.stem.replace(".", "_")
        out_json = LOG_DIR / f"{args.suite}_{stem}.json"
        log = LOG_DIR / f"{args.suite}_{stem}.log"
        entry: Dict[str, Any] = {"model_path": str(rel), "log": str(log.relative_to(ROOT)).replace("\\", "/")}
        if args.suite == "npu":
            entry["xrt_smi_before"] = xrt_partitions()
        cmd = [sys.executable, str(ignition / "live_ignition.py"), "--model", str(model), "--source", source,
               "--headless", "--frames", str(frames), "--warmup", str(args.warmup), "--json", str(out_json)]
        if out_json.exists():
            out_json.unlink()
        print(f"[suite] {args.suite}: {model.name} ({frames} frames, source {Path(source).name})", flush=True)
        t0 = time.perf_counter()
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout, cwd=str(ignition))
        entry["wall_s"] = round(time.perf_counter() - t0, 1)
        entry["returncode"] = res.returncode
        text = f"$ {' '.join(cmd)}\n" + res.stdout + (("\n[stderr]\n" + res.stderr) if res.stderr.strip() else "")
        log.write_text(scrub(text), encoding="utf-8", newline="\n")
        if out_json.is_file():
            entry.update(json.loads(out_json.read_text(encoding="utf-8")))
            out_json.unlink()
        if args.suite == "npu":
            entry["xrt_smi_after"] = xrt_partitions()
        g = entry.get("g2g_ms", {})
        print(f"[suite]   rc {res.returncode} | G2G mean {g.get('mean', float('nan')):.3f} ms "
              f"P99 {g.get('p99', float('nan')):.3f} | RSS drift {entry.get('rss_mb', {}).get('drift', float('nan')):+.2f} MB",
              flush=True)
        runs.append(entry)
    doc = {"suite": args.suite, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "frames": frames, "warmup": args.warmup, "source": source,
           "ignite_xdna_commit": git_head(ROOT), "ignition_commit": git_head(ignition),
           "all_returncodes_zero": all(r["returncode"] == 0 for r in runs), "runs": runs}
    suite["json"].write_text(json.dumps(scrub(doc), indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"[suite] wrote {suite['json'].relative_to(ROOT)}", flush=True)
    render_report()
    return 0 if doc["all_returncodes_zero"] else 1


def _fmt(v: Any, digits: int = 2) -> str:
    return "n/a" if v is None or (isinstance(v, float) and v != v) else f"{v:.{digits}f}"


def table(doc: Dict[str, Any]) -> str:
    rows = ["| Model | Task | Backend | Frames | G2G mean (ms) | P50 | P95 | P99 | FPS | Preprocess | Network | "
            "Post | Dispatch | Readback | RSS drift (MB) | rc |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in doc["runs"]:
        g, s = r.get("g2g_ms", {}), r.get("stages_ms", {})
        rows.append(f"| `{Path(r['model_path']).name}` | {r.get('task', 'n/a')} | {r.get('backend', 'n/a')} | "
                    f"{r.get('frames_timed', 0)} | {_fmt(g.get('mean'))} | {_fmt(g.get('p50'))} | {_fmt(g.get('p95'))} | "
                    f"{_fmt(g.get('p99'))} | {_fmt(r.get('fps_from_mean'), 1)} | {_fmt(s.get('preprocess'))} | "
                    f"{_fmt(s.get('network'))} | {_fmt(s.get('postprocess'))} | {_fmt(s.get('dispatch'))} | "
                    f"{_fmt(s.get('readback'))} | {_fmt(r.get('rss_mb', {}).get('drift'))} | {r['returncode']} |")
    host = next((r.get("host") for r in doc["runs"] if r.get("host")), {}) or {}
    meta = (f"Suite `{doc['suite']}`, {doc['created_utc']}, {doc['frames']} timed frames after {doc['warmup']} warm-up, "
            f"source `{Path(doc['source']).name}`, host `{host.get('hostname', '?')}`, onnxruntime "
            f"{host.get('onnxruntime', '?')}, ignite-xdna `{doc.get('ignite_xdna_commit')}`, Ignition "
            f"`{doc.get('ignition_commit')}`. All return codes zero: **{doc['all_returncodes_zero']}**.")
    return meta + "\n\n" + "\n".join(rows)


def render_report() -> None:
    """Replace the generated blocks between the BEGIN/END markers of docs/MODEL_ZOO_BENCHMARKS.md."""
    if not DOC.is_file():
        return
    text = DOC.read_text(encoding="utf-8")
    for name, suite in SUITES.items():
        if not suite["json"].is_file():
            continue
        doc = json.loads(suite["json"].read_text(encoding="utf-8"))
        begin, end = f"<!-- BEGIN {name} (tools/model_zoo_bench.py) -->", f"<!-- END {name} -->"
        if begin in text and end in text:
            head, rest = text.split(begin, 1)
            _, tail = rest.split(end, 1)
            text = f"{head}{begin}\n{table(doc)}\n{end}{tail}"
    DOC.write_text(text, encoding="utf-8", newline="\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run one suite")
    run.add_argument("--suite", choices=sorted(SUITES), required=True)
    run.add_argument("--frames", type=int, default=0, help="timed frames per model (default: the suite's)")
    run.add_argument("--warmup", type=int, default=10)
    run.add_argument("--source", default=None, help="image, video or webcam index (default assets/bus.jpg)")
    run.add_argument("--models", nargs="*", default=None, help="override the suite's model list")
    run.add_argument("--ignition", default=None, help="Ignition checkout (default: a sibling 'Ignition' directory)")
    run.add_argument("--timeout", type=float, default=1800.0, help="seconds per model process")
    sub.add_parser("report", help="rewrite the generated tables from the suite JSON files")
    args = ap.parse_args()
    if args.cmd == "run":
        return run_suite(args)
    render_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

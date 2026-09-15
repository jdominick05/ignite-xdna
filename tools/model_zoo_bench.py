#!/usr/bin/env python3
"""Model-zoo latency suite: Ignition's `ignition suite` over a list of models, one JSON file per suite,
and the comparison tables in docs/MODEL_ZOO_BENCHMARKS.md.

    python tools/model_zoo_bench.py run --suite onnx-cpu                  # 300 frames per ONNX model on CPU
    python tools/model_zoo_bench.py run --suite npu --frames 500          # the .ignite containers on Device 0
    python tools/model_zoo_bench.py report                                # rewrite the tables from the JSON

`run` passes the models to `python -m ignition.cli suite` from an Ignition checkout (f9c85fb or later).
Ignition's suite runs every model in its own ignition.live process (--headless --frames N --json) and
writes one JSON record and one log per model, plus index.json, into a new directory
results/model_zoo/<suite>_<UTC time>/, so a run never overwrites an earlier run's records or logs. A
record holds the run summary, the return code, `git describe --dirty` of the Ignition and ignite-xdna
checkouts and, for an .ignite container, what `xrt-smi examine -r aie-partitions` said before and after:
a foreign hardware context makes the number contention, not a result. Checkout paths and the profile
directory are scrubbed.

Ignition comes from --ignition, else $IGNITION_ROOT, else a sibling Ignition checkout; ignite-xdna from
this checkout. Both src/ directories go first on PYTHONPATH, and the run stops before any model if the
interpreter imports either package from somewhere else, so the records credit the code that ran. Only
when every run is clean (return code 0, timed frames, an idle NPU before each container) does `run`
rewrite the suite's JSON file and the tables; otherwise it exits 1 and leaves both as they were.

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
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "MODEL_ZOO_BENCHMARKS.md"
LOG_DIR = ROOT / "results" / "model_zoo"

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

# Prints the directory each package loads its modules from (its __path__, so this repository's root forwarder,
# ignite_xdna/__init__.py, counts as src/ignite_xdna).
_WHERE = ("import json, ignition, ignite_xdna; "
          "print(json.dumps([list(ignition.__path__)[0], list(ignite_xdna.__path__)[0]]))")


def find_ignition(explicit: Optional[str]) -> Path:
    """The Ignition checkout whose src/ holds the suite runner: --ignition, else $IGNITION_ROOT, else a sibling."""
    if explicit:
        candidates = [Path(explicit)]
    else:
        candidates = [Path(os.environ["IGNITION_ROOT"])] if os.environ.get("IGNITION_ROOT") else []
        candidates += [p / "Ignition" for p in ROOT.parents]
    for c in candidates:
        if (c / "src" / "ignition" / "suite.py").is_file():
            return c.resolve()
    raise FileNotFoundError("no Ignition checkout with src/ignition/suite.py (`ignition suite`, Ignition f9c85fb "
                            "or later); pass --ignition")


def wrong_imports(env: Dict[str, str], ignition: Path) -> Optional[str]:
    """None when this interpreter, run with ``env``, imports ignition from ``ignition``/src and ignite_xdna from
    this checkout's src; otherwise what it imported instead."""
    res = subprocess.run([sys.executable, "-c", _WHERE], capture_output=True, text=True, env=env, cwd=str(ROOT),
                         timeout=300)
    if res.returncode != 0:
        return f"importing ignition and ignite_xdna failed:\n{res.stderr.strip()}"
    files = [Path(f).resolve() for f in json.loads(res.stdout.strip().splitlines()[-1])]
    wrong = [f"{name} imports from {file}, not from {src}"
             for name, file, src in zip(("ignition", "ignite_xdna"), files, (ignition / "src", ROOT / "src"))
             if not file.is_relative_to(src.resolve())]
    return "; ".join(wrong) or None


def rel(path: Path) -> str:
    """A path relative to this checkout with forward slashes, as the suite JSON files cite them."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def run_suite(args: argparse.Namespace) -> int:
    suite = SUITES[args.suite]
    ignition = find_ignition(args.ignition)
    models = [m if m.is_absolute() else ROOT / m for m in map(Path, args.models or suite["models"])]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ignition / "src"), str(ROOT / "src")]
                                        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    wrong = wrong_imports(env, ignition)
    if wrong:
        print(f"[suite] {wrong}", file=sys.stderr)
        return 2
    out_dir = LOG_DIR / f"{args.suite}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    cmd = [sys.executable, "-m", "ignition.cli", "suite", *map(str, models),
           "--source", args.source or str(ROOT / "assets" / "bus.jpg"), "--frames", str(args.frames or suite["frames"]),
           "--warmup", str(args.warmup), "--out", str(out_dir), "--timeout", str(args.timeout)]
    print(f"[suite] {args.suite}: {len(models)} models through Ignition's suite, records in {rel(out_dir)}", flush=True)
    rc = subprocess.run(cmd, env=env, cwd=str(ROOT)).returncode
    if not (out_dir / "index.json").is_file():
        print(f"[suite] ignition suite exited {rc} without writing records", file=sys.stderr)
        return rc or 2
    index = json.loads((out_dir / "index.json").read_text(encoding="utf-8"))
    if rc != 0 or not index["all_ok"]:
        print(f"[suite] not every run was clean (ignition suite exited {rc}): {rel(suite['json'])} and the tables "
              f"are unchanged, and the records stay in {rel(out_dir)}", file=sys.stderr)
        return 1
    runs = []
    for row in index["records"]:
        record = json.loads((out_dir / f"{row['name']}.json").read_text(encoding="utf-8"))
        record["record"] = rel(out_dir / f"{row['name']}.json")
        record["log"] = rel(out_dir / record["log"])
        runs.append(record)
    doc = {"suite": args.suite, "created_utc": index["created_utc"], "frames": index["frames"],
           "warmup": index["warmup"], "source": index["source"], "ignite_xdna_commit": index["ignite_xdna_commit"],
           "ignition_commit": index["ignition_commit"], "records": rel(out_dir),
           "all_returncodes_zero": all(r["returncode"] == 0 for r in runs), "runs": runs}
    suite["json"].write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"[suite] wrote {rel(suite['json'])}", flush=True)
    render_report()
    return 0


def _fmt(v: Any, digits: int = 2) -> str:
    return "n/a" if v is None or (isinstance(v, float) and v != v) else f"{v:.{digits}f}"


def _name(path: Any) -> str:
    """The last component of a recorded path, whichever slash it was written with."""
    return re.split(r"[\\/]", str(path))[-1]


def table(doc: Dict[str, Any]) -> str:
    rows = ["| Model | Task | Backend | Frames | G2G mean (ms) | P50 | P95 | P99 | FPS | Preprocess | Network | "
            "Post | Dispatch | Readback | RSS drift (MB) | rc |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in doc["runs"]:
        g, s = r.get("g2g_ms", {}), r.get("stages_ms", {})
        rows.append(f"| `{_name(r['model_path'])}` | {r.get('task', 'n/a')} | {r.get('backend', 'n/a')} | "
                    f"{r.get('frames_timed', 0)} | {_fmt(g.get('mean'))} | {_fmt(g.get('p50'))} | {_fmt(g.get('p95'))} | "
                    f"{_fmt(g.get('p99'))} | {_fmt(r.get('fps_from_mean'), 1)} | {_fmt(s.get('preprocess'))} | "
                    f"{_fmt(s.get('network'))} | {_fmt(s.get('postprocess'))} | {_fmt(s.get('dispatch'))} | "
                    f"{_fmt(s.get('readback'))} | {_fmt(r.get('rss_mb', {}).get('drift'))} | {r['returncode']} |")
    host = next((r.get("host") for r in doc["runs"] if r.get("host")), {}) or {}
    meta = (f"Suite `{doc['suite']}`, {doc['created_utc']}, {doc['frames']} timed frames after {doc['warmup']} warm-up, "
            f"source `{_name(doc['source'])}`, host `{host.get('hostname', '?')}`, onnxruntime "
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
    run = sub.add_parser("run", help="run one suite through Ignition's `ignition suite`")
    run.add_argument("--suite", choices=sorted(SUITES), required=True)
    run.add_argument("--frames", type=int, default=0, help="timed frames per model (default: the suite's)")
    run.add_argument("--warmup", type=int, default=10)
    run.add_argument("--source", default=None, help="image, video or webcam index (default assets/bus.jpg)")
    run.add_argument("--models", nargs="*", default=None, help="override the suite's model list")
    run.add_argument("--ignition", default=None,
                     help="Ignition checkout with `ignition suite` (default: $IGNITION_ROOT, then a sibling 'Ignition')")
    run.add_argument("--timeout", type=float, default=1800.0, help="seconds per model process")
    sub.add_parser("report", help="rewrite the generated tables from the suite JSON files")
    args = ap.parse_args()
    if args.cmd == "run":
        try:
            return run_suite(args)
        except FileNotFoundError as exc:
            print(f"[suite] {exc}", file=sys.stderr)
            return 2
    render_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

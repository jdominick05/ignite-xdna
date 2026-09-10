"""Split an Ignition AdaRound log's wall time into ORT extraction, torch training and the rest.

`python -m quant adaround` prints Quark's per-layer profiler lines ("finetuning layer N,
onnx inference ... time consumed Xs" and "... torch training time consumed Xs") and a
closing ADAROUND_COMPLETE record. Summing the first two per phase is what says how much of
a run a faster torch device could touch at all: ORT activation extraction stays on the cpu
whatever --device says, so only the torch column can move.

    python tools/adaround_log_times.py results/quant/quant_resnet50_ignition_cle_adaround_c64.log
    python tools/adaround_log_times.py --baseline <cpu arm log> <cpu arm log> <gpu arm log> ...

With --baseline, each log also gets its speedup over that one, for the torch phase alone
and for the whole run. That is only meaningful between logs from the same machine and the
same env -- the DEVICE_INFO lines, when the log has them, are printed so a reader can check.
Refuses a log whose layer lines are missing, duplicated, or disagree with its header.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HEADER = re.compile(r"^ADAROUND layers=(\d+) images=(\d+) .*?torch=(\S+) threads=(\d+) ort=(\S+)"
                    r"(?: optim_device=(\S+))?")
PHASE = re.compile(r"^Quark_latency_profiler: finetuning layer (\d+), "
                   r"(onnx inference|torch training)\b.*time consumed ([0-9.]+)s")
COMPLETE = re.compile(r"^ADAROUND_COMPLETE (\{.*\})\s*$")
INFO_KEYS = ("hostname", "cpu", "conda_env", "torch_hip", "cuda_device_0", "omp_num_threads")


def parse(path: Path) -> dict:
    header = None
    phases: dict[str, dict[int, float]] = {"onnx inference": {}, "torch training": {}}
    complete = None
    info: dict[str, str] = {}
    device_line = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if m := HEADER.match(line):
            header = {"layers": int(m[1]), "images": int(m[2]), "torch": m[3], "threads": int(m[4]),
                      "ort": m[5], "optim_device": m[6] or "cpu (header predates optim_device)"}
        elif m := PHASE.match(line):
            layer, phase = int(m[1]), m[2]
            if layer in phases[phase]:
                raise ValueError(f"{path}: layer {layer} has two '{phase}' lines")
            phases[phase][layer] = float(m[3])
        elif m := COMPLETE.match(line):
            complete = json.loads(m[1])
        elif line.startswith("DEVICE_INFO "):
            key, _, value = line[len("DEVICE_INFO "):].partition("=")
            info[key] = value
        elif line.startswith("ADAROUND DEVICE "):
            device_line = line[len("ADAROUND DEVICE "):]
    if header is None or complete is None:
        raise ValueError(f"{path}: no ADAROUND header or no ADAROUND_COMPLETE record -- an incomplete run?")
    for phase, per_layer in phases.items():
        if sorted(per_layer) != list(range(header["layers"])):
            raise ValueError(f"{path}: '{phase}' lines cover {len(per_layer)} layers, header says "
                             f"{header['layers']}")
    onnx_s, torch_s = sum(phases["onnx inference"].values()), sum(phases["torch training"].values())
    wall = float(complete["wall_seconds"])
    return {"log": str(path), "header": header, "info": info, "device_line": device_line,
            "onnx_s": onnx_s, "torch_s": torch_s, "other_s": wall - onnx_s - torch_s, "wall_s": wall,
            "peak_rss_bytes": complete.get("peak_rss_bytes"), "output_sha256": complete.get("output_sha256"),
            "changed_elements": complete.get("changed_elements")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--baseline", type=Path, help="log to compute speedups against (typically the cpu arm)")
    args = ap.parse_args(argv)
    rows = [parse(p) for p in args.logs]
    base = parse(args.baseline) if args.baseline else None
    for r in rows:
        h = r["header"]
        print(f"== {r['log']}")
        for key in INFO_KEYS:
            if key in r["info"]:
                print(f"   {key:<16} {r['info'][key]}")
        print(f"   {'optim_device':<16} {h['optim_device']}   torch={h['torch']} threads={h['threads']} ort={h['ort']}")
        if r["device_line"]:
            print(f"   {'adaround_device':<16} {r['device_line']}")
        print(f"   layers={h['layers']} images={h['images']}")
        print(f"   onnx inference (ORT extraction, cpu) : {r['onnx_s']:9.2f} s")
        print(f"   torch training                       : {r['torch_s']:9.2f} s")
        print(f"   everything else                      : {r['other_s']:9.2f} s")
        print(f"   wall_seconds                         : {r['wall_s']:9.2f} s")
        print(f"   peak_rss_bytes                       : {r['peak_rss_bytes']}")
        print(f"   changed_elements                     : {r['changed_elements']}")
        print(f"   output_sha256                        : {r['output_sha256']}")
        if base is not None:
            print(f"   speedup vs {Path(base['log']).name}: torch phase {base['torch_s'] / r['torch_s']:.2f}x, "
                  f"whole run {base['wall_s'] / r['wall_s']:.2f}x")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

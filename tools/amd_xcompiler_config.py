#!/usr/bin/env python3
"""Write a VitisAI EP config_file that is the EP's own config from a compiled cache, with xcompiler
attributes overridden.

    python -I tools/amd_xcompiler_config.py --context <cache>/context.json --out cfg.json \\
        [--set enable_weights_prefetch=bool:true] [--set prefetch_lp=uint:4] ...

The EP records the config it compiled with in context.json ("config"). This copies its passes,
targets, target and mepTable - not the session-specific fields (cacheDir, cacheKey, sessionOptions,
onnxPath, the md5s, version) - and applies each --set to the xcompilerAttrs of both places the EP
keeps them: the X1 target's targetOpts and the fuse_DPU pass's passDpuParam. Pass the result as the
provider option config_file (tools/amd_vitisai_yolo.py --config-file).

Whether an override reached the compiler is NOT shown by the session building: read the compiled
xmodel's root attributes (tools/amd_xmodel_census.py --root-attrs) and check the value moved. With no
--set, the output is the unchanged config, which is the control for the config_file path itself.
Used on YOLOv8s only so far (Desktop 2, 2026-10-13).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

KINDS = {"bool": ("boolValue", lambda s: {"true": True, "false": False}[s.lower()]),
         "uint": ("uintValue", int), "int": ("intValue", int), "string": ("stringValue", str)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--context", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target", default="X1")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=KIND:VALUE")
    a = ap.parse_args()
    c = json.loads(a.context.read_text(encoding="utf-8"))["config"]
    cfg = {k: c[k] for k in ("passes", "targets", "target", "mepTable") if k in c}
    t = next(x for x in cfg["targets"] if x["name"] == a.target)
    sinks = [t.setdefault("targetOpts", {}).setdefault("xcompilerAttrs", {})]
    for p in cfg["passes"]:
        if "passDpuParam" in p:
            sinks.append(p["passDpuParam"].setdefault("xcompilerAttrs", {}))
    for spec in a.set:
        name, _, kv = spec.partition("=")
        kind, _, val = kv.partition(":")
        field, conv = KINDS[kind]
        for s in sinks:
            s[name] = {field: conv(val)}
        print(f"SET {name} = {{{field}: {conv(val)!r}}} in {len(sinks)} places")
    a.out.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    print(f"WROTE {a.out.name} ({a.out.stat().st_size} B), target {cfg['target']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

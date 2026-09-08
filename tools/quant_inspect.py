#!/usr/bin/env python3
"""Fingerprint local ONNX models without running them or importing Quark.

Activate resnet_env or resnet_env17 first. Prints JSON including artifact hashes.
Example: python tools/quant_inspect.py models/resnet50_xint8_c64.onnx
This reports the export contract and describes QDQ; it enforces neither, and says
nothing about EP acceptance or parity. Quantization keeps the strict loader.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant.graph import Graph
from onnx.checker import ValidationError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path, nargs="+")
    args = parser.parse_args()
    reports = []
    for path in args.models:
        try:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            graph = Graph.load(path, strict=False)
            fingerprint = graph.fingerprint()
        except (OSError, ValueError, RuntimeError, ValidationError) as exc:
            parser.exit(1, f"{path.name}: {exc}\n")
        reports.append({"model": path.name, "sha256": digest,
                        "export_contract": graph.contract_error or "ok",
                        "onnx_checker": graph.checker_error or "ok",
                        "node_order_changed_in_memory": graph.node_order_changed,
                        "fingerprint": fingerprint})
    print(json.dumps({
        "method": "ONNX static inspection; contract violations reported, not enforced; no model execution or EP acceptance check",
        "utc": datetime.now(timezone.utc).isoformat(),
        "environment": {p: metadata.version(p) for p in ("numpy", "onnx")},
        "models": reports,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Compare an owned model with a reference; emit a reviewable JSON gate result."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant.graph import Graph
from quant.qdq import read_pos_table
from quant.quantize import file_hash
from quant.refine import refine
from quant.verify import graph_diff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    owned, reference = Graph.load(args.model), Graph.load(args.reference)
    sidecar = Path(str(args.model) + ".quant.json")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    if provenance["output_sha256"] != file_hash(args.model):
        raise ValueError("Sidecar hash does not match model")
    result = graph_diff(owned, reference)
    print("PRODUCER_REPORT_FROM_SIDECAR", json.dumps({k: v for k, v in provenance.items() if k not in ("positions", "calibration")}, indent=2))
    print("REFERENCE_SHA256", file_hash(args.reference))
    oq, rq = read_pos_table(owned), read_pos_table(reference)
    pos_delta = {name: [oq[name].pos if name in oq else None, rq[name].pos if name in rq else None]
                 for name in oq.keys() | rq.keys()
                 if name not in oq or name not in rq or oq[name].pos != rq[name].pos}
    print("POSITION_DELTA", json.dumps(pos_delta, sort_keys=True))
    provenance_pass = True
    if "calibration" in provenance:
        oracle = json.loads(args.reference.with_suffix(".reference.json").read_text(encoding="utf-8"))
        provenance_pass = (oracle["listing"] == provenance["calibration"]["listing"] and
                           oracle["preprocess"] == provenance["preprocess"] and
                           oracle["float_model_sha256"] == provenance["input_sha256"] and
                           oracle["model_sha256"] == file_hash(args.reference) and
                           oracle["include_cle"] == provenance["cle"])
        print("SAME_FLOAT_PREPROCESS_LISTING_CLE", provenance_pass)
        print("CALIBRATION_STATS", json.dumps(provenance["calibration"], indent=2))
    print("GRAPH_DIFF", json.dumps(asdict(result), indent=2))
    checks = {name: asdict(refine(graph)) for name, graph in (("own", owned), ("reference", reference))}
    print("REFINE_FIXED_POINT", json.dumps(checks, indent=2))
    passed = (result.ok() and not pos_delta and provenance_pass and
              all(not c["log"] and c["converged"] for c in checks.values()))
    print("GRAPH_DIFF_PASS", passed)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

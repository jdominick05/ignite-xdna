"""Compare an owned model with a reference; emit a reviewable JSON gate result.

GRAPH_DIFF_PASS keeps the alpha's tolerance (scales/zero points exact, int8 within 1 LSB).
INT8_EXACT is the bitwise verdict on every <w>_quantized initializer; for an AdaRound
artifact it is the parity result, since AdaRound moves weights by exactly one LSB.
"""
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

ADAROUND_KEYS = ("DataSize", "FixedSeed", "BatchSize", "NumIterations", "LearningRate",
                 "OptimAlgorithm", "OptimDevice", "InferDevice", "EarlyStop")


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
        adaround = provenance.get("adaround")
        if adaround is not None or oracle.get("include_fast_ft", False):
            own = None if adaround is None else {k: adaround["config"].get(k) for k in ADAROUND_KEYS}
            reference_ft = oracle.get("fast_finetune")
            adaround_pass = (own is not None and reference_ft is not None and oracle.get("include_fast_ft", False)
                             and all(own[k] == reference_ft.get(k) for k in ADAROUND_KEYS))
            print("ADAROUND_PROVENANCE", json.dumps({"own": own, "reference": reference_ft, "pass": adaround_pass}))
            if adaround is not None:
                print("ADAROUND_SUMMARY", json.dumps({k: v for k, v in adaround.items() if k != "layers"}, indent=2))
            provenance_pass = provenance_pass and adaround_pass
        print("CALIBRATION_STATS", json.dumps(provenance["calibration"], indent=2))
    print("GRAPH_DIFF", json.dumps(asdict(result), indent=2))
    print("WEIGHT_DIFF_SUMMARY", json.dumps({
        "int8_initializers_compared": result.compared_int8_initializers,
        "initializers_differing": len(result.weight_lsb),
        "elements_differing": sum(result.weight_changed_elements.values()),
        "max_lsb": max(result.weight_lsb.values(), default=0)}))
    print("INT8_EXACT", not result.weight_lsb)
    checks = {name: asdict(refine(graph)) for name, graph in (("own", owned), ("reference", reference))}
    print("REFINE_FIXED_POINT", json.dumps(checks, indent=2))
    passed = (result.ok() and not pos_delta and provenance_pass and
              all(not c["log"] and c["converged"] for c in checks.values()))
    print("GRAPH_DIFF_PASS", passed)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

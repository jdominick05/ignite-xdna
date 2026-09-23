"""Compare two .ignite containers: which schedule quantities and which blobs are the same, and which are not.

WHY THIS EXISTS. Two containers compiled from models that differ only in their weight VALUES - a plain
XINT8 model and its AdaRound twin, say - should carry the same schedule: the same instruction stream,
the same packets and rounds, the same workspace. If they do, a latency difference between them is not
the schedule's, and a quality difference is the weights'. That is an assumption until it is compared,
and this compares it.

What is compared:
* the engine, the task and the top-level shapes and byte counts the runtime sizes itself by;
* every graph-engine quantity a schedule or a packer could move (the same list
  ``tools/rebuild_container_check.py`` compares, imported from it so the two cannot drift), plus every
  placement's geometry, key by key;
* each blob's sha256, by name.

A placement's ``scale`` and ``zero_point``, the manifest's ``quant_scales`` and the dense output's
quantization are the model's calibration, not its schedule: a weights-only twin calibrated on its own
may carry different ones. They are reported, never failed.

THE EMBEDDED XCLBIN IS NOT BYTE-STABLE ACROSS BUILDS, even of one model: it carries a creation timestamp
and a random UUID, in its header and again in its metadata, and a few more bytes near its AIE image
move from build to build. So a differing ``engine.xclbin`` says nothing on its own. The kernel object's
hash is compared as a schedule quantity instead, and ``--rebuild-of-a`` takes a fresh rebuild of the
first container (``tools/rebuild_container_check.py --out``) and checks that every region where the
second container's xclbin differs from the first's is a region where that rebuild differs too: build
identity, not content.

The schedule quantities, the placements and the blobs named by ``--expect-same-blob`` (default: the
instruction stream) must match, or the exit status is 1. Every other blob is reported, same or not,
and never fails the comparison: for a weights-only twin the weight packets are SUPPOSED to differ.

    python tools/container_diff.py build/sesr_m7.ignite build/sesr_m7_adaround.ignite

WHAT IT IS NOT. It reads two manifests and hashes their blobs. It compiles nothing and opens no
device, and a match says the two schedules are the same, not that either is correct.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ignite_xdna.compiler.serializer import IgniteModelReader  # noqa: E402
from rebuild_container_check import COMPARED, TOP_COMPARED  # noqa: E402

QUANT_KEYS = ("scale", "zero_point")


def geometry(p: dict) -> dict:
    return {k: v for k, v in p.items() if k not in QUANT_KEYS}


def calibration(m: dict) -> dict:
    ge = m.get("graph_engine", {})
    dense = m.get("dense_output") or {}
    return {"quant_scales": m.get("quant_scales"),
            "dense_output": {k: dense.get(k) for k in QUANT_KEYS},
            "placements": {n: {k: p.get(k) for k in QUANT_KEYS} for n, p in sorted(ge.get("placements", {}).items())}}


def facts(path: Path) -> dict:
    with IgniteModelReader(str(path)) as r:
        m = r.manifest
        raw = {name: r.get_blob_bytes(name) for name in sorted(r.blobs)}
    return {"manifest": m, "blobs": {n: hashlib.sha256(b).hexdigest()[:16] for n, b in raw.items()},
            "xclbin": raw.get("engine.xclbin", b"")}


def kernel_object(m: dict) -> str:
    return m.get("kernel_object_sha256") or m.get("graph_engine", {}).get("kernel_object_sha256") or ""


def regions(x: bytes, y: bytes, gap: int = 8) -> list:
    """[start, end) byte regions where x and y differ, runs closer than ``gap`` merged (a byte can agree
    by chance inside a differing field, so exact runs are not stable from one pair of builds to the next)."""
    if len(x) != len(y):
        return [(0, max(len(x), len(y)))]
    out, i, n = [], 0, len(x)
    while i < n:
        if x[i] == y[i]:
            i += 1
            continue
        j = i
        while j < n and x[j] != y[j]:
            j += 1
        if out and i - out[-1][1] < gap:
            out[-1] = (out[-1][0], j)
        else:
            out.append((i, j))
        i = j
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path, help="the reference container")
    ap.add_argument("b", type=Path, help="the container compared with it")
    ap.add_argument("--expect-same-blob", action="append", default=None,
                    help="a blob that must be byte-identical (default: insts.bin); repeat for more")
    ap.add_argument("--rebuild-of-a", type=Path, default=None,
                    help="a fresh rebuild of A from the same model: B's xclbin may differ from A's only where it does")
    args = ap.parse_args()
    must_blobs = args.expect_same_blob or ["insts.bin"]

    fa, fb = facts(args.a), facts(args.b)
    ma, mb = fa["manifest"], fb["manifest"]
    ga, gb = ma.get("graph_engine", {}), mb.get("graph_engine", {})

    rows = [("engine", ma.get("engine"), mb.get("engine")),
            ("kernel_object_sha256", kernel_object(ma), kernel_object(mb))]
    rows += [(k, ma.get(k), mb.get(k)) for k in TOP_COMPARED]
    rows += [(f"graph_engine.{k}", ga.get(k), gb.get(k)) for k in COMPARED]
    pa, pb = ga.get("placements", {}), gb.get("placements", {})
    rows.append(("placement names", sorted(pa), sorted(pb)))
    rows += [(f"placement {n}", geometry(pa[n]), geometry(pb[n])) for n in sorted(set(pa) & set(pb))]
    schedule_differ = [k for k, x, y in rows if x != y]
    ca, cb = calibration(ma), calibration(mb)
    calib_differ = sorted(
        [k for k in ("quant_scales", "dense_output") if ca[k] != cb[k]]
        + [f"placement {n}" for n in ca["placements"] if ca["placements"][n] != cb["placements"].get(n)])

    blobs = {}
    for name in sorted(set(fa["blobs"]) | set(fb["blobs"])):
        x, y = fa["blobs"].get(name), fb["blobs"].get(name)
        blobs[name] = {"a": x, "b": y, "same": x == y}
    blob_fail = [n for n in must_blobs if not blobs.get(n, {}).get("same")]

    xclbin = {"regions_a_b": regions(fa["xclbin"], fb["xclbin"])}
    if args.rebuild_of_a is not None:
        fr = facts(args.rebuild_of_a)
        if fr["manifest"].get("source_model_sha256") != ma.get("source_model_sha256"):
            raise SystemExit(f"{args.rebuild_of_a} was not built from A's model")
        noise = regions(fa["xclbin"], fr["xclbin"])
        # Contained, not merely overlapping: a whole-file difference overlaps every noise region. The
        # slack absorbs a byte at a field's edge that happens to agree in one pair of builds.
        slack = 8
        unexplained = [r for r in xclbin["regions_a_b"]
                       if not any(s - slack <= r[0] and r[1] <= e + slack for s, e in noise)]
        xclbin.update(regions_a_rebuild=noise, regions_not_in_rebuild_noise=unexplained,
                      rebuild_kernel_object_same=kernel_object(fr["manifest"]) == kernel_object(ma))
        if unexplained:
            blob_fail.append("engine.xclbin (beyond rebuild noise)")

    print("CONTAINER_DIFF " + json.dumps({
        "a": {"path": args.a.as_posix(), "model_name": ma.get("model_name"),
              "source_model_sha256": (ma.get("source_model_sha256") or "")[:16]},
        "b": {"path": args.b.as_posix(), "model_name": mb.get("model_name"),
              "source_model_sha256": (mb.get("source_model_sha256") or "")[:16]},
        "schedule_quantities_compared": len(rows), "schedule_quantities_differ": schedule_differ,
        "schedule": {k: x for k, x, _ in rows if k.startswith("graph_engine.") or k in TOP_COMPARED or k == "engine"},
        "calibration_differs": calib_differ, "xclbin": xclbin,
        "blobs": blobs, "expect_same_blobs": must_blobs}, sort_keys=True))
    for k, x, y in rows:
        if x != y:
            print(f"  DIFFERS {k}: {json.dumps(x)[:200]} -> {json.dumps(y)[:200]}")
    for k in calib_differ:
        print(f"  calibration differs (reported, not failed): {k}")
    for name, b in blobs.items():
        print(f"  blob {name}: {'same' if b['same'] else 'differs'} ({b['a']} / {b['b']})")
    print(f"  xclbin regions where B differs from A: {xclbin['regions_a_b']}")
    if "regions_a_rebuild" in xclbin:
        print(f"  xclbin regions where a rebuild of A differs from A: {xclbin['regions_a_rebuild']}")
        print(f"  B's regions not explained by rebuild noise: {xclbin['regions_not_in_rebuild_noise']}")
    if schedule_differ or blob_fail:
        print(f"FAIL: {len(schedule_differ)} schedule quantities differ; blobs expected identical and not: {blob_fail}")
        return 1
    print(f"PASS: the same schedule on all {len(rows)} compared quantities, and {', '.join(must_blobs)} identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Score accumulate models offline wherever a sitting already found the device equal to the emulator.

WHY. A sitting that found the device byte-equal to the emulator under the model in force (R) fixes
the device's bits at every value it compared: they are R's bits. A candidate C then stands or falls
on that sitting's data with no device. Wherever C's bits differ from R's at a compared value, C
disagrees with the silicon there and is refuted. Everywhere else it is consistent with it. That is a
deduction from a logged equality, not a new measurement. It is only as good as the rebuild of the
sitting's inputs, so every mode checks that rebuild against the sitting's own record before it
counts anything, and refuses when the check fails.

MODES
  sesr     bf16_sr_silicon_check.py's exactness frames: its images through npu.sesr.preprocess, then
           its seeded inputs. Each frame runs through the whole network (graph_reference_bf16.
           run_direct) under every model. Scored: the tail and every tensor resident at frame end,
           which is what the sitting compared. Each --silicon-log must name this container and QDQ
           (by hash), these images and seeds, and R. Its per-frame resident value count must equal
           the one rebuilt here. Only the frames it found exact under R are scored. The tensors a
           reuse container overwrites are reported too, as not compared on silicon.
  sweep    kernels/bf16_conv/engine_bf16.py's --sweep cases (every shape, the psum chains, the
           residual pairs), from engine_bf16.sweep_cases. Each --silicon-log case must be one of
           them and logged byte-equal under R. The log's counts for the models it scored must
           equal the rebuilt ones.
  sixteen  kernels/bf16_conv/engine_bf16_16core.py's plan for --seed (--integer A W B for the
           integer arm), drained as its emulate_plan drains it. The rebuild under R must equal the
           sitting's own --expected file (its build directory's expected.npy) byte for byte, and
           every column of every iteration in each --silicon-log must be byte-equal.
  where    No sitting: a prediction. The sesr frames through the whole network under every model,
           and every value of every tensor (padding lanes flagged) where a model's bits part from
           R's, with both bit patterns. Against a container whose every tensor is resident
           (--no-workspace-reuse), a later sitting can check it value by value, so commit its log
           before that sitting runs.

A frame or case the sitting did NOT find exact under R is not scored: the device's bits are
unknown there. A dump replay (bf16_sr_silicon_check.py --from-dump --mac-model C) answers those.
A sitting that leaves nothing to score under R is refused, never reported as consistent. So is a sweep
log that never scored R at all.

R defaults to the model in force, which has been exp_sum since 2026-09-23. Every rescore log
committed before then was made under aligned, the default at the time. Reproduce one with
--reference aligned. --models takes a comma list (--models wide,exp_sum), so it cannot swallow the
mode after it.

    bash scripts/research-lowlevel.sh --log results/aie/<name>.log --checks-only -- \\
        bash scripts/research-iron.sh tools/bf16_mac_model_rescore.py sesr \\
        --container build/<c>.ignite --qdq models/<q>.onnx --silicon-log results/aie/<sitting>.log
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src", ROOT / "tools", ROOT / "kernels" / "bf16_conv"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ignite_xdna.compiler import engine_bf16_emulator as em  # noqa: E402


def emit(tag: str, obj) -> None:
    print(f"BF16_MAC_RESCORE_{tag} " + json.dumps(obj, sort_keys=True), flush=True)


def tagged(path: Path, tag: str) -> list:
    """Every JSON line of a log carrying `tag` as its first word."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        head, _, body = line.partition(" ")
        if head == tag:
            rows.append(json.loads(body))
    return rows


def refuse_if_empty(log, scored: dict, ref: str) -> None:
    """A sitting with nothing exact under the reference fixes no bits. Scoring it would report every
    model consistent on zero values: a silent empty pass."""
    if not any(scored.values()):
        raise SystemExit(f"{log}: nothing in this sitting was exact under {ref!r}, so nothing can be scored; "
                         "check --reference")


def verdict(scored: dict, differ: dict) -> dict:
    """Per model: values scored, how many disagree with the silicon, and what that makes it."""
    return {m: {"scored": scored[m], "differ_from_silicon": differ[m],
                "verdict": "refuted" if differ[m] else ("consistent" if scored[m] else "not scored")}
            for m in differ}


# --------------------------------------------------------------------------------------------- sesr
def sesr_frames(chk, cv2, images, seeds, qdq) -> list:
    """[(label, input)] exactly as bf16_sr_silicon_check.py stages its exactness frames: each image
    through npu.sesr.preprocess, then each seeded input rounded to bf16."""
    plan = [(Path(p).stem, np.asarray(chk.sesr.preprocess(cv2.imread(str(p)), chk.TILE)[0][0], np.float32))
            for p in images]
    plan += [(f"seed{k}", em.to_bf16(chk.wo.seeded_input(qdq, k)[0])) for k in seeds]
    for label, x in plan:
        if not np.array_equal(em.to_bf16(x), x):
            raise SystemExit(f"{label}: the rebuilt input is not bf16-exact, so it is not what was staged")
    return plan


def where_mode(args, ref: str, models: list) -> int:
    """Every value, in every tensor of the network, where a model's bits part from the reference's on
    the exactness frames. With a container whose every tensor is resident (--no-workspace-reuse),
    this is a prediction a sitting can check value by value; commit it before that sitting runs."""
    import cv2  # noqa: PLC0415
    import bf16_sr_silicon_check as chk  # noqa: PLC0415 - read-only: its frames and hashes
    from ignite_xdna.compiler import graph_ir  # noqa: PLC0415
    from ignite_xdna.compiler import graph_reference_bf16 as gr16  # noqa: PLC0415

    images = args.image or chk.DEFAULT_IMAGES
    seeds = [0] if args.seed is None else args.seed
    ir = graph_ir.lower_yolov8n(str(args.qdq))
    emit("SETUP", {"mode": "where", "qdq": chk.rel(args.qdq), "qdq_sha256": chk.sha16(args.qdq), "reference": ref,
                   "models": models, "images": [chk.rel(p) for p in images], "seeds": seeds,
                   "tensors": {n: t.channels for n, t in ir.tensors.items()}})
    totals = {m: 0 for m in models}
    for label, x in sesr_frames(chk, cv2, images, seeds, args.qdq):
        outs = {m: gr16.run_direct(ir, x, mac_model=m) for m in [ref] + models}
        bits = {m: {n: em.bf16_bits(t) for n, t in o.items()} for m, o in outs.items()}
        count = {m: {} for m in models}
        for m in models:
            for n in bits[ref]:
                real = ir.tensors[n].channels
                for c, y, xx in zip(*np.nonzero(bits[m][n] != bits[ref][n])):
                    emit("WHERE", {"frame": label, "model": m, "tensor": n, "c": int(c), "y": int(y), "x": int(xx),
                                   "padding_lane": bool(c >= real),
                                   "bits": {ref: f"{int(bits[ref][n][c, y, xx]):04x}",
                                            m: f"{int(bits[m][n][c, y, xx]):04x}"}})
                    count[m][n] = count[m].get(n, 0) + 1
            totals[m] += sum(count[m].values())
        emit("WHERE_FRAME", {"frame": label, "differ_by_model": count})
    emit("WHERE_TOTAL", {"differ_by_model": totals})
    return 0


def sesr_mode(args, ref: str, models: list) -> int:
    import cv2  # noqa: PLC0415 - only this mode needs them
    import bf16_sr_silicon_check as chk  # noqa: PLC0415 - read-only: its frames, survivors and hashes
    from ignite_xdna.compiler import graph_ir  # noqa: PLC0415
    from ignite_xdna.compiler import graph_reference_bf16 as gr16  # noqa: PLC0415
    from ignite_xdna.compiler.serializer import IgniteModelReader  # noqa: PLC0415

    images = args.image or chk.DEFAULT_IMAGES
    seeds = [0] if args.seed is None else args.seed
    facts = chk.container_facts(args.container)
    qdq_sha = chk.sha16(args.qdq)
    logs = []
    for log in args.silicon_log:
        setup = tagged(log, "BF16_SR_SETUP")
        if len(setup) != 1:
            raise SystemExit(f"{log}: expected one BF16_SR_SETUP line, found {len(setup)}")
        s = setup[0]
        want = {"container": facts["sha256"], "qdq": qdq_sha, "model": ref}
        got = {"container": s["bf16"]["sha256"], "qdq": s["qdq"]["sha256"], "model": s["mac_model"]}
        if got != want:
            raise SystemExit(f"{log}: the sitting ran {got}, this rescore rebuilds {want}")
        log_images = [Path(p).stem for p in s["images"]]
        missing = [n for n in log_images if n not in {Path(p).stem for p in images}] + \
                  [k for k in s["seeds"] if k not in seeds]
        if missing:
            raise SystemExit(f"{log}: frames {missing} are not rebuilt here; pass their --image / --seed")
        frames = {r["label"]: r for r in tagged(log, "BF16_SR_EXACT") if "repeat_of" not in r}
        logs.append((log, frames))

    ir = graph_ir.lower_yolov8n(str(args.qdq))
    with IgniteModelReader(str(args.container)) as r:
        manifest = r.manifest
    resident = sorted(chk.survivors(manifest["graph_engine"]))
    tail = manifest["dense_output"]["tensor"]
    # What the sitting read, exactly: a resident tensor at its logical channels (read_tensor trims the
    # block padding), and the tail at its full width, padding lanes included.
    channels = {n: int(manifest["graph_engine"]["placements"][n]["channels"]) for n in resident}
    emit("SETUP", {"mode": "sesr", "container": chk.rel(args.container), "container_sha256": facts["sha256"],
                   "qdq": chk.rel(args.qdq), "qdq_sha256": qdq_sha, "reference": ref, "models": models,
                   "images": [chk.rel(p) for p in images], "seeds": seeds, "resident": resident, "tail": tail,
                   "silicon_logs": [chk.rel(p) for p, _ in logs]})

    per_frame = {}
    for label, x in sesr_frames(chk, cv2, images, seeds, args.qdq):
        outs = {m: gr16.run_direct(ir, x, mac_model=m) for m in [ref] + models}
        bits = {m: {n: em.bf16_bits(t) for n, t in o.items()} for m, o in outs.items()}

        def seen(m):
            """The values the sitting compared: each resident tensor as read, plus the tail's padding
            lanes (its logical channels are already a resident tensor when the tail is one)."""
            parts = {n: bits[m][n][:channels[n]] for n in resident}
            parts["tail" if tail not in resident else "tail padding lanes"] = \
                bits[m][tail] if tail not in resident else bits[m][tail][channels[tail]:]
            return parts

        base = seen(ref)
        row = {"label": label, "compared_values": int(sum(bits[ref][n][:channels[n]].size for n in resident)),
               "tail_values": int(bits[ref][tail].size), "scored_values": int(sum(v.size for v in base.values())),
               "differ_from_reference": {}, "not_compared_differ": {}}
        for m in models:
            got = seen(m)
            row["differ_from_reference"][m] = {n: int(np.count_nonzero(got[n] != base[n])) for n in base}
            everywhere = int(sum(np.count_nonzero(bits[m][n] != bits[ref][n]) for n in bits[ref]))
            row["not_compared_differ"][m] = everywhere - sum(row["differ_from_reference"][m].values())
        per_frame[label] = row
        emit("FRAME", row)

    rc = 0
    for log, frames in logs:
        scored, differ, used = {m: 0 for m in models}, {m: 0 for m in models}, []
        for label, rep in frames.items():
            row = per_frame[label]
            if rep["resident_tensors"]["values"] != row["compared_values"] or rep["tail"]["values"] != row["tail_values"]:
                raise SystemExit(f"{log} {label}: the sitting compared {rep['resident_tensors']['values']} resident and "
                                 f"{rep['tail']['values']} tail values, the rebuild {row['compared_values']} and "
                                 f"{row['tail_values']}: not the same tensors")
            if not rep["exact"]:
                continue
            used.append(label)
            for m in models:
                scored[m] += row["scored_values"]
                differ[m] += sum(row["differ_from_reference"][m].values())
        refuse_if_empty(log, scored, ref)
        emit("SITTING", {"log": log.name, "frames_exact_under_reference": used,
                         "frames_not_scored": sorted(set(frames) - set(used)), "by_model": verdict(scored, differ)})
        rc |= int(any(differ.values()))
    return rc


# -------------------------------------------------------------------------------------------- sweep
def sweep_mode(args, ref: str, models: list) -> int:
    import engine_bf16 as eh  # noqa: PLC0415 - imports aie.iron; no device is opened

    cases = dict(eh.sweep_cases(args.seed))
    bits = {}
    for label, pkts in cases.items():
        b = {m: eh.expect(pkts, m) for m in [ref] + models}
        bits[label] = b
        emit("CASE", {"case": label, "elements": int(b[ref].size),
                      "differ_from_reference": {m: int(np.count_nonzero(b[m] != b[ref])) for m in models}})
    emit("SETUP", {"mode": "sweep", "seed": args.seed, "reference": ref, "models": models, "cases": list(cases),
                   "silicon_logs": [p.name for p in args.silicon_log]})
    rc = 0
    for log in args.silicon_log:
        rows = [r for r in tagged(log, "ENGINE_BF16") if "case" in r]
        if not rows:
            raise SystemExit(f"{log}: no ENGINE_BF16 case lines")
        scored, differ = {m: 0 for m in models}, {m: 0 for m in models}
        for r in rows:
            if r["case"] not in cases:
                raise SystemExit(f"{log}: case {r['case']!r} is not one engine_bf16.sweep_cases builds")
            b = bits[r["case"]]
            logged = r["bytes_equal_by_model"]
            if ref not in logged:
                raise SystemExit(f"{log} {r['case']}: the sitting scored {sorted(logged)}, not the reference "
                                 f"{ref!r}; pass --reference with one it scored")
            if logged[ref] != r["elements"]:
                continue
            # the rebuild must reproduce every count the sitting logged for the models it scored
            for m, n in logged.items():
                if int(np.count_nonzero(eh.expect(cases[r["case"]], m) == b[ref])) != n:
                    raise SystemExit(f"{log} {r['case']}: logged {m} = {n}; the rebuild disagrees")
            for m in models:
                scored[m] += int(b[ref].size)
                differ[m] += int(np.count_nonzero(b[m] != b[ref]))
        refuse_if_empty(log, scored, ref)
        emit("SITTING", {"log": log.name, "cases": len(rows), "by_model": verdict(scored, differ)})
        rc |= int(any(differ.values()))
    return rc


# ------------------------------------------------------------------------------------------ sixteen
def sixteen_mode(args, ref: str, models: list) -> int:
    import engine_bf16_16core as e16  # noqa: PLC0415 - ml_dtypes only; no device is opened

    ints = tuple(args.integer) if args.integer else None
    plan = e16.build_plan(args.seed, ints)
    flat = {}
    for m in [ref] + models:
        cols = e16.emulate_plan(plan, mac_model=m)
        flat[m] = np.concatenate([np.concatenate(c) if c else np.zeros(0, np.uint8) for c in cols]).view(np.uint16)
    saved = np.load(args.expected).view(np.uint16)
    rebuilt_equals_saved = bool(np.array_equal(flat[ref], saved))
    emit("SETUP", {"mode": "sixteen", "seed": args.seed, "integer": ints, "reference": ref, "models": models,
                   "expected": args.expected.as_posix(), "elements": int(saved.size),
                   "rebuilt_reference_equals_expected": rebuilt_equals_saved,
                   "silicon_logs": [p.name for p in args.silicon_log]})
    if not rebuilt_equals_saved:
        raise SystemExit(f"the rebuilt plan under {ref!r} is not the sitting's {args.expected}; not the same packets")
    rc = 0
    for log in args.silicon_log:
        rows = tagged(log, "ENGINE_BF16_16CORE")
        exact = bool(rows) and all(r["bytes_equal"] == r["bytes"] for r in rows)
        per_col = {int(r["bytes"]) for r in rows}
        if per_col != {saved.nbytes // e16.COLS}:
            raise SystemExit(f"{log}: columns carry {sorted(per_col)} B, the expected file {saved.nbytes // e16.COLS} B")
        scored = {m: int(saved.size) if exact else 0 for m in models}
        differ = {m: int(np.count_nonzero(flat[m] != saved)) if exact else 0 for m in models}
        refuse_if_empty(log, scored, ref)
        emit("SITTING", {"log": log.name, "column_iterations": len(rows), "every_column_byte_equal": exact,
                         "by_model": verdict(scored, differ)})
        rc |= int(any(differ.values()))
    return rc


def model_list(text: str) -> list:
    """--models a,b. A comma list rather than nargs="+", which swallowed the mode that follows it."""
    names = [s for s in text.split(",") if s]
    bad = [s for s in names if s not in em.MAC_MODELS]
    if bad or not names:
        raise argparse.ArgumentTypeError(f"{bad or text!r}: choose from {','.join(em.MAC_MODELS)}")
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", default=em.MAC_MODEL, choices=em.MAC_MODELS,
                    help="the model the sittings were exact under (default: the one in force)")
    ap.add_argument("--models", type=model_list, default=None,
                    help="comma list of the models to score (default: every other one the emulator has)")
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("sesr")
    p.add_argument("--container", type=Path, required=True)
    p.add_argument("--qdq", type=Path, required=True)
    p.add_argument("--image", type=Path, action="append", default=None)
    p.add_argument("--seed", type=int, action="append", default=None)
    p.add_argument("--silicon-log", type=Path, action="append", default=[])
    p = sub.add_parser("where")
    p.add_argument("--qdq", type=Path, required=True)
    p.add_argument("--image", type=Path, action="append", default=None)
    p.add_argument("--seed", type=int, action="append", default=None)
    p = sub.add_parser("sweep")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--silicon-log", type=Path, action="append", default=[])
    p = sub.add_parser("sixteen")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--integer", type=int, nargs=3, default=None, metavar=("A", "W", "B"))
    p.add_argument("--expected", type=Path, required=True, help="the sitting's build directory's expected.npy")
    p.add_argument("--silicon-log", type=Path, action="append", default=[])
    args = ap.parse_args()
    ref = args.reference
    models = [m for m in (args.models or em.MAC_MODELS) if m != ref]
    rc = {"sesr": sesr_mode, "where": where_mode, "sweep": sweep_mode, "sixteen": sixteen_mode}[args.mode](args, ref, models)
    emit("RESULT", {"mode": args.mode} if args.mode == "where" else {"mode": args.mode, "any_model_refuted": bool(rc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

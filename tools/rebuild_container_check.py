"""Prove a compiler change is a no-op by rebuilding a committed container and diffing the manifest.

WHY THIS EXISTS. ``tests/`` never calls ``compile_graph_container``: the one test that would needs a
build artifact that is not checked in, so it is among the failures this repo carries. That leaves the
single largest function in the compiler with no offline coverage, and "the suite still passes" is
therefore not evidence that a refactor of it changed nothing. A rebuild is the evidence.

The check is deliberately blunt. It recompiles the model a container was built from and compares
every quantity in the manifest that a schedule or a packer could move - instruction-stream bytes,
activation packets, weight-packet bytes, rounds, workspace bytes, the opcode set and the tile
geometry - plus the engine the container declares and whether any placement grew a key. A refactor
that is genuinely a no-op reproduces all of them; one that is not says which one moved.

Generic on purpose: the model, the reference container and the output path are all arguments, so
this serves any container in build/ and any compiler change, not one model.

    bash scripts/research-iron.sh tools/rebuild_container_check.py \\
        --container build/sesr_m7.ignite --model models/sesr_m7_xint8.onnx \\
        --task super_resolution

``--model`` defaults to ``models/<manifest model_name>.onnx``, which is what ignite-compile records,
so it can usually be left off. Output goes to ``scratch/`` and is never written back into ``build/``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

# Quantities a schedule, a packer or a manifest change could move. Anything that shifts here is a
# behaviour change, whatever the diff looked like.
COMPARED = ("insts_bytes", "activation_packets", "weight_fills", "wpackets_bytes", "rounds",
            "workspace_bytes", "kernel_ops", "tile")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--container", type=Path, required=True, help="the committed container to reproduce")
    ap.add_argument("--model", type=Path, default=None,
                    help="the ONNX it was built from (default: models/<model_name>.onnx)")
    ap.add_argument("--task", default=None, help="passed through to compile_graph_container")
    ap.add_argument("--dense-recipe", default=None)
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write the rebuild (default: scratch/<stem>_rebuild.ignite)")
    args = ap.parse_args()

    from ignite_xdna.compiler import engine_compile
    from ignite_xdna.compiler.serializer import IgniteModelReader

    # Check the module under test, not the package __init__. The repo root carries a forwarder whose
    # __path__ points into src/, so ignite_xdna.__file__ can name the forwarder while every submodule
    # still resolves correctly - and reporting that path reads like a fault when it is not one. An
    # editable install shadowing the worktree is the real hazard: the rebuild would quietly use the
    # INSTALLED compiler and prove nothing whatever about the change under test.
    where = Path(engine_compile.__file__).resolve()
    print(f"[env] engine_compile from {where}")
    if ROOT not in where.parents:
        raise SystemExit(f"FAIL: imported the installed compiler, not {ROOT}. Run through "
                         f"scripts/research-iron.sh, which puts src on PYTHONPATH first.")

    reference = (ROOT / args.container) if not args.container.is_absolute() else args.container
    with IgniteModelReader(str(reference)) as reader:
        want_manifest = reader.manifest
    want = want_manifest["graph_engine"]

    model = args.model or Path("models") / f"{want_manifest['model_name']}.onnx"
    model = (ROOT / model) if not model.is_absolute() else model
    if not model.exists():
        raise SystemExit(f"FAIL: {model} is missing; pass --model")
    out = args.out or Path("scratch") / f"{reference.stem}_rebuild.ignite"
    out = (ROOT / out) if not out.is_absolute() else out
    out.parent.mkdir(parents=True, exist_ok=True)

    from ignite_xdna.compiler.engine_compile import compile_graph_container

    t0 = time.perf_counter()
    manifest = compile_graph_container(model, out, build_dir=ROOT / "scratch" / f"{out.stem}_prj",
                                       task=args.task or want_manifest.get("task"),
                                       dense_recipe=args.dense_recipe, verbose=False)
    wall = time.perf_counter() - t0
    got = manifest["graph_engine"]

    bad = [(k, want.get(k), got.get(k)) for k in COMPARED if got.get(k) != want.get(k)]
    if manifest.get("engine") != want_manifest.get("engine"):
        bad.append(("engine", want_manifest.get("engine"), manifest.get("engine")))

    def keyset(ge):
        return sorted({k for p in ge.get("placements", {}).values() for k in p})

    if keyset(got) != keyset(want):
        bad.append(("placement keys", keyset(want), keyset(got)))

    print("REBUILD " + json.dumps({
        "model": model.name, "reference": str(args.container), "engine": manifest.get("engine"),
        "compile_seconds": round(wall, 2),
        **{k: got.get(k) for k in COMPARED},
        "placement_keys": keyset(got),
    }, sort_keys=True))

    if bad:
        for key, w, g in bad:
            print(f"  MISMATCH {key}: committed {w!r} -> rebuilt {g!r}")
        print(f"FAIL: the rebuild does not reproduce {reference.name}; {len(bad)} quantity moved")
        return 1
    print(f"PASS: {reference.name} reproduced on every compared quantity, in {wall:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

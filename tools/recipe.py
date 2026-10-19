#!/usr/bin/env python3
"""Build recipes: how every model here goes from its source weights to what the NPU runs.

A recipe is recipes/<model>.toml: an ordered list of steps (fetch, export, cut, quantize, compile, check), each an
exact command with the files it reads and writes. models/ and build/ are not tracked, so a recipe is the only record
of how a model or container was made; a graph-engine container also carries its own build record in its manifest
(``build``: the command, the compiler commit and the recipe step that ran it).

  python tools/recipe.py list
  python tools/recipe.py show yolov8n
  python tools/recipe.py check                        # offline: every recipe parses, its scripts exist, and each
                                                      # ignite-compile step's flags parse with the real CLI
  python tools/recipe.py status [--root DIR]          # which outputs exist, and which containers in build/ no
                                                      # recipe produces
  python tools/recipe.py build yolov8n [--step compile-overlap] [--from STEP] [--force] [--dry-run] [--root DIR]

Each step names an env: ``shell`` (as written), ``engine`` (the mlir-aie-iron python), ``iron`` (python through
scripts/research-iron.sh, which the graph-engine compiler needs), or a conda env activated through scripts/lib.sh's
``use_env`` (``resnet_env`` to export and quantize, ``resnet_env17`` for the VitisAI EP). ``run`` is the argv after
``python`` for every env but ``shell``. Steps run from --root (default: the checkout this file is in) with
MSYS_NO_PATHCONV=1, so a node name such as /model.10/... reaches the compiler unmangled.

A step whose outputs all exist is skipped; one with only some of them refuses to run; --force rebuilds. A step may
pin ``sha256`` for its outputs (fetches always should): build refuses a mismatch and status reports one.
``recorded_sha256`` instead records what the measured artifact hashed to, for steps that are not bit-reproducible
(export, quantize): a difference is reported, never refused. ``build``
never pushes, never deletes anything but the outputs a forced step rewrites, and opens the NPU only in steps that
say so (``npu = true``), which it refuses while xrt-smi shows a hardware context.

``status`` in a recipe is load-bearing: ``verified`` means the recipe was run as written and its output checked
(the ``verified`` table says when, where and against what); ``reconstructed`` means it was assembled from history
and has not been re-run; ``blocked`` means a step is known not to work today (``blocked_by`` says why).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / "recipes"
HOME = Path.home()
ENGINE_PY = HOME / "miniforge3" / "envs" / "mlir-aie-iron" / "python.exe"
XRT_SMI = Path(r"C:\Windows\System32\AMD\xrt-smi.exe")
ENVS = {"shell", "engine", "iron", "resnet_env", "resnet_env17", "ryzen-ai-1.8.0", "laya-ref"}
STATUSES = {"verified", "reconstructed", "blocked"}


def load(path: Path) -> dict:
    with path.open("rb") as f:
        r = tomllib.load(f)
    r.setdefault("name", path.stem)
    return r


def recipes() -> dict:
    return {p.stem: load(p) for p in sorted(RECIPES.glob("*.toml"))}


def problems(name: str, r: dict) -> list:
    out = []
    if r.get("name") != name:
        out.append(f"name {r.get('name')!r} differs from the file name")
    if r.get("status") not in STATUSES:
        out.append(f"status {r.get('status')!r} is not one of {sorted(STATUSES)}")
    if r.get("status") == "blocked" and not r.get("blocked_by"):
        out.append("status blocked needs blocked_by")
    if r.get("status") == "verified" and not r.get("verified"):
        out.append("status verified needs a [verified] table (date, machine, what was checked)")
    steps = r.get("step", [])
    if not steps:
        out.append("no [[step]]")
    seen = set()
    for s in steps:
        sid = s.get("id")
        if not sid or sid in seen:
            out.append(f"step id {sid!r} missing or repeated")
        seen.add(sid)
        if s.get("env") not in ENVS:
            out.append(f"step {sid}: env {s.get('env')!r} is not one of {sorted(ENVS)}")
        run = s.get("run")
        if not isinstance(run, list) or not run or not all(isinstance(a, str) for a in run):
            out.append(f"step {sid}: run must be a non-empty list of strings")
            continue
        if s.get("env") != "shell" and run[0].endswith(".py") and not (ROOT / run[0]).exists():
            out.append(f"step {sid}: script {run[0]} does not exist in this checkout")
        for key in ("inputs", "outputs"):
            if not isinstance(s.get(key, []), list):
                out.append(f"step {sid}: {key} must be a list")
        for field in ("sha256", "recorded_sha256"):
            for path, digest in s.get(field, {}).items():
                if path not in s.get("outputs", []):
                    out.append(f"step {sid}: {field} names {path}, which is not one of its outputs")
                if not (isinstance(digest, str) and len(digest) == 64
                        and all(c in "0123456789abcdef" for c in digest)):
                    out.append(f"step {sid}: {field} of {path} is not 64 lowercase hex digits")
        if compile_argv(s) is not None:
            try:
                parse_compile(compile_argv(s))
            except SystemExit:
                out.append(f"step {sid}: ignite-compile rejects its flags")
    return out


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_mismatches(step: dict, root: Path, field: str = "sha256") -> list:
    """Outputs that exist and whose sha256 differs from the step's ``sha256`` (pinned: a mismatch is an error) or
    ``recorded_sha256`` (what the artifact behind the published numbers hashed to: a difference is reported, since an
    export or a quantization need not be bit-reproducible)."""
    return [f"{path}: {got[:12]} != {want[:12]}" for path, want in step.get(field, {}).items()
            if (root / path).exists() and (got := sha256_of(root / path)) != want]


def unrecorded_containers(name: str, step: dict, root: Path) -> list:
    """Containers this step writes that exist but whose manifest build record does not name this recipe step: built
    before records existed, by hand, or by another recipe. Such a file is not this step's output, whatever its name."""
    sys.path.insert(0, str(ROOT / "src"))
    from ignite_xdna.compiler.serializer import IgniteModelReader
    out = []
    for o in step.get("outputs", []):
        path = root / o
        if path.suffix != ".ignite" or not path.exists():
            continue
        with IgniteModelReader(path) as reader:
            rec = (reader.manifest.get("build") or {}).get("recipe")
        if rec != f"{name}:{step['id']}":
            out.append(f"{o} (built by {rec or 'no recorded recipe'})")
    return out


def compile_argv(step: dict):
    """The ignite-compile arguments of a step that runs the compiler, else None."""
    run = step.get("run", [])
    if run[:2] == ["-m", "ignite_xdna.compiler.cli"]:
        return run[2:]
    if run[:1] == ["ignite-compile"]:
        return run[1:]
    return None


def parse_compile(argv):
    sys.path.insert(0, str(ROOT / "src"))
    from ignite_xdna.compiler.cli import parse_args
    return parse_args(list(argv))


def command(step: dict) -> list:
    """The argv a step runs. Scripts, the IRON wrapper and therefore the compiler always come from the checkout
    this file is in; --root only says whose models/, data/ and build/ the step reads and writes."""
    env, run = step["env"], list(step["run"])
    if env == "shell":
        return run
    if run[0].endswith(".py") and not Path(run[0]).is_absolute():
        run[0] = str(ROOT / run[0])
    if env == "engine":
        return [str(ENGINE_PY), *run]
    if env == "iron":
        return ["bash", str(ROOT / "scripts" / "research-iron.sh"), *run]
    # A conda env: activated through lib.sh, because Quark needs the env's Scripts\ on PATH.
    return ["bash", "-c", f'. "{(ROOT / "scripts" / "lib.sh").as_posix()}" && use_env {env} && exec python "$@"',
            "recipe", *run]


def npu_idle() -> bool:
    out = subprocess.run([str(XRT_SMI), "examine", "-r", "aie-partitions"], capture_output=True, text=True).stdout
    return "No hardware contexts running" in out


def cmd_list(a) -> int:
    for name, r in recipes().items():
        steps = ", ".join(s.get("id", "?") for s in r.get("step", []))
        print(f"{name:24s} {r.get('status', '?'):13s} {steps}")
    return 0


def cmd_show(a) -> int:
    r = recipes().get(a.name)
    if r is None:
        raise SystemExit(f"no recipe {a.name}; see: python tools/recipe.py list")
    print(f"# {a.name}: {r.get('description', '')}\n# status: {r.get('status')}")
    for k in ("blocked_by", "evidence"):
        if r.get(k):
            print(f"# {k}: {r[k]}")
    for s in r.get("step", []):
        print(f"\n## {s['id']} ({s['env']}{', NPU' if s.get('npu') else ''})")
        if s.get("note"):
            print(f"# {s['note']}")
        print(shlex.join(command(s)))
    return 0


def cmd_check(a) -> int:
    bad = 0
    found = recipes()
    for name, r in found.items():
        for p in problems(name, r):
            print(f"FAIL {name}: {p}")
            bad += 1
    outputs = {}
    for name, r in found.items():
        for s in r.get("step", []):
            for o in s.get("outputs", []):
                if o in outputs:
                    print(f"FAIL {name}: {s['id']} writes {o}, which {outputs[o]} also writes")
                    bad += 1
                outputs[o] = f"{name}:{s['id']}"
    print(f"{'ok' if not bad else 'FAIL'}   {len(found)} recipes, {bad} problem(s)")
    return 1 if bad else 0


def cmd_status(a) -> int:
    root = Path(a.root).resolve()
    produced = set()
    for name, r in recipes().items():
        for s in r.get("step", []):
            outs = s.get("outputs", [])
            produced.update(outs)
            have = sum((root / o).exists() for o in outs)
            bad, moved = hash_mismatches(s, root), hash_mismatches(s, root, "recorded_sha256")
            foreign = unrecorded_containers(name, s, root)
            print(f"{name:20s} {s['id']:22s} {have}/{len(outs)} outputs present"
                  + (f"  NOT FROM THIS RECIPE {foreign}" if foreign else "")
                  + (f"  HASH MISMATCH {bad}" if bad else "")
                  + (f"  differs from the recorded artifact {moved}" if moved else ""))
    orphans = sorted(p.relative_to(root).as_posix() for p in (root / "build").glob("*.ignite")
                     if p.relative_to(root).as_posix() not in produced)
    print(f"\n{len(orphans)} container(s) in build/ that no recipe produces")
    for o in orphans:
        print(f"  {o}")
    return 0


def cmd_build(a) -> int:
    r = recipes().get(a.name)
    if r is None:
        raise SystemExit(f"no recipe {a.name}; see: python tools/recipe.py list")
    if r.get("status") == "blocked" and not a.force:
        raise SystemExit(f"{a.name} is blocked: {r.get('blocked_by')} (--force runs it anyway)")
    root = Path(a.root).resolve()
    steps = r["step"]
    ids = [s["id"] for s in steps]
    for want in (a.step, a.from_step):
        if want and want not in ids:
            raise SystemExit(f"{a.name} has no step {want}; steps: {', '.join(ids)}")
    if a.step:
        steps = [s for s in steps if s["id"] == a.step]
    elif a.from_step:
        steps = steps[ids.index(a.from_step):]
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    for s in steps:
        outs = [root / o for o in s.get("outputs", [])]
        have = [o for o in outs if o.exists()]
        foreign = unrecorded_containers(a.name, s, root)
        if foreign and not a.force:
            raise SystemExit(f"[recipe] {a.name}:{s['id']}: {', '.join(foreign)} exists but was not built by this "
                             "step; check nothing depends on it, then --force replaces it")
        if outs and len(have) == len(outs) and not a.force:
            bad = hash_mismatches(s, root)
            if bad:
                raise SystemExit(f"[recipe] {a.name}:{s['id']}: existing outputs differ from the pinned hash {bad}")
            print(f"[recipe] {a.name}:{s['id']} skipped: its outputs exist (--force rebuilds)")
            continue
        if have and len(have) < len(outs) and not a.force:
            raise SystemExit(f"[recipe] {a.name}:{s['id']}: only some outputs exist ({', '.join(map(str, have))}); "
                             "look before forcing a rebuild")
        missing = [i for i in s.get("inputs", []) if not (root / i).exists()]
        if missing and not a.dry_run:
            raise SystemExit(f"[recipe] {a.name}:{s['id']}: missing inputs {missing}")
        argv = command(s)
        print(f"[recipe] {a.name}:{s['id']} $ {shlex.join(argv)}", flush=True)
        if a.dry_run:
            continue
        if s.get("npu") and not npu_idle():
            raise SystemExit(f"[recipe] {a.name}:{s['id']} needs the NPU and xrt-smi shows a hardware context")
        env["IGNITE_RECIPE"] = f"{a.name}:{s['id']}"
        # `python -m` puts the working directory first on sys.path, and a checkout's root holds an ignite_xdna
        # forwarder to its own src: with --root naming another checkout, its compiler would run instead of this one.
        step_env = dict(env, PYTHONSAFEPATH="1") if s.get("run", [""])[0] == "-m" else env
        rc = subprocess.run(argv, cwd=root, env=step_env).returncode
        if rc:
            raise SystemExit(f"[recipe] {a.name}:{s['id']} failed (exit {rc})")
        absent = [str(o) for o in outs if not o.exists()]
        if absent:
            raise SystemExit(f"[recipe] {a.name}:{s['id']} exited 0 but did not write {absent}")
        bad = hash_mismatches(s, root)
        if bad:
            raise SystemExit(f"[recipe] {a.name}:{s['id']} wrote outputs that differ from the pinned hash {bad}")
        foreign = unrecorded_containers(a.name, s, root)
        if foreign:
            raise SystemExit(f"[recipe] {a.name}:{s['id']} wrote {foreign} without this step's build record: the "
                             "compiler that ran is not this checkout's")
        moved = hash_mismatches(s, root, "recorded_sha256")
        if moved:
            print(f"[recipe] {a.name}:{s['id']}: NOTE the output differs from the artifact the published numbers "
                  f"were measured on {moved}; re-measure before quoting them for this build")
    return 0


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    p = sub.add_parser("show")
    p.add_argument("name")
    p.set_defaults(fn=cmd_show)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    p = sub.add_parser("status")
    p.add_argument("--root", default=str(ROOT))
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("build")
    p.add_argument("name")
    p.add_argument("--step", help="run only this step")
    p.add_argument("--from", dest="from_step", help="run from this step to the end")
    p.add_argument("--force", action="store_true", help="rebuild steps whose outputs exist; run a blocked recipe")
    p.add_argument("--dry-run", action="store_true", help="print the commands only")
    p.add_argument("--root", default=str(ROOT), help="checkout whose models/ and build/ the steps read and write")
    p.set_defaults(fn=cmd_build)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())

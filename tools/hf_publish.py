# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stage, gate and verify what this repo publishes to Hugging Face. It never uploads: `hf upload` does that.

WHY THIS EXISTS. README.md used to promise that model artifacts are not redistributed. Publishing them is
only defensible per family, so tools/hf_manifest.json is the single list of what goes up. Each entry names
its source path, its path on the Hub, the local path the tests expect, its family, and its evidence: how it
is known to pair with its ONNX. Each family names its upstream licence, the URL that licence was read from,
and `verified: true`. Staging REFUSES any entry whose family lacks a verified licence. It also refuses any
file whose bytes contain a private string (a profile path, an assistant's name, a session id). That second
gate runs on the staged bytes, so ONNX metadata and a container's JSON manifest are covered too.

    python tools/hf_publish.py stage                 # -> scratch/hf_stage/model, records sha256s in the manifest
    python tools/hf_publish.py stage-results --rev HEAD   # tracked results/ at that commit -> scratch/hf_stage/results
    python tools/hf_publish.py scan DIR              # the privacy gate alone, on any directory
    python tools/hf_publish.py verify                # re-download at the manifest's revision, compare sha256

Then, by hand:
    hf upload jdominick05/ignite-xdna scratch/hf_stage/model . --commit-message "..."
    hf upload jdominick05/ignite-xdna-results scratch/hf_stage/results . --repo-type dataset --commit-message "..."
and record the model upload's commit sha as the manifest's `revision`, which is what fetch_hf_artifacts.py pins.
"""
import argparse
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from fetch_hf_artifacts import MANIFEST, download, hf_token, sha256_file  # noqa: E402

STAGE = ROOT / "scratch" / "hf_stage"

# Bytes that must never reach a public repo. Matched case-insensitively against the raw file, and again
# against its UTF-16LE form so a wide-string path is caught too. Kept narrow on purpose: the scrubbed
# `C:\Users\<user>` form that results/ already uses publicly does not match.
PRIVATE = [rb"users[\\/]+ignis", rb"users[\\/]+flame", rb"claude", rb"anthropic", rb"session_0"]
PRIVATE_RE = re.compile(b"|".join(PRIVATE), re.IGNORECASE)


def _utf16(word):
    """A regex for `word` stored as UTF-16LE: each character followed by a NUL byte."""
    return b"".join(re.escape(ch.encode()) + b"\x00" for ch in word)


PRIVATE_RE16 = re.compile(b"|".join([_utf16("users") + rb"[\\/]\x00" + _utf16("ignis"),
                                     _utf16("users") + rb"[\\/]\x00" + _utf16("flame"),
                                     _utf16("claude"), _utf16("anthropic")]), re.IGNORECASE)


def scan_bytes(data):
    hits = {m.group(0).decode("latin-1") for m in PRIVATE_RE.finditer(data)}
    hits |= {m.group(0).decode("utf-16le", "replace") for m in PRIVATE_RE16.finditer(data)}
    return sorted(hits)


def scan_dir(root):
    bad = []
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            hits = scan_bytes(p.read_bytes())
            if hits:
                bad.append((p, hits))
    return bad


def load_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def save_manifest(m):
    MANIFEST.write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")


def cmd_stage(args):
    m = load_manifest()
    fams = m["families"]
    out = STAGE / "model"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    errors = []
    for e in m["entries"]:
        fam = fams.get(e["family"])
        if not fam or not fam.get("verified") or not fam.get("license") or not fam.get("license_source"):
            errors.append(f"{e['dest']}: family {e['family']!r} has no verified licence")
            continue
        if not e.get("evidence"):
            errors.append(f"{e['dest']}: no evidence recorded")
            continue
        src = ROOT / e["src"]
        if not src.is_file():
            errors.append(f"{e['dest']}: source {e['src']} missing")
            continue
        hits = scan_bytes(src.read_bytes())
        if hits:
            errors.append(f"{e['dest']}: private strings {hits}")
            continue
        dst = out / e["dest"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        e["sha256"] = sha256_file(dst)
        e["bytes"] = dst.stat().st_size
    if errors:
        print("[stage] REFUSED:\n  " + "\n  ".join(errors))
        return 1

    # A licence notice in every folder a family's files land in (a family can span its own folder and fixtures/).
    for name, fam in fams.items():
        files = [e["dest"] for e in m["entries"] if e["family"] == name]
        for folder in sorted({str(Path(f).parent) for f in files}):
            here = [f for f in files if str(Path(f).parent) == folder]
            sources = " and ".join(f"<{u.strip()}>" for u in fam["license_source"].split(";"))
            lines = [f"# {name}: licence and provenance", "",
                     f"Weights: {fam['upstream']}. Licence: **{fam['license']}**, read from {sources}.", ""]
            if fam.get("note"):
                lines += [fam["note"].strip(), ""]
            lines += ["The quantization, export and container compilation are ignite-xdna's work, licensed",
                      "AGPL-3.0-or-later. Where the upstream licence differs, a file here carries both, and any",
                      "use of it must satisfy both.", "", "Files:", ""]
            lines += [f"- `{Path(f).name}`" for f in here]
            (out / folder / "LICENSE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    card = ROOT / m["model_card"]
    shutil.copyfile(card, out / "README.md")
    public = {k: v for k, v in m.items() if k not in ("model_card", "results_card")}
    (out / "manifest.json").write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")

    bad = scan_dir(out)
    if bad:
        print("[stage] REFUSED, private strings in staged files:")
        for p, hits in bad:
            print(f"  {p.relative_to(out)}: {hits}")
        return 1
    save_manifest(m)
    total = sum(e["bytes"] for e in m["entries"])
    print(f"[stage] {len(m['entries'])} files, {total / 1e6:.1f} MB -> {out}; privacy scan clean; "
          f"sha256s recorded in {MANIFEST.relative_to(ROOT)}")
    return 0


def cmd_stage_results(args):
    m = load_manifest()
    sha = subprocess.run(["git", "rev-parse", args.rev], cwd=ROOT, check=True, capture_output=True,
                         text=True).stdout.strip()
    out = STAGE / "results"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    tar_bytes = subprocess.run(["git", "-c", "core.autocrlf=false", "archive", "--format=tar", sha, "results"],
                               cwd=ROOT, check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as t:
        t.extractall(out, filter="data")
    card = (ROOT / m["results_card"]).read_text(encoding="utf-8").replace("{{SHA}}", sha)
    (out / "README.md").write_text(card, encoding="utf-8")
    bad = scan_dir(out)
    if bad:
        print("[stage-results] REFUSED, private strings:")
        for p, hits in bad[:50]:
            print(f"  {p.relative_to(out)}: {hits}")
        return 1
    n = sum(1 for p in out.rglob("*") if p.is_file())
    print(f"[stage-results] {n} files from results/ at {sha} -> {out}; privacy scan clean")
    return 0


def cmd_scan(args):
    bad = scan_dir(args.dir)
    for p, hits in bad:
        print(f"  {p}: {hits}")
    print(f"[scan] {len(bad)} file(s) with private strings")
    return 1 if bad else 0


def cmd_verify(args):
    m = load_manifest()
    rev = args.revision or m.get("revision")
    if not rev:
        print("[verify] manifest has no revision; pass --revision")
        return 1
    tmp = STAGE / "verify"
    if tmp.exists():
        shutil.rmtree(tmp)
    token = hf_token()
    for e in m["entries"]:
        download(m["model_repo"], "model", rev, e["dest"], tmp / e["dest"], e["sha256"], token)
        print(f"  ok {e['dest']}")
    shutil.rmtree(tmp)
    print(f"[verify] {len(m['entries'])} files at {m['model_repo']}@{rev} match the manifest's sha256")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stage")
    r = sub.add_parser("stage-results")
    r.add_argument("--rev", default="HEAD")
    s = sub.add_parser("scan")
    s.add_argument("dir", type=Path)
    v = sub.add_parser("verify")
    v.add_argument("--revision", default=None)
    args = ap.parse_args()
    return {"stage": cmd_stage, "stage-results": cmd_stage_results, "scan": cmd_scan, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

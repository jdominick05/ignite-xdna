# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fetch the published models, containers and test fixtures from Hugging Face into models/ and build/.

WHY THIS EXISTS. models/ and build/ are git-ignored and cost hours to regenerate, so on a fresh clone most of
tests/ skips: the tests look for artifacts at hardcoded repo-relative paths (models/yolov8n_cut_xint8.onnx,
build/sesr_m7.ignite, ...). The canonical set is published at https://huggingface.co/jdominick05/ignite-xdna,
and tools/hf_manifest.json maps every published file to the exact local path the tests and pipelines expect.
This puts each one there, so the tests run instead of skipping, with no test edits.

Every file is checked against the sha256 the manifest records. Downloads are pinned to the manifest's
`revision` (a Hub commit sha, never `main`); --revision overrides it. A local file that already has the
recorded hash is left alone. A local file with a DIFFERENT hash is never overwritten without --force, because
it is usually somebody's own build.

Standard library only, so it runs in any env (the engine envs do not ship huggingface_hub). While the repo
is private a token is needed: HF_TOKEN, else $HF_HOME/token, else ~/.cache/huggingface/token.

    python tools/fetch_hf_artifacts.py                     # everything with a local path
    python tools/fetch_hf_artifacts.py --family yolov8n --family sesr_m7
    python tools/fetch_hf_artifacts.py --list              # what would be fetched, and what is already here

A fetched container is the one this repo device-verified (manifest `evidence`), not necessarily the build a
given docs/BENCHMARKS.md figure was measured on: those figures name their own logs and builds.
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tools" / "hf_manifest.json"
ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hf_token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"].strip()
    homes = [Path(os.environ["HF_HOME"])] if os.environ.get("HF_HOME") else []
    homes.append(Path.home() / ".cache" / "huggingface")
    for home in homes:
        tok = home / "token"
        if tok.is_file():
            return tok.read_text(encoding="utf-8").strip()
    return None


def resolve_url(repo_id, repo_type, revision, path_in_repo):
    prefix = "" if repo_type == "model" else f"{repo_type}s/"
    quoted = "/".join(urllib.parse.quote(part) for part in path_in_repo.split("/"))
    return f"{ENDPOINT}/{prefix}{repo_id}/resolve/{revision}/{quoted}"


def download(repo_id, repo_type, revision, path_in_repo, dest, expected_sha256, token=None):
    """Download one file to dest (via a .part file) and check its sha256. Raises on any mismatch."""
    url = resolve_url(repo_id, repo_type, revision, path_in_repo)
    req = urllib.request.Request(url, headers={"User-Agent": "ignite-xdna-fetch/1"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    h = hashlib.sha256()
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as f:
            for chunk in iter(lambda: r.read(1 << 20), b""):
                h.update(chunk)
                f.write(chunk)
    except urllib.error.HTTPError as ex:
        part.unlink(missing_ok=True)
        hint = " (private repo: set HF_TOKEN or run `hf auth login`)" if ex.code in (401, 403, 404) else ""
        raise RuntimeError(f"{path_in_repo}: HTTP {ex.code} from {url}{hint}") from ex
    got = h.hexdigest()
    if expected_sha256 and got != expected_sha256:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{path_in_repo}: sha256 {got} != manifest {expected_sha256}")
    os.replace(part, dest)
    return got


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--revision", default=None, help="Hub commit sha (default: the manifest's `revision`)")
    ap.add_argument("--family", action="append", default=None, help="only these families (repeatable)")
    ap.add_argument("--list", action="store_true", help="report local state, download nothing")
    ap.add_argument("--force", action="store_true", help="overwrite a local file whose hash differs")
    args = ap.parse_args()

    m = json.loads(args.manifest.read_text(encoding="utf-8"))
    revision = args.revision or m.get("revision")
    if not revision:
        sys.exit("manifest has no `revision` yet (nothing published); pass --revision")
    token = hf_token()
    entries = [e for e in m["entries"] if e.get("local") and (not args.family or e["family"] in args.family)]
    if not entries:
        sys.exit("no manifest entries match")

    fetched = kept = refused = 0
    for e in entries:
        local = ROOT / e["local"]
        if local.is_file():
            have = sha256_file(local)
            if have == e["sha256"]:
                print(f"  ok      {e['local']}")
                kept += 1
                continue
            if not args.force:
                print(f"  DIFFERS {e['local']} (local {have[:12]} != published {e['sha256'][:12]}; --force replaces it)")
                refused += 1
                continue
        if args.list:
            print(f"  missing {e['local']}  <- {e['dest']}")
            continue
        print(f"  fetch   {e['local']}  <- {m['model_repo']}@{revision[:12]}:{e['dest']}", flush=True)
        download(m["model_repo"], "model", revision, e["dest"], local, e["sha256"], token)
        fetched += 1
    print(f"[fetch] {fetched} fetched, {kept} already present, {refused} left alone (local copy differs)")
    return 1 if refused and not args.list else 0


if __name__ == "__main__":
    sys.exit(main())

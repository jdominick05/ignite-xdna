#!/usr/bin/env python3
"""Audit numeric claims in the docs against the logs in results/.

Two checks, both advisory — this reports, it does not decide:

  1. TRACEABILITY. Every decimal figure quoted in a doc should appear in at least
     one file under results/. A figure that appears nowhere in the logs was either
     typed from memory, transcribed wrong, or belongs to a log that never got
     committed. All three are worth knowing about.

  2. RETRACTION ORPHANS. This repo retracts and supersedes numbers regularly. When
     that happens the old figure often survives in a doc nobody edited. This lists
     every retraction/supersession sentence alongside the figures near it, so you
     can check whether the dead number still lives elsewhere.

Usage:
  python scripts/audit_numbers.py                # full audit
  python scripts/audit_numbers.py --untraced     # just check 1
  python scripts/audit_numbers.py --retractions  # just check 2
"""
import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# Decimals and ratios worth checking. Bare integers produce far too much noise
# (opset 17, 4x4, line numbers), so require a decimal point or a slash pair.
FIGURE = re.compile(r"(?<![\w.])(\d{1,6}\.\d{1,3})(?![\w.])")
NODES = re.compile(r"(?<![\w.])(\d{2,5}\s*/\s*\d{2,5})(?![\w.])")

RETRACTION = re.compile(
    r"\b(retracted|retraction|superseded|supersedes|no longer|was wrong|"
    r"incorrect|corrected|refuted|does not hold)\b",
    re.I,
)

SKIP_DIRS = {".git", "node_modules", "__pycache__"}


def md_files(root: Path):
    """The repo's own markdown, i.e. what git tracks.

    rglob would also pull in the vendored RyzenAI-SW/ checkout and any scratch .md in
    the working tree, whose figures have nothing to do with this repo's logs.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "*.md"],
            capture_output=True, text=True, check=True,
        ).stdout.split("\n")
        tracked = sorted(root / line for line in out if line.strip())
        if tracked:
            return tracked
    except (OSError, subprocess.CalledProcessError):
        pass
    return sorted(
        p for p in root.rglob("*.md")
        if not SKIP_DIRS & set(p.parts) and "results" not in p.parts[:-1]
    )


def log_corpus(root: Path) -> str:
    results = root / "results"
    if not results.exists():
        return ""
    chunks = []
    for p in results.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".log", ".txt", ".json"} and not p.name.startswith("dets_"):
            try:
                chunks.append(p.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass
    return "\n".join(chunks)


def normalize(fig: str) -> str:
    return re.sub(r"\s+", "", fig)


def untraced(root: Path):
    corpus = log_corpus(root)
    if not corpus:
        print("no results/ logs found — traceability check skipped")
        return {}
    corpus_norm = re.sub(r"\s+", "", corpus)
    missing = defaultdict(list)
    for doc in md_files(root):
        body = doc.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(body.splitlines(), 1):
            for fig in FIGURE.findall(line) + NODES.findall(line):
                n = normalize(fig)
                if n not in corpus_norm:
                    missing[str(doc.relative_to(root))].append((lineno, fig, line.strip()[:110]))
    return missing


def retractions(root: Path):
    hits = []
    for doc in md_files(root):
        body = doc.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(body.splitlines(), 1):
            if RETRACTION.search(line):
                figs = sorted(set(FIGURE.findall(line) + NODES.findall(line)))
                hits.append((str(doc.relative_to(root)), lineno, line.strip()[:200], figs))
    return hits


def main():
    # The docs are full of arrows and em dashes; a cp1252 console raises on the first
    # one and takes the whole audit with it.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--untraced", action="store_true")
    ap.add_argument("--retractions", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    both = not (args.untraced or args.retractions)

    if both or args.untraced:
        print("=" * 72)
        print("UNTRACED FIGURES — quoted in a doc, not found in any results/ log")
        print("=" * 72)
        miss = untraced(root)
        total = sum(len(v) for v in miss.values())
        if not total:
            print("none — every figure traces to a log")
        for doc, items in sorted(miss.items()):
            print(f"\n{doc}  ({len(items)})")
            for lineno, fig, ctx in items[:40]:
                print(f"  L{lineno:<5} {fig:<12} {ctx}")
            if len(items) > 40:
                print(f"  ... {len(items) - 40} more")
        print(f"\ntotal untraced: {total}")
        print("\nNot all of these are bugs — derived figures (a ratio you computed, a")
        print("percentage of nameplate) legitimately won't appear in a log. Check that")
        print("each one is derived and not remembered.\n")

    if both or args.retractions:
        print("=" * 72)
        print("RETRACTION / SUPERSESSION NOTICES — check each for orphaned old figures")
        print("=" * 72)
        hits = retractions(root)
        if not hits:
            print("none found")
        for doc, lineno, line, figs in hits:
            print(f"\n{doc}:{lineno}")
            print(f"  {line}")
            if figs:
                print(f"  figures on this line: {', '.join(figs)}")
        print(f"\ntotal notices: {len(hits)}")
        print("\nFor each: grep the retracted figure across all docs. If it still appears")
        print("anywhere without the retraction attached, that's the orphan to fix.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

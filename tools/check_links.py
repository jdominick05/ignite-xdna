#!/usr/bin/env python3
"""Verify every markdown link in the repo resolves, and enforce the README size budget.

Run from the repo root:  python scripts/check_links.py
Exit code 1 if anything is broken, so it can gate a commit.
"""
import re
import subprocess
import sys
from pathlib import Path

README_LINE_BUDGET = 250

LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.M)
# Only flag real URL schemes, not Windows drive letters (C:\...) or bare words.
EXTERNAL = re.compile(r"^(https?|mailto|ftp):", re.I)


def slug(text: str) -> str:
    """GitHub's heading -> anchor transform."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    # GitHub replaces each space with one hyphen and does NOT collapse runs, so a
    # heading with " - " or an em dash keeps a double hyphen in its anchor. Collapsing
    # here would accept links that are broken on GitHub itself.
    return text.replace(" ", "-")


def anchors(path: Path) -> set:
    try:
        body = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()
    seen, out = {}, set()
    for _, text in HEADING.findall(body):
        base = slug(text)
        n = seen.get(base, 0)
        out.add(base if n == 0 else f"{base}-{n}")
        seen[base] = n + 1
    return out


def md_files(root: Path):
    """The repo's own markdown, i.e. what git tracks.

    rglob would also pull in the vendored RyzenAI-SW/ checkout and any scratch .md
    sitting in the working tree, neither of which this repo is responsible for.
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
    return sorted(p for p in root.rglob("*.md") if ".git" not in p.parts)


def main(root="."):
    root = Path(root).resolve()
    docs = md_files(root)
    anchor_cache = {}
    problems = []

    for doc in docs:
        body = doc.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(body.splitlines(), 1):
            for target in LINK.findall(line):
                if EXTERNAL.match(target) or target.startswith("#") is False and "://" in target:
                    continue
                path_part, _, anchor = target.partition("#")

                if not path_part:  # same-file anchor
                    dest = doc
                else:
                    dest = (doc.parent / path_part).resolve()
                    if not dest.exists():
                        problems.append(f"{doc.relative_to(root)}:{lineno}  missing path -> {target}")
                        continue

                if anchor:
                    if dest.suffix != ".md":
                        continue
                    if dest not in anchor_cache:
                        anchor_cache[dest] = anchors(dest)
                    if anchor.lower() not in anchor_cache[dest]:
                        problems.append(
                            f"{doc.relative_to(root)}:{lineno}  missing anchor -> {target}"
                        )

    readme = root / "README.md"
    if readme.exists():
        n = len(readme.read_text(encoding="utf-8", errors="replace").splitlines())
        if n > README_LINE_BUDGET:
            problems.append(
                f"README.md is {n} lines, over the {README_LINE_BUDGET}-line landing-page budget. "
                "Move depth into docs/BENCHMARKS.md rather than trimming caveats."
            )

    print(f"checked {len(docs)} markdown files")
    if problems:
        print(f"\n{len(problems)} problem(s):\n")
        for p in problems:
            print("  " + p)
        return 1
    print("all links resolve; README within budget")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))

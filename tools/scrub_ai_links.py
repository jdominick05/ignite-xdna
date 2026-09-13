#!/usr/bin/env python3
"""
Scrub AI assistant links and session URLs from git commit messages.

Ensures that no links to AI assistants (Claude, Codex, ChatGPT, Antigravity,
Gemini, OpenAI, Anthropic, etc.) are present in git commit messages before pushing.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

# Patterns matching AI assistant URLs or session links
AI_URL_PATTERNS = [
    re.compile(r"https?://[^\s]*(?:claude\.ai|codex\.|chatgpt\.com|chat\.openai\.com|platform\.openai\.com|anthropic\.com|openai\.com|gemini\.google|antigravity)[^\s]*", re.I),
    re.compile(r"agy://[^\s]+", re.I),
]

# Patterns matching session trailer lines that contain URLs
SESSION_TRAILER_URL_PATTERN = re.compile(
    r"^\s*([A-Za-z0-9_-]+-Session):\s*(?:https?://[^\s]*|agy://[^\s]*)\s*$",
    re.I,
)


def extract_session_id(val: str) -> str:
    """Extract clean session ID, stripping any URL schemes, domains, or paths."""
    v = val.strip()
    # Strip URL prefixes like https://claude.ai/code/, https://codex.local/session/, agy://, etc.
    v = re.sub(
        r"^(?:https?://[^/]+/)*(?:(?:code|session)/)*(?:session_)?(?:agy://)?",
        "",
        v,
        flags=re.I,
    )
    # Strip any query parameters or trailing slashes
    v = v.split("?")[0].rstrip("/")
    return v


def has_ai_links(msg: str) -> bool:
    """Return True if message contains any AI assistant links/URLs."""
    for line in msg.splitlines():
        if SESSION_TRAILER_URL_PATTERN.match(line):
            return True
        for pat in AI_URL_PATTERNS:
            if pat.search(line):
                return True
    return False


def clean_commit_message(msg: str) -> str:
    """
    Remove all AI assistant links and URLs from a commit message.

    - Session trailer lines with URLs (e.g. Agy-Session: https://claude.ai/...)
      are stripped entirely to avoid leaving broken/empty trailers.
    - Inline URLs to AI assistants are stripped.
    - Preserves normal git trailers, commit subjects, and regular URLs (e.g. github.com).
    """
    lines = []
    for line in msg.splitlines():
        # Drop session trailer lines containing URLs
        if SESSION_TRAILER_URL_PATTERN.match(line):
            continue

        # Strip any AI assistant URLs from remaining lines
        cleaned_line = line
        for pat in AI_URL_PATTERNS:
            cleaned_line = pat.sub("", cleaned_line).rstrip()

        # If a session trailer became empty, drop it
        if re.match(r"^\s*[A-Za-z0-9_-]+-Session:\s*$", cleaned_line):
            continue

        lines.append(cleaned_line)

    # Trim trailing blank lines
    while lines and not lines[-1].strip():
        lines.pop()

    return "\n".join(lines) + "\n" if lines else ""


def get_unpushed_commits(base: str | None = None, head: str = "HEAD") -> list[str]:
    """Get list of commit hashes in base..head in topological order (oldest first)."""
    if base:
        cmd = ["git", "rev-list", "--reverse", f"{base}..{head}"]
    else:
        cmd = ["git", "rev-list", "--reverse", head]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return []
    return [c.strip() for c in res.stdout.splitlines() if c.strip()]


def check_commits(commits: list[str]) -> list[tuple[str, str, str]]:
    """
    Check commits for AI assistant links.
    Returns list of (commit_hash, subject, offending_content).
    """
    violations = []
    for commit in commits:
        body = subprocess.check_output(
            ["git", "log", "-n", "1", "--format=%B", commit], text=True
        )
        if has_ai_links(body):
            subj = subprocess.check_output(
                ["git", "log", "-n", "1", "--format=%s", commit], text=True
            ).strip()
            # Find offending lines
            offending = [
                line.strip()
                for line in body.splitlines()
                if SESSION_TRAILER_URL_PATTERN.match(line)
                or any(pat.search(line) for pat in AI_URL_PATTERNS)
            ]
            violations.append((commit, subj, "; ".join(offending)))
    return violations


def scrub_commits(commits: list[str], branch: str | None = None) -> int:
    """
    Scrub commit messages in-place for unpushed commits using git plumbing.
    Returns number of commits modified.
    """
    if not commits:
        return 0

    # First check which commits need modification
    to_modify = []
    for c in commits:
        body = subprocess.check_output(
            ["git", "log", "-n", "1", "--format=%B", c], text=True
        )
        if has_ai_links(body):
            to_modify.append(c)

    if not to_modify:
        return 0

    # If only HEAD needs modification and commits has 1 commit:
    # We can do this cleanly via git plumbing or git commit --amend
    old_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

    # Find the parent of the first commit in commits
    first_commit = commits[0]
    parents_raw = subprocess.check_output(
        ["git", "log", "-n", "1", "--format=%P", first_commit], text=True
    ).strip().split()
    parent = parents_raw[0] if parents_raw else None

    # Rebuild the commit chain
    current_parent = parent
    for c in commits:
        tree = subprocess.check_output(
            ["git", "log", "-n", "1", "--format=%T", c], text=True
        ).strip()
        an = subprocess.check_output(
            ["git", "log", "-n", "1", "--format=%an", c], text=True
        ).strip()
        ae = subprocess.check_output(
            ["git", "log", "-n", "1", "--format=%ae", c], text=True
        ).strip()
        ad = subprocess.check_output(
            ["git", "log", "-n", "1", "--format=%ad", c], text=True
        ).strip()
        cn = subprocess.check_output(
            ["git", "log", "-n", "1", "--format=%cn", c], text=True
        ).strip()
        ce = subprocess.check_output(
            ["git", "log", "-n", "1", "--format=%ce", c], text=True
        ).strip()
        cd = subprocess.check_output(
            ["git", "log", "-n", "1", "--format=%cd", c], text=True
        ).strip()
        old_body = subprocess.check_output(
            ["git", "log", "-n", "1", "--format=%B", c], text=True
        )
        new_body = clean_commit_message(old_body)

        env = os.environ.copy()
        env.update({
            "GIT_AUTHOR_NAME": an,
            "GIT_AUTHOR_EMAIL": ae,
            "GIT_AUTHOR_DATE": ad,
            "GIT_COMMITTER_NAME": cn,
            "GIT_COMMITTER_EMAIL": ce,
            "GIT_COMMITTER_DATE": cd,
        })

        commit_cmd = ["git", "commit-tree", tree, "-m", new_body]
        if current_parent:
            commit_cmd.extend(["-p", current_parent])

        new_commit = subprocess.check_output(commit_cmd, env=env, text=True).strip()
        current_parent = new_commit

    # Update HEAD/branch to the new commit tip
    if branch:
        subprocess.check_call(["git", "update-ref", f"refs/heads/{branch}", current_parent])
    subprocess.check_call(["git", "reset", "--hard", current_parent])

    return len(to_modify)


def find_remote_base(remote: str, branch: str) -> str | None:
    """Find the tracking base commit on remote for branch."""
    # 1. Exact remote branch ref
    res = subprocess.run(
        ["git", "rev-parse", "--verify", f"{remote}/{branch}"],
        capture_output=True,
        text=True,
    )
    if res.returncode == 0:
        return res.stdout.strip()

    # 2. Merge-base with remote tracking branches (e.g. origin/main, origin/HEAD)
    for fallback in [f"{remote}/main", f"{remote}/master", f"{remote}/HEAD"]:
        mb = subprocess.run(
            ["git", "merge-base", "HEAD", fallback],
            capture_output=True,
            text=True,
        )
        if mb.returncode == 0 and mb.stdout.strip():
            return mb.stdout.strip()

    # 3. Fallback to HEAD~1 to avoid touching full history on untracked branches
    res = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD~1"],
        capture_output=True,
        text=True,
    )
    if res.returncode == 0:
        return res.stdout.strip()

    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrub AI assistant links from commit messages.")
    parser.add_argument("--check", action="store_true", help="Check commits and exit 1 if AI links found.")
    parser.add_argument("--scrub", action="store_true", help="Scrub AI links from commits.")
    parser.add_argument("--extract-id", type=str, default=None, help="Extract raw session ID without URLs from a string.")
    parser.add_argument("--clean-file", type=str, default=None, help="Clean AI assistant links from a commit message file.")
    parser.add_argument("--base", type=str, default=None, help="Base commit to check/scrub from (default: remote tracking ref).")
    parser.add_argument("--remote", type=str, default="origin", help="Remote name to compare against.")
    parser.add_argument("--branch", type=str, default=None, help="Branch name (default: current branch).")

    args = parser.parse_args()

    if args.extract_id is not None:
        print(extract_session_id(args.extract_id))
        return 0

    if args.clean_file is not None:
        with open(args.clean_file, "r", encoding="utf-8") as f:
            content = f.read()
        cleaned = clean_commit_message(content)
        with open(args.clean_file, "w", encoding="utf-8") as f:
            f.write(cleaned)
        return 0

    # Get current branch if not provided
    branch = args.branch
    if not branch:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()

    base = args.base
    if not base:
        base = find_remote_base(args.remote, branch)

    commits = get_unpushed_commits(base=base, head="HEAD")

    if args.scrub:
        if not commits:
            print("No unpushed commits to scrub.")
            return 0
        count = scrub_commits(commits, branch=branch)
        if count > 0:
            print(f"Scrubbed AI assistant links from {count} commit(s).")
        else:
            print("All unpushed commits are clean (no AI assistant links found).")
        return 0

    if args.check:
        if not commits:
            print("No unpushed commits to check.")
            return 0
        violations = check_commits(commits)
        if violations:
            print(f"ERROR: Found {len(violations)} commit(s) with AI assistant links:")
            for c, subj, off in violations:
                print(f"  {c[:8]} {subj}")
                print(f"    offending: {off}")
            return 1
        print("OK: No AI assistant links found in outgoing commits.")
        return 0

    # Default action if neither --check nor --scrub specified: scrub then verify
    if commits:
        count = scrub_commits(commits, branch=branch)
        if count > 0:
            print(f"Scrubbed AI assistant links from {count} commit(s).")
        violations = check_commits(get_unpushed_commits(base=base, head="HEAD"))
        if violations:
            print(f"ERROR: {len(violations)} commit(s) still contain AI assistant links:")
            for c, subj, off in violations:
                print(f"  {c[:8]} {subj} ({off})")
            return 1
    print("OK: Commits verified clean of AI assistant links.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

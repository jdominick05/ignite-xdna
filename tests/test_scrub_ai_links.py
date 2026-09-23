"""Unit tests for tools/scrub_ai_links.py."""

import subprocess
import sys

import pytest
from tools.scrub_ai_links import clean_commit_message, has_ai_links


def test_has_ai_links_detects_claude():
    msg = "feat: something\n\nClaude-Session: https://claude.ai/code/session_12345"
    assert has_ai_links(msg) is True


def test_has_ai_links_detects_agy():
    msg = "feat: something\n\nAgy-Session: https://claude.ai/code/session_12345"
    assert has_ai_links(msg) is True


def test_has_ai_links_detects_agy_uri():
    msg = "feat: something\n\nAgy-Session: agy://5561e8e7-10eb-4ebc-aaaa-e992a06ef8d7"
    assert has_ai_links(msg) is True


def test_has_ai_links_detects_codex():
    msg = "feat: something\n\nCodex-Session: https://codex.local/session/12345"
    assert has_ai_links(msg) is True


def test_has_ai_links_detects_inline_url():
    msg = "feat: something\n\nReferenced https://chatgpt.com/c/12345 for help"
    assert has_ai_links(msg) is True


def test_has_ai_links_ignores_github_and_normal_links():
    msg = "feat: something\n\nSee https://github.com/jdominick05/ignite-xdna/pull/1\nCo-Authored-By: Jane Doe <jane@example.com>"
    assert has_ai_links(msg) is False


def test_clean_commit_message_removes_session_urls():
    msg = """feat(compiler): awesome feature

Detailed explanation of changes:
- Item 1
- Item 2

Co-Authored-By: Antigravity <noreply@google.com>
Agy-Session: https://claude.ai/code/session_5561e8e7
"""
    cleaned = clean_commit_message(msg)
    assert "https://claude.ai" not in cleaned
    assert "Agy-Session" not in cleaned
    assert "Co-Authored-By: Antigravity <noreply@google.com>" in cleaned
    assert "Detailed explanation of changes:" in cleaned
    assert cleaned.endswith("\n")


def test_clean_commit_message_preserves_clean_commit():
    msg = """feat(compiler): awesome feature

Detailed explanation of changes:
- Item 1
- Item 2
"""
    cleaned = clean_commit_message(msg)
    assert cleaned.strip() == msg.strip()


def test_clean_commit_message_removes_inline_ai_urls():
    msg = "feat: update prompt from https://claude.ai/code/123 for accuracy"
    cleaned = clean_commit_message(msg)
    assert "https://claude.ai" not in cleaned
    assert "feat: update prompt from  for accuracy" in cleaned


def test_extract_session_id_url():
    from tools.scrub_ai_links import extract_session_id

    assert extract_session_id("https://claude.ai/code/session_5561e8e7") == "5561e8e7"
    assert (
        extract_session_id(
            "https://codex.local/session/01a0994d-b6aa-7e20-bc28-6db6dd752df4"
        )
        == "01a0994d-b6aa-7e20-bc28-6db6dd752df4"
    )
    assert (
        extract_session_id("agy://5561e8e7-10eb-4ebc-aaaa-e992a06ef8d7")
        == "5561e8e7-10eb-4ebc-aaaa-e992a06ef8d7"
    )
    assert extract_session_id("5561e8e7") == "5561e8e7"


def test_clean_file(tmp_path):
    from tools.scrub_ai_links import main
    import sys

    f = tmp_path / "msg.txt"
    f.write_text("feat: test\n\nAgy-Session: https://claude.ai/code/123\n")
    old_argv = sys.argv
    try:
        sys.argv = ["scrub_ai_links", "--clean-file", str(f)]
        ret = main()
        assert ret == 0
        content = f.read_text()
        assert "https://claude.ai" not in content
        assert "Agy-Session" not in content
        assert content.strip() == "feat: test"
    finally:
        sys.argv = old_argv


def _git(repo, *args):
    return subprocess.run(
        ["git", "-c", "user.name=T", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false", *args],
        cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _run_main(*argv):
    from tools.scrub_ai_links import main

    old_argv = sys.argv
    try:
        sys.argv = ["scrub_ai_links", *argv]
        return main()
    finally:
        sys.argv = old_argv


@pytest.fixture
def side_repo(tmp_path, monkeypatch):
    """main checked out; branch `side` carries one commit with a session URL. Returns the base."""
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "base")
    base = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "checkout", "-q", "-b", "side")
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m",
         "side: linked\n\nClaude-Session: https://claude.ai/code/session_12345")
    _git(tmp_path, "checkout", "-q", "main")
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "main: clean")
    monkeypatch.chdir(tmp_path)
    return base


def test_check_reads_the_named_branch_not_head(side_repo, capsys):
    # push.sh --branch side from a checkout of main: the range is base..side, not base..HEAD
    assert _run_main("--check", "--branch", "side", "--base", side_repo) == 1
    assert "side: linked" in capsys.readouterr().out


def test_scrub_refuses_a_branch_not_checked_out_here(side_repo, capsys):
    side, main_ = _git(".", "rev-parse", "side"), _git(".", "rev-parse", "main")
    assert _run_main("--branch", "side", "--base", side_repo) == 1
    assert "Run --scrub from a checkout of 'side'" in capsys.readouterr().out
    assert _git(".", "rev-parse", "side") == side
    assert _git(".", "rev-parse", "main") == main_
    assert _git(".", "rev-parse", "--abbrev-ref", "HEAD") == "main"


def test_clean_branch_not_checked_out_here_passes(side_repo):
    # the gate's case: push.sh --branch main from a checkout of another branch, nothing to scrub
    _git(".", "checkout", "-q", "side")
    main_, side = _git(".", "rev-parse", "main"), _git(".", "rev-parse", "side")
    assert _run_main("--branch", "main", "--base", side_repo) == 0
    assert _git(".", "rev-parse", "main") == main_
    assert _git(".", "rev-parse", "side") == side


def test_scrub_rewrites_the_checked_out_branch(side_repo):
    _git(".", "checkout", "-q", "side")
    tree = _git(".", "rev-parse", "side^{tree}")
    assert _run_main("--scrub", "--branch", "side", "--base", side_repo) == 0
    assert "claude.ai" not in _git(".", "log", "-1", "--format=%B", "side")
    assert _git(".", "rev-parse", "side^{tree}") == tree
    assert _git(".", "rev-parse", "side~1") == side_repo


"""Unit tests for tools/scrub_ai_links.py."""

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


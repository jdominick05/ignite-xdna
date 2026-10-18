#!/usr/bin/env bash
# Push the current branch to origin (GitHub, github.com/jdominick05/ignite-xdna,
# the one remote every machine pulls from). Run this after scripts/commit.sh --
# commit.sh never pushes, this is the separate, confirmed step that does.
#
#   ./scripts/push.sh              # push the current branch to origin
#   ./scripts/push.sh --branch main
#   ./scripts/push.sh --no-scrub   # skip AI assistant link scrubbing/verification
#
# Before pushing, verifies that outgoing commits do not contain any links
# for AI assistants (e.g. Claude, Codex, ChatGPT, Antigravity, Gemini).
# Automatically scrubs any AI assistant links from unpushed commits so that
# the public remote never receives AI assistant links.
#
# Refuses if origin is not GitHub: a clone still pointing at the retired GitLab
# remote fixes that with
#   git remote set-url origin https://github.com/jdominick05/ignite-xdna.git
#   git remote remove github     # if it still has the old mirror name
#
# Never force-pushes. A rejected push (another machine pushed first) needs
# `git pull --rebase` (or a fetch + merge) before retrying: sync at the start
# and end of every session.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
NO_SCRUB=0

while [ $# -gt 0 ]; do
    case "$1" in
        --branch)   BRANCH="$2"; shift ;;
        --no-scrub) NO_SCRUB=1 ;;
        -h|--help)  usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *) die "unknown arg $1" ;;
    esac
    shift
done

URL="$(git remote get-url origin 2>/dev/null)" || die "remote 'origin' is not configured (git remote -v)"
case "$URL" in
    *github.com[:/]*) ;;
    *) die "origin is $URL, not GitHub -- see the fix in ./scripts/push.sh --help" ;;
esac

if [ "$NO_SCRUB" -eq 0 ]; then
    step "verifying no AI assistant links in outgoing commits"
    python "$REPO_ROOT/tools/scrub_ai_links.py" --branch "$BRANCH" || die "failed to scrub or verify AI assistant links in outgoing commits"
    ok "verified no AI assistant links in outgoing commits"
fi

step "pushing '$BRANCH' to origin ($URL)"
git push origin "$BRANCH" || die "push to origin failed -- likely a non-fast-forward; fetch/rebase and retry"
ok "pushed '$BRANCH' to origin"

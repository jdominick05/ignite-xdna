#!/usr/bin/env bash
# Push the current branch to both remotes: origin (GitLab, the primary
# multi-machine sync point) and github (public mirror). Run this after
# scripts/commit.sh -- commit.sh never pushes, this is the separate,
# confirmed step that does.
#
#   ./scripts/push.sh              # push current branch to origin, then github
#   ./scripts/push.sh --branch main
#   ./scripts/push.sh --only origin
#   ./scripts/push.sh --only github
#
# Never force-pushes. Fails loudly (and keeps going to try the other remote)
# if either push is rejected -- e.g. another machine pushed to origin first,
# in which case `git pull --rebase` (or a fetch + merge) on origin comes
# before retrying, per CLAUDE.md's "sync at the edges of a session" rule.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
ONLY=""

while [ $# -gt 0 ]; do
    case "$1" in
        --branch) BRANCH="$2"; shift ;;
        --only)   ONLY="$2"; shift ;;
        -h|--help) usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *) die "unknown arg $1" ;;
    esac
    shift
done

case "$ONLY" in
    ""|origin|github) ;;
    *) die "--only must be 'origin' or 'github'" ;;
esac

for r in origin github; do
    git remote get-url "$r" >/dev/null 2>&1 || die "remote '$r' is not configured (git remote -v)"
done

step "pushing '$BRANCH'"
info "origin: $(git remote get-url origin)"
info "github: $(git remote get-url github)"

FAILED=()

push_remote() {
    local remote="$1"
    step "-> $remote"
    if git push "$remote" "$BRANCH"; then
        ok "$remote up to date"
    else
        warn "push to $remote failed -- see above"
        FAILED+=("$remote")
    fi
}

if [ -n "$ONLY" ]; then
    push_remote "$ONLY"
else
    push_remote origin
    push_remote github
fi

if [ "${#FAILED[@]}" -gt 0 ]; then
    die "failed remote(s): ${FAILED[*]} -- likely a non-fast-forward; fetch/rebase and retry"
fi

ok "pushed '$BRANCH' to $( [ -n "$ONLY" ] && echo "$ONLY" || echo "origin and github" )"

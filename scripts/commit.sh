#!/usr/bin/env bash
# Commit staged (or given) files with this repo's required hygiene checks and
# attribution trailer. Never pushes -- push stays a separate, confirmed step.
#
#   ./scripts/commit.sh --subject "..." --body-file /tmp/body.txt \
#       --session-url https://claude.ai/code/session_XXXX \
#       results/foo.log RESEARCH.md
#
#   ./scripts/commit.sh -m "..." -F body.txt -s <url>   # short flags, uses
#                                                        # whatever is already staged
#
# CLAUDE_SESSION_URL can supply --session-url instead of passing it every time.
#
# Checks before committing, all restricted to files under results/ (where
# CLAUDE.md's Files section states the rule):
#   - no embedded NUL bytes -- PowerShell's `*>` writes UTF-16, which makes
#     every later grep/rg silently match nothing
#   - no literal local profile path -- replace with C:\Users\<user> by hand
#     before staging, this script only detects it, it does not rewrite logs
#   - staging a new/changed results/*.log without README.md in the same
#     commit prints a warning (not a block -- a rerun confirming an existing
#     number doesn't need one), per CLAUDE.md's "update README.md before
#     every commit" rule
#
# Positional args (if any) are `git add`-ed by name -- never -A, never `.`.
# With none, whatever is already staged is committed as-is.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

SUBJECT="" BODY_FILE="" SESSION_URL="${CLAUDE_SESSION_URL:-}"
COAUTHOR="Claude Sonnet 5 <noreply@anthropic.com>"
FILES=()

while [ $# -gt 0 ]; do
    case "$1" in
        -m|--subject)     SUBJECT="$2"; shift ;;
        -F|--body-file)   BODY_FILE="$2"; shift ;;
        -s|--session-url) SESSION_URL="$2"; shift ;;
        -c|--coauthor)    COAUTHOR="$2"; shift ;;
        -h|--help)        usage "${BASH_SOURCE[0]}"; exit 0 ;;
        --)               shift; while [ $# -gt 0 ]; do FILES+=("$1"); shift; done; continue ;;
        -*)               die "unknown flag $1" ;;
        *)                FILES+=("$1") ;;
    esac
    shift
done

[ -n "$SUBJECT" ]     || die "need --subject/-m"
[ -n "$SESSION_URL" ] || die "need --session-url/-s (or export CLAUDE_SESSION_URL)"

if [ "${#FILES[@]}" -gt 0 ]; then
    step "staging ${#FILES[@]} file(s)"
    git add -- "${FILES[@]}"
fi

git diff --cached --quiet && die "nothing staged -- pass files to commit, or git add first"

step "checking staged results/ logs are UTF-8 and profile-scrubbed"
UNAME="${USERNAME:-${USER:-}}"
BAD=0
while IFS= read -r f; do
    case "$f" in results/*.log) ;; *) continue ;; esac
    [ -f "$f" ] || continue
    if ! python -c "import sys; sys.exit(1 if b'\x00' in open(sys.argv[1],'rb').read() else 0)" "$f"; then
        warn "$f looks UTF-16 (embedded NUL bytes) -- decode to UTF-8 before committing"
        BAD=1
    fi
    if [ -n "$UNAME" ]; then
        win_pat="$(printf 'Users\\%s\\' "$UNAME")"
        posix_pat="Users/$UNAME/"
        if grep -qF "$win_pat" "$f" 2>/dev/null || grep -qF "$posix_pat" "$f" 2>/dev/null; then
            warn "$f still has the local profile path -- replace with C:\\Users\\<user>"
            BAD=1
        fi
    fi
done < <(git diff --cached --name-only)
[ "$BAD" = 0 ] || die "fix the above, then re-stage and re-run"
ok "staged results/ logs clean"

if git diff --cached --name-only | grep -q '^results/.*\.log$'; then
    if ! git diff --cached --name-only | grep -qx 'README.md'; then
        warn "staging a results/*.log without README.md in this commit -- if this is" \
             "a new finding, retraction, or number that supersedes one already in the" \
             "README, fold it in before committing (CLAUDE.md's maintenance rule)."
    fi
fi

TMPMSG="$(mktemp)"
trap 'rm -f "$TMPMSG"' EXIT
{
    printf '%s\n' "$SUBJECT"
    if [ -n "$BODY_FILE" ]; then
        need_file "$BODY_FILE"
        printf '\n'
        cat "$BODY_FILE"
    fi
    printf '\nCo-Authored-By: %s\nClaude-Session: %s\n' "$COAUTHOR" "$SESSION_URL"
} > "$TMPMSG"

step "committing"
git commit -F "$TMPMSG"
ok "committed, not pushed -- review with 'git log -1', push only after confirming."

#!/usr/bin/env bash
# Commit staged (or given) files with this repo's required hygiene checks and
# attribution trailer. Never pushes -- push stays a separate, confirmed step.
#
#   ./scripts/commit.sh --subject "..." --body-file /tmp/body.txt \
#       --session-id session_XXXX \
#       results/foo.log RESEARCH.md
# Codex sessions use --session-trailer Codex-Session and their own --coauthor.
#
#   ./scripts/commit.sh -m "..." -F body.txt -s <id>   # short flags, uses
#                                                      # whatever is already staged
#
# Session identifiers are optional and never include URLs/links. If a URL is
# supplied via --session-url or CLAUDE_SESSION_URL, it is automatically stripped
# to a raw ID so that no links for AI assistants are ever committed or pushed.
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

SUBJECT="" BODY_FILE="" SESSION_RAW="${CLAUDE_SESSION_URL:-}"
SESSION_TRAILER="Claude-Session"
NO_SESSION=0
NO_COAUTHOR=0
# Override with --coauthor (or CLAUDE_COAUTHOR) when the session is running a
# different model -- the trailer should name the model that actually wrote the
# commit, and a session can switch models partway through.
COAUTHOR="${CLAUDE_COAUTHOR:-Claude Sonnet 5 <noreply@anthropic.com>}"
FILES=()

while [ $# -gt 0 ]; do
    case "$1" in
        -m|--subject)     SUBJECT="$2"; shift ;;
        -F|--body-file)   BODY_FILE="$2"; shift ;;
        -s|--session-url|--session-id|--session) SESSION_RAW="$2"; shift ;;
        --no-session)     NO_SESSION=1 ;;
        --session-trailer) SESSION_TRAILER="$2"; shift ;;
        -c|--coauthor)    COAUTHOR="$2"; shift ;;
        --no-coauthor)    NO_COAUTHOR=1 ;;
        -h|--help)        usage "${BASH_SOURCE[0]}"; exit 0 ;;
        --)               shift; while [ $# -gt 0 ]; do FILES+=("$1"); shift; done; continue ;;
        -*)               die "unknown flag $1" ;;
        *)                FILES+=("$1") ;;
    esac
    shift
done

[ -n "$SUBJECT" ] || die "need --subject/-m"
case "$SESSION_TRAILER" in Claude-Session|Codex-Session|Agy-Session) ;; *) die "unsupported session trailer" ;; esac

# Extract plain session ID and ensure NO AI assistant links/URLs are included
SESSION_ID=""
if [ "$NO_SESSION" -eq 0 ] && [ -n "$SESSION_RAW" ]; then
    SESSION_ID="$(python "$REPO_ROOT/tools/scrub_ai_links.py" --extract-id "$SESSION_RAW")"
fi

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
    if [ "$NO_COAUTHOR" -eq 0 ] && [ -n "$COAUTHOR" ]; then
        printf '\nCo-Authored-By: %s\n' "$COAUTHOR"
    fi
    if [ "$NO_SESSION" -eq 0 ] && [ -n "$SESSION_ID" ]; then
        printf '%s: %s\n' "$SESSION_TRAILER" "$SESSION_ID"
    fi
} > "$TMPMSG"

# Safety guarantee: scrub any stray AI assistant links
python "$REPO_ROOT/tools/scrub_ai_links.py" --clean-file "$TMPMSG"

step "committing"
git commit -F "$TMPMSG"
ok "committed, not pushed -- review with 'git log -1', push only after confirming."

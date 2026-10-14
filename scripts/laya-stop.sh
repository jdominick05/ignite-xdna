#!/usr/bin/env bash
# Stop the Laya decision daemon started by ./scripts/laya-start.sh, and confirm the NPU is released.
#
#   ./scripts/laya-stop.sh              # the daemon on 127.0.0.1:8765
#   ./scripts/laya-stop.sh --port 8766
#
# Asks the daemon to stop (it closes its NPU context and exits), waits up to 60 s, and only then ends the
# pid it recorded in scratch/laya/daemon_<port>.pid -- that process tree and nothing else. Finally checks
# that xrt-smi reports no hardware contexts.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

PORT=8765
while [ $# -gt 0 ]; do
    case "$1" in
        --port)    PORT="$2"; shift ;;
        -h|--help) usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *)         die "unknown flag $1" ;;
    esac
    shift
done
PIDF=scratch/laya/daemon_$PORT.pid
alive() { [ -n "${pid:-}" ] && tasklist //FI "PID eq $pid" 2>/dev/null | grep -qw "$pid"; }

step "Stopping the Laya daemon on 127.0.0.1:$PORT"
pid=$(cat "$PIDF" 2>/dev/null || true)
use_env mlir-aie-iron >/dev/null
if python tools/laya_daemon.py stop --port "$PORT" >/dev/null 2>&1; then
    info "stop requested"
else
    info "no daemon answered on port $PORT"
fi
for _ in $(seq 1 60); do alive || break; sleep 1; done
if alive; then
    warn "pid $pid did not exit after 60 s; ending its process tree"
    taskkill //PID "$pid" //T //F >/dev/null 2>&1 || true
fi
rm -f "$PIDF"

smi=/c/Windows/System32/AMD/xrt-smi.exe
if "$smi" examine -r aie-partitions 2>/dev/null | grep -q "No hardware contexts running"; then
    ok "stopped; the NPU is idle"
else
    warn "xrt-smi still shows a hardware context (another session's, or a slot still being recycled)"
fi

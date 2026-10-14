#!/usr/bin/env bash
# Start the Laya decision daemon (tools/laya_daemon.py) in the background, on the NPU by default.
#
#   ./scripts/laya-start.sh                    # NPU hybrid on 127.0.0.1:8765
#   ./scripts/laya-start.sh --backend cpu      # or dml; cpu/dml run in ryzen-ai-1.8.0
#   ./scripts/laya-start.sh --port 8766
#
# The NPU backend (tools/laya_hybrid.py) runs in mlir-aie-iron and holds ONE hardware context until
# ./scripts/laya-stop.sh, so it refuses to start unless xrt-smi reports the NPU idle, and any NPU
# measurement needs it stopped first. The daemon is detached: it outlives this shell. Its output goes to
# scratch/laya/daemon_<port>.log and the pid of its launcher (cmd.exe, parent of the python daemon) to
# scratch/laya/daemon_<port>.pid; all git-ignored.
# Needs the model in scratch/laya/laya-onnx and, for npu, the GEMM builds in scratch/gemm_generic/aot6.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

BACKEND=npu PORT=8765
while [ $# -gt 0 ]; do
    case "$1" in
        --backend) BACKEND="$2"; shift ;;
        --port)    PORT="$2"; shift ;;
        -h|--help) usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *)         die "unknown flag $1" ;;
    esac
    shift
done
case "$BACKEND" in npu|cpu|dml) ;; *) die "--backend must be npu, cpu or dml" ;; esac

MODEL_DIR=scratch/laya/laya-onnx
LOG=scratch/laya/daemon_$PORT.log
PIDF=scratch/laya/daemon_$PORT.pid
need_file "$MODEL_DIR/laya_az0_noguard.onnx" "fetch receptron/laya-onnx and run tools/laya_npu_screen.py noguard"
[ "$BACKEND" = npu ] && need_file scratch/gemm_generic/aot6/g_1024_3072.xclbin "build with tools/gemm_generic_probe.py compile"

step "Laya daemon: $BACKEND on 127.0.0.1:$PORT"
if [ -f "$PIDF" ] && tasklist //FI "PID eq $(cat "$PIDF")" 2>/dev/null | grep -qw "$(cat "$PIDF")"; then
    die "already running (pid $(cat "$PIDF")); stop it with ./scripts/laya-stop.sh --port $PORT"
fi
if [ "$BACKEND" = npu ]; then
    smi=/c/Windows/System32/AMD/xrt-smi.exe
    "$smi" examine -r aie-partitions 2>/dev/null | grep -q "No hardware contexts running" \
        || die "the NPU is not idle (xrt-smi shows a hardware context); refusing to share it"
    use_env mlir-aie-iron
else
    use_env ryzen-ai-1.8.0
fi

mkdir -p scratch/laya
rm -f "$LOG" "$LOG.err"
# Launched through WMI (Win32_Process.Create), so the daemon inherits no handle from this shell: a child that
# inherited the stdout pipe of $(...) kept the capture open forever (2026-10-14). WMI gives it the system
# environment, so the launcher sets the activated env's PATH itself.
CMDF=scratch/laya/daemon_$PORT.cmd
envw=$(cygpath -w "$CONDA_PREFIX")
printf '@echo off\r\ncd /d "%s"\r\nset "PATH=%s;%s\\Library\\bin;%s\\Scripts;%%PATH%%"\r\n"%s\\python.exe" tools\\laya_daemon.py serve --backend %s --model-dir %s --port %s > "%s" 2> "%s"\r\n' \
    "$(cygpath -w "$REPO_ROOT")" "$envw" "$envw" "$envw" "$envw" "$BACKEND" "$(cygpath -w "$MODEL_DIR")" "$PORT" \
    "$(cygpath -w "$LOG")" "$(cygpath -w "$LOG.err")" > "$CMDF"
pid=$(powershell.exe -NoProfile -Command \
    "(Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd.exe /c \"$(cygpath -w "$REPO_ROOT/$CMDF")\"'; CurrentDirectory='$(cygpath -w "$REPO_ROOT")'}).ProcessId" \
    </dev/null 2>/dev/null | tr -d '\r')
[ -n "$pid" ] || die "could not start the daemon"
echo "$pid" > "$PIDF"
info "pid $pid; log $LOG"

for _ in $(seq 1 120); do
    grep -q "listening" "$LOG" 2>/dev/null && break
    if ! tasklist //FI "PID eq $pid" 2>/dev/null | grep -qw "$pid"; then
        tail -20 "$LOG" "$LOG.err" 2>/dev/null
        rm -f "$PIDF"
        die "the daemon exited during start-up"
    fi
    sleep 1
done
grep -q "listening" "$LOG" || die "no 'listening' after 120 s; see $LOG and $LOG.err"
info "$(grep '^\[serve\]' "$LOG" | head -1)"
ok "Laya is up on 127.0.0.1:$PORT ($BACKEND). Stop it with ./scripts/laya-stop.sh${PORT:+ --port $PORT}"

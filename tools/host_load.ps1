# Host contention witness: what else was running while a measurement was taken.
#
#   powershell -NoProfile -File tools/host_load.ps1                     # one-shot, parseable
#   powershell -NoProfile -File tools/host_load.ps1 -Out results/.../load_<tag>_<role>.log
#   powershell -NoProfile -File tools/host_load.ps1 -Watch -Out <file> -IntervalSec 10
#
# This is the host-side counterpart to the xrt-smi check in scripts/lib.sh. That one
# answers "is another session on the AIE tiles"; this one answers "is another session
# on the CPU". Desktop 2 runs three Claude sessions, agy and PyCharm against the same
# 16 threads, so a wall-time or peak-memory figure taken without a witness is a guess
# -- which is exactly how the 2026-09-08 MODNet AdaRound oracle started.
#
# PowerShell rather than Python on purpose: the guard runs before `conda activate`,
# and the base interpreter here has no numpy, let alone psutil.
#
# One-shot output, one record per line, stable enough for lib.sh to parse:
#
#   HOST_LOAD busy_cores=<f> logical_cores=<n> free_ram_gb=<f> sample_sec=<f>
#   HOST_LOAD_PEER <cores> <name>:<pid> <reason> <command line, truncated>
#   HOST_LOAD_TOP <cores> <name>:<pid>
#   HOST_LOAD_VERDICT CLEAR|BUSY|PEER
#
# PEER outranks BUSY. A peer is a build tool, one of this repo's own heavy producers,
# or -- the case that catches whatever nobody thought to name -- any single process
# holding CORES_PEER cores or more.
param(
    [string]$Out = "",
    [switch]$Watch,
    [int]$IntervalSec = 10,
    [double]$SampleSec = 1.0
)

$ErrorActionPreference = "Stop"
$CORES_PEER = 2.0          # one process at or above this is a heavy peer whatever it is
$CMD_CHARS  = 140          # command lines are truncated to keep a witness line readable

# Build tools, by process name. Anything that forks a compiler lands here through its
# children, so agy or a Claude session running a build is caught by what it spawned.
$PEER_NAMES = '^(cl|link|lib|ninja|cmake|make|nmake|msbuild|devenv|gcc|g\+\+|cc1|cc1plus|' +
              'clang|clang\+\+|lld|lld-link|ld|rustc|cargo|go|javac|node|tsc|' +
              'xchesscc|xchesscc_wrapper|aie-opt|aie-translate|llc|opt|nvcc|hipcc)$'

# This repo's own heavy producers, by command-line token. Deliberately specific: the
# repository directory is named ...-quantization, so a bare "quant" would match every
# python process started from an absolute path in the tree.
$PEER_CMDS = '(3d_quantize_compare\.py|3_quantize\.py|3b_quantize_cut\.py|' +
             '1_export\.py|-m\s+quant\b|quant\.__main__|quant\.adaround|' +
             'build_hwinfo_bridge|setup\.py\s+build)'

$enc = New-Object System.Text.UTF8Encoding($false)
function Emit([string]$line) {
    Write-Output $line
    if ($Out -ne "") { [System.IO.File]::AppendAllText($Out, $line + "`r`n", $enc) }
}

function Get-CpuTimes {
    $t = @{}
    foreach ($p in Get-Process) { if ($null -ne $p.CPU) { $t[$p.Id] = $p.CPU } }
    return $t
}

# Sample per-process CPU over an interval and return one row per busy process.
function Get-Sample([double]$seconds) {
    $before = Get-CpuTimes
    $t0 = Get-Date
    Start-Sleep -Milliseconds ([int]($seconds * 1000))
    $dt = ((Get-Date) - $t0).TotalSeconds
    if ($dt -le 0) { $dt = $seconds }
    $rows = @()
    foreach ($p in Get-Process) {
        if ($null -eq $p.CPU) { continue }
        $base = 0
        if ($before.ContainsKey($p.Id)) { $base = $before[$p.Id] }
        $d = $p.CPU - $base
        if ($d -le 0) { continue }
        $rows += [pscustomobject]@{
            Name  = $p.ProcessName
            Id    = $p.Id
            Cores = [math]::Round($d / $dt, 2)
            WsMb  = [math]::Round($p.WorkingSet64 / 1MB, 0)
        }
    }
    return ,($rows | Sort-Object Cores -Descending)
}

# Witness files are committed as evidence, and every log under results/ carries
# C:\Users\<user> rather than the profile that produced it. Redact here rather than
# leaving it to whoever stages the file: scripts/commit.sh would reject it, but only
# after the run that wrote it is long finished.
# Three separator forms reach a command line here: a Windows path from the exe itself,
# a forward-slash Windows path from a script argument, and Git Bash's /c/Users/ form.
function Redact([string]$s) {
    if ($null -eq $s) { return "" }
    $s = $s -replace '(?i)([A-Za-z]:[\\/]Users[\\/])[^\\/\s"]+', '${1}<user>'
    $s = $s -replace '(?i)(/[A-Za-z]/Users/)[^/\s"]+', '${1}<user>'
    return $s
}

function Get-FreeRamGb {
    return [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1)
}

$cores = [int]$env:NUMBER_OF_PROCESSORS
if ($cores -le 0) { $cores = 1 }

if ($Watch) {
    if ($Out -eq "") { throw "-Watch needs -Out <file>" }
    # Append, never truncate: the one-shot guard has usually already written its
    # snapshot here, and one witness per run is easier to read back than two.
    [System.IO.File]::AppendAllText($Out,
        "# host load witness  interval=${IntervalSec}s  logical_cores=$cores  host=$env:COMPUTERNAME`r`n" +
        "# utc_iso  busy_cores  free_ram_gb  top5(name:pid=cores)`r`n", $enc)
    while ($true) {
        $rows = Get-Sample $IntervalSec
        $sum = ($rows | Measure-Object Cores -Sum).Sum
        if ($null -eq $sum) { $sum = 0 }
        $top = ($rows | Select-Object -First 5 | ForEach-Object { "$($_.Name):$($_.Id)=$($_.Cores)" }) -join ' '
        $line = "{0}  {1,6}  {2,6}  {3}" -f (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'),
                                            [math]::Round($sum, 2), (Get-FreeRamGb), $top
        [System.IO.File]::AppendAllText($Out, $line + "`r`n", $enc)
    }
    exit 0
}

# --- one-shot ------------------------------------------------------------
$rows = Get-Sample $SampleSec
$sum = ($rows | Measure-Object Cores -Sum).Sum
if ($null -eq $sum) { $sum = 0 }

# Command lines, for the peer classification only. Win32_Process rather than tasklist:
# every producer here is python.exe, so the name alone cannot tell this session's
# oracle from another session's, and the argument list is the only discriminator.
$cmds = @{}
try {
    foreach ($p in Get-CimInstance Win32_Process) { $cmds[[int]$p.ProcessId] = $p.CommandLine }
} catch { }

Emit ("HOST_LOAD busy_cores={0} logical_cores={1} free_ram_gb={2} sample_sec={3}" -f `
      [math]::Round($sum, 2), $cores, (Get-FreeRamGb), $SampleSec)

$peers = 0
foreach ($r in $rows) {
    $cmd = ""
    if ($cmds.ContainsKey($r.Id)) { if ($null -ne $cmds[$r.Id]) { $cmd = Redact ($cmds[$r.Id] -replace '\s+', ' ') } }
    # Never report the guard, its own watcher, or the shell that launched them.
    if ($r.Id -eq $PID) { continue }
    if ($cmd -match 'host_load\.ps1') { continue }

    $reason = ""
    if ($r.Name -match $PEER_NAMES)    { $reason = "build-tool" }
    elseif ($cmd -match $PEER_CMDS)    { $reason = "repo-producer" }
    elseif ($r.Cores -ge $CORES_PEER)  { $reason = "cpu-heavy" }
    if ($reason -eq "") { continue }

    if ($cmd.Length -gt $CMD_CHARS) { $cmd = $cmd.Substring(0, $CMD_CHARS) + "..." }
    if ($cmd -eq "") { $cmd = "(command line unavailable)" }
    Emit ("HOST_LOAD_PEER {0} {1}:{2} {3} {4}" -f $r.Cores, $r.Name, $r.Id, $reason, $cmd)
    $peers++
}

foreach ($r in ($rows | Select-Object -First 5)) {
    Emit ("HOST_LOAD_TOP {0} {1}:{2} ws_mb={3}" -f $r.Cores, $r.Name, $r.Id, $r.WsMb)
}

$limit = [double]$env:HOST_LOAD_MAX_CORES
if ($limit -le 0) { $limit = [math]::Max(2.0, $cores * 0.25) }

if ($peers -gt 0)      { Emit "HOST_LOAD_VERDICT PEER";  exit 0 }
elseif ($sum -gt $limit) { Emit "HOST_LOAD_VERDICT BUSY"; exit 0 }
else                   { Emit "HOST_LOAD_VERDICT CLEAR"; exit 0 }

# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
<#
Read-only preflight for local inference work. Never stops another process.
Use -RequireIdle before performance work; exit 2 means busy or unverified.
#>
[CmdletBinding()]
param(
    [switch]$RequireIdle,
    [ValidateRange(250, 5000)][int]$SampleMilliseconds = 500
)
$ErrorActionPreference = 'Stop'
$blockers = [System.Collections.Generic.List[string]]::new()
$candidates = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^(llama.*|python.*|onnx.*|ollama.*|vllm.*)\.exe$'
})
$before = @{}
foreach ($item in $candidates) {
    try { $before[$item.ProcessId] = (Get-Process -Id $item.ProcessId).TotalProcessorTime.TotalSeconds }
    catch { }
}
$watch = [System.Diagnostics.Stopwatch]::StartNew()
Start-Sleep -Milliseconds $SampleMilliseconds
$elapsed = $watch.Elapsed.TotalSeconds
$cpu = @(foreach ($item in $candidates) {
    try {
        $process = Get-Process -Id $item.ProcessId
        if (-not $before.ContainsKey($item.ProcessId)) { continue }
        $cores = [Math]::Max(0, ($process.TotalProcessorTime.TotalSeconds - $before[$item.ProcessId]) / $elapsed)
        if ($cores -ge 0.1) { $blockers.Add("CPU inference work: PID $($item.ProcessId)") }
        [pscustomobject]@{ pid = $item.ProcessId; name = $item.Name; active_cpu_cores = [Math]::Round($cores, 3) }
    } catch { }
})
$gpu = @()
$gpuKnown = $false
try {
    # Include all busy GPU engines: graphics or unrelated compute can contend
    # with inference too. Command lines and user prompts are never printed.
    $engines = @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine)
    $gpuKnown = $engines.Count -gt 0
    $gpu = @($engines | Where-Object { [double]$_.UtilizationPercentage -ge 1 } | ForEach-Object {
        $enginePid = $null
        if ($_.Name -match 'pid_(\d+)_') { $enginePid = [int]$Matches[1] }
        [pscustomobject]@{ pid = $enginePid; engine = $_.Name; utilization_percent = [double]$_.UtilizationPercentage }
    })
    if ($gpu.Count) { $blockers.Add('GPU engine activity observed') }
} catch { $gpuKnown = $false }
if (-not $gpuKnown) { $blockers.Add('GPU engine telemetry unavailable') }
$npuKnown = $false
$npuIdle = $false
$xrtPath = Join-Path $env:WINDIR 'System32\AMD\xrt-smi.exe'
try {
    if (Test-Path -LiteralPath $xrtPath) {
        $witness = (& $xrtPath examine -r aie-partitions 2>&1 | Out-String)
        $npuKnown = $LASTEXITCODE -eq 0 -and $witness -match 'AIE Partitions'
        $npuIdle = $npuKnown -and $witness -match 'No hardware contexts running'
    }
} catch { $npuKnown = $false }
if (-not $npuKnown) { $blockers.Add('NPU context telemetry unavailable') }
elseif (-not $npuIdle) { $blockers.Add('NPU contexts active') }
[pscustomobject]@{
    utc = [DateTime]::UtcNow.ToString('o')
    sample_seconds = [Math]::Round($elapsed, 3)
    inference_processes = $cpu
    gpu_telemetry_known = $gpuKnown
    busy_gpu_engines = $gpu
    npu_telemetry_known = $npuKnown
    npu_idle = $npuIdle
    idle_verified = $blockers.Count -eq 0
    blockers = @($blockers)
    scope = 'Preflight snapshot only; bounded runs must continue monitoring contention.'
} | ConvertTo-Json -Depth 5
if ($RequireIdle -and $blockers.Count) { exit 2 }

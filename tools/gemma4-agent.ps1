# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
<# Bounded native Gemma tools: arithmetic and reads in an explicit workspace. #>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Workspace,
    [string]$Task,
    [string]$TaskFile,
    [string]$Artifacts = (Join-Path $env:USERPROFILE 'gemma4-assistant'),
    [string]$LlamaRuntime = (Join-Path $env:USERPROFILE 'tools\llamacpp-b1343-rocm'),
    [string]$NpuPython = (Join-Path $env:USERPROFILE 'miniforge3\envs\mlir-aie-iron\python.exe'),
    [string]$TokenizerPython = (Join-Path $env:USERPROFILE 'gemma4-assistant\conversion-venv\Scripts\python.exe'),
    [string]$Hub = 'L:\huggingface\hub',
    [string]$GpuActivityHelper
)
$ErrorActionPreference = 'Stop'
if ($PSBoundParameters.ContainsKey('Task') -and $PSBoundParameters.ContainsKey('TaskFile')) { throw 'Use Task or TaskFile, not both' }
$workspacePath = (Resolve-Path -LiteralPath $Workspace).ProviderPath
if ($PSBoundParameters.ContainsKey('TaskFile')) { $TaskFile = (Resolve-Path -LiteralPath $TaskFile).ProviderPath }
if ($PSBoundParameters.ContainsKey('GpuActivityHelper')) { $GpuActivityHelper = (Resolve-Path -LiteralPath $GpuActivityHelper).ProviderPath }
& (Join-Path $PSScriptRoot 'check-inference-contention.ps1')
$snapshotPath = Join-Path $Hub 'models--google--gemma-4-E2B-it\snapshots\3e22461f65e89153144f8adb70e3b8c2cc9845a7'
$agentArguments = @((Join-Path $PSScriptRoot 'gemma4_agent.py'), '--artifacts', $Artifacts,
    '--runtime', $LlamaRuntime, '--tokenizer-python', $TokenizerPython, '--snapshot', $snapshotPath,
    '--workspace', $workspacePath)
if ($PSBoundParameters.ContainsKey('Task')) { $agentArguments += @('--task', $Task) }
if ($PSBoundParameters.ContainsKey('TaskFile')) { $agentArguments += @('--task-file', $TaskFile) }
if ($PSBoundParameters.ContainsKey('GpuActivityHelper')) { $agentArguments += @('--gpu-activity-helper', $GpuActivityHelper) }
Push-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
try {
    & $NpuPython @agentArguments
    if ($LASTEXITCODE -ne 0) { throw "Gemma agent exited with code $LASTEXITCODE" }
} finally { Pop-Location }

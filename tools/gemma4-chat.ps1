# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
<# Launch the experimental local E2B GPU/NPU terminal assistant. #>
[CmdletBinding()]
param(
    [string]$Artifacts = (Join-Path $env:USERPROFILE 'gemma4-assistant'),
    [string]$LlamaRuntime = (Join-Path $env:USERPROFILE 'tools\llamacpp-b1343-rocm'),
    [string]$NpuPython = (Join-Path $env:USERPROFILE 'miniforge3\envs\mlir-aie-iron\python.exe'),
    [string]$TokenizerPython = (Join-Path $env:USERPROFILE 'gemma4-assistant\conversion-venv\Scripts\python.exe'),
    [string]$Hub = 'L:\huggingface\hub',
    [ValidateRange(1, 64)][int]$MaxNewTokens = 64,
    [ValidateRange(1, 4)][int]$Depth = 2,
    [ValidateRange(0, 60)][int]$AdmissionWaitSeconds = 0,
    [ValidateRange(1, 2)][int]$InputDepth = 1,
    [switch]$VectorCopy,
    [switch]$PrefixBinding,
    [string]$GpuActivityHelper,
    [ValidateSet(512, 2560)][int]$ContextTokens = 512,
    [string]$Prompt,
    [string]$PromptFile
)
$ErrorActionPreference = 'Stop'
if ($PSBoundParameters.ContainsKey('Prompt') -and $PSBoundParameters.ContainsKey('PromptFile')) {
    throw 'Use Prompt or PromptFile, not both'
}
if ($PSBoundParameters.ContainsKey('PromptFile')) { $PromptFile = (Resolve-Path -LiteralPath $PromptFile).ProviderPath }
if ($PSBoundParameters.ContainsKey('GpuActivityHelper')) { $GpuActivityHelper = (Resolve-Path -LiteralPath $GpuActivityHelper).ProviderPath }
$studyRoot = Split-Path -Parent $PSScriptRoot
# Read-only snapshot before starting; runtime repeats admission per request.
& (Join-Path $PSScriptRoot 'check-inference-contention.ps1')
$snapshotPath = Join-Path $Hub 'models--google--gemma-4-E2B-it\snapshots\3e22461f65e89153144f8adb70e3b8c2cc9845a7'
$chatArguments = @((Join-Path $PSScriptRoot 'gemma4_chat.py'), '--artifacts', $Artifacts,
    '--llama-runtime', $LlamaRuntime, '--tokenizer-python', $TokenizerPython,
    '--snapshot', $snapshotPath, '--max-new-tokens', $MaxNewTokens, '--depth', $Depth,
    '--input-depth', $InputDepth, '--context-tokens', $ContextTokens,
    '--admission-wait', $AdmissionWaitSeconds)
if ($PSBoundParameters.ContainsKey('Prompt')) { $chatArguments += @('--prompt', $Prompt) }
if ($PSBoundParameters.ContainsKey('PromptFile')) { $chatArguments += @('--prompt-file', $PromptFile) }
if ($VectorCopy) { $chatArguments += '--vector-copy' }
if ($PrefixBinding) { $chatArguments += '--prefix-binding' }
if ($PSBoundParameters.ContainsKey('GpuActivityHelper')) { $chatArguments += @('--gpu-activity-helper', $GpuActivityHelper) }
Push-Location -LiteralPath $studyRoot
try {
    & $NpuPython @chatArguments
    if ($LASTEXITCODE -ne 0) { throw "Gemma chat exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}

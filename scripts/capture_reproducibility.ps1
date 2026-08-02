[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$MiniCPMModelDir,

    [Parameter(Mandatory = $true)]
    [string]$TransformerCheckpoint,

    [string]$ResNetCheckpoint = "",
    [string]$Venv = ".venv-minicpm-torch280",
    [string]$AutoGPTQSource = ".deps\AutoGPTQ-minicpmo",
    [string]$OutputDir = "",
    [switch]$AllowBaselineDifferences
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Resolve-RepositoryPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [switch]$Directory
    )
    $candidate = if ([System.IO.Path]::IsPathRooted($Path)) {
        $Path
    } else {
        Join-Path $repoRoot $Path
    }
    $resolved = (Resolve-Path -LiteralPath $candidate).Path
    $kind = if ($Directory) { "Container" } else { "Leaf" }
    if (-not (Test-Path -LiteralPath $resolved -PathType $kind)) {
        throw "$Label has the wrong path type: $resolved"
    }
    return $resolved
}

$venvPath = if ([System.IO.Path]::IsPathRooted($Venv)) {
    $Venv
} else {
    Join-Path $repoRoot $Venv
}
$python = Join-Path $venvPath "Scripts/python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Virtual environment Python not found: $python"
}
$modelDir = Resolve-RepositoryPath -Path $MiniCPMModelDir -Label "MiniCPM snapshot" -Directory
$transformer = Resolve-RepositoryPath `
    -Path $TransformerCheckpoint `
    -Label "Transformer checkpoint"
$autoGptq = Resolve-RepositoryPath `
    -Path $AutoGPTQSource `
    -Label "AutoGPTQ source" `
    -Directory

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $outputPath = Join-Path $repoRoot "var\reproducibility-$timestamp"
} elseif ([System.IO.Path]::IsPathRooted($OutputDir)) {
    $outputPath = [System.IO.Path]::GetFullPath($OutputDir)
} else {
    $outputPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputDir))
}
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null

$arguments = @(
    (Join-Path $PSScriptRoot "capture_reproducibility.py"),
    "--output-dir", $outputPath,
    "--baseline", (Join-Path $repoRoot "reproducibility/windows-cuda-baseline.json"),
    "--minicpm-model-dir", $modelDir,
    "--transformer-checkpoint", $transformer,
    "--autogptq-source", $autoGptq
)
if (-not [string]::IsNullOrWhiteSpace($ResNetCheckpoint)) {
    $resnet = Resolve-RepositoryPath -Path $ResNetCheckpoint -Label "ResNet checkpoint"
    $arguments += @("--resnet-checkpoint", $resnet)
}
if (-not $AllowBaselineDifferences) {
    $arguments += "--strict"
}

Push-Location $repoRoot
try {
    & $python @arguments
    $captureExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
if ($captureExitCode -ne 0) {
    throw "Reproducibility capture found baseline differences; see $outputPath"
}

Write-Host "Reproducibility evidence captured in: $outputPath"
Write-Host "  environment-report.json"
Write-Host "  pip-freeze.txt"
Write-Host "  requirements-lock-candidate.txt"

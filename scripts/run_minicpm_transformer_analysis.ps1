[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$MiniCPMModelDir,

    [Parameter(Mandatory = $true)]
    [string]$TransformerCheckpoint,

    [Parameter(Mandatory = $true)]
    [string]$InputImage,

    [string]$Venv = ".venv-minicpm-torch280",
    [string]$Device = "cuda:0",
    [ValidateRange(1, 100)]
    [int]$Repeats = 5,
    [ValidateRange(0, 16384)]
    [double]$MemoryGrowthLimitMB = 64,
    [string]$OutputDir = "",
    [switch]$AllowUnverifiedModelCode,
    [switch]$SkipEnvironmentVerification
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Resolve-InputPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [switch]$Directory
    )

    $candidate = if ([System.IO.Path]::IsPathRooted($Path)) {
        $Path
    }
    else {
        Join-Path $repoRoot $Path
    }
    $resolved = (Resolve-Path -LiteralPath $candidate).Path
    $expectedType = if ($Directory) { "Container" } else { "Leaf" }
    if (-not (Test-Path -LiteralPath $resolved -PathType $expectedType)) {
        throw "$Label has the wrong path type: $resolved"
    }
    return $resolved
}

$venvPath = if ([System.IO.Path]::IsPathRooted($Venv)) {
    $Venv
}
else {
    Join-Path $repoRoot $Venv
}
$python = Join-Path $venvPath "Scripts/python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Virtual environment Python not found: $python"
}

$modelDir = Resolve-InputPath -Path $MiniCPMModelDir -Label "MiniCPM model" -Directory
$checkpoint = Resolve-InputPath -Path $TransformerCheckpoint -Label "Transformer checkpoint"
$image = Resolve-InputPath -Path $InputImage -Label "Input image"

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $outputPath = Join-Path $repoRoot "var\autogptq-baseline-$timestamp"
}
elseif ([System.IO.Path]::IsPathRooted($OutputDir)) {
    $outputPath = [System.IO.Path]::GetFullPath($OutputDir)
}
else {
    $outputPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputDir))
}
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null

if (-not $SkipEnvironmentVerification) {
    & (Join-Path $PSScriptRoot "verify_minicpm_env.ps1") -Venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        throw "MiniCPM environment verification failed"
    }
}

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"

& $python -m pip freeze --all |
    Set-Content -LiteralPath (Join-Path $outputPath "pip-freeze.txt") -Encoding UTF8
if ($LASTEXITCODE -ne 0) {
    throw "Could not capture pip freeze"
}

git -C $repoRoot rev-parse HEAD |
    Set-Content -LiteralPath (Join-Path $outputPath "git-commit.txt") -Encoding UTF8
git -C $repoRoot status --short --branch |
    Set-Content -LiteralPath (Join-Path $outputPath "git-status.txt") -Encoding UTF8

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi --query-gpu=name,memory.total,driver_version `
        --format=csv,noheader |
        Set-Content -LiteralPath (Join-Path $outputPath "nvidia-smi.txt") -Encoding UTF8
}

$arguments = @(
    (Join-Path $PSScriptRoot "analyze_minicpm_transformer.py"),
    "--input", $image,
    "--minicpm-model-dir", $modelDir,
    "--transformer", $checkpoint,
    "--device", $Device,
    "--repeats", $Repeats.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--memory-growth-limit-mb",
    $MemoryGrowthLimitMB.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--output-dir", $outputPath
)
if ($AllowUnverifiedModelCode) {
    $arguments += "--allow-unverified-model-code"
}

& $python @arguments
$analysisExitCode = $LASTEXITCODE
$reportPath = Join-Path $outputPath "report.json"
if (Test-Path -LiteralPath $reportPath -PathType Leaf) {
    $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
    Write-Host "Status: $($report.status)"
    Write-Host "Stage: $($report.stage)"
    if ($report.status -eq "passed") {
        Write-Host "Load: $($report.checks.model_loading.elapsed_ms) ms"
        Write-Host "First inference: $($report.checks.first_full_inference.elapsed_ms) ms"
        Write-Host "Repeat mean: $($report.checks.repeated_inference.timing_summary.mean_ms) ms"
        Write-Host (
            "Peak allocated VRAM: " +
            "$($report.checks.cuda_memory.maximum_peak_allocated_vram_mb) MiB"
        )
    }
}

if ($analysisExitCode -ne 0) {
    throw "MiniCPM + Transformer analysis failed; see $reportPath"
}

Write-Host "MiniCPM + Transformer analysis passed: $reportPath"

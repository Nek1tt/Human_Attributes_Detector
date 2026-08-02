[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Video,
    [string]$Yolo = "",
    [string]$ResNetCheckpoint = "",
    [string]$TransformerCheckpoint = "",
    [string]$Venv = ".venv-cpu-py311",
    [string]$Device = "cpu",
    [string]$OutputDir = "",
    [int]$Timeout = 3600,
    [switch]$WithUnitTests,
    [switch]$WithRuff,
    [switch]$NoBundle
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputDir = Join-Path $repoRoot "var\video-tests\$timestamp-resnet"
}

$parameters = @{
    Backend = "resnet"
    Video = $Video
    Yolo = $Yolo
    ResNetCheckpoint = $ResNetCheckpoint
    TransformerCheckpoint = $TransformerCheckpoint
    Venv = $Venv
    Device = $Device
    OutputDir = $OutputDir
    Timeout = $Timeout
}
if (-not $WithUnitTests) { $parameters["SkipUnit"] = $true }
if ($WithRuff) { $parameters["WithRuff"] = $true }

& (Join-Path $PSScriptRoot "run_video_test.ps1") @parameters
if (-not $NoBundle) {
    & (Join-Path $PSScriptRoot "export_validation_bundle.ps1") `
        -EvidenceDir $OutputDir
}


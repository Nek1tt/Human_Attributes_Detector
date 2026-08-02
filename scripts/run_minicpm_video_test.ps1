[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Video,
    [Parameter(Mandatory = $true)]
    [string]$MiniCPMModelDir,
    [Parameter(Mandatory = $true)]
    [string]$TransformerCheckpoint,
    [string]$Yolo = "",
    [string]$Venv = ".venv-minicpm-torch280",
    [string]$Device = "cuda",
    [string]$OutputDir = "",
    [int]$Timeout = 7200,
    [switch]$WithUnitTests,
    [switch]$WithRuff,
    [switch]$AllowUnverifiedModelCode,
    [switch]$NoBundle
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputDir = Join-Path $repoRoot "var\video-tests\$timestamp-minicpm"
}

$parameters = @{
    Backend = "minicpm"
    Video = $Video
    Yolo = $Yolo
    MiniCPMModelDir = $MiniCPMModelDir
    TransformerCheckpoint = $TransformerCheckpoint
    Venv = $Venv
    Device = $Device
    OutputDir = $OutputDir
    Timeout = $Timeout
}
if (-not $WithUnitTests) { $parameters["SkipUnit"] = $true }
if ($WithRuff) { $parameters["WithRuff"] = $true }
if ($AllowUnverifiedModelCode) { $parameters["AllowUnverifiedModelCode"] = $true }

& (Join-Path $PSScriptRoot "run_video_test.ps1") @parameters
if (-not $NoBundle) {
    & (Join-Path $PSScriptRoot "export_validation_bundle.ps1") `
        -EvidenceDir $OutputDir
}


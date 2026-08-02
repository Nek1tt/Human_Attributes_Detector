[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$MiniCPMModelDir,

    [string]$Venv = ".venv-minicpm-torch280",
    [switch]$Download
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$baselinePath = Join-Path $repoRoot "reproducibility/windows-cuda-baseline.json"
$baseline = Get-Content -LiteralPath $baselinePath -Raw | ConvertFrom-Json
$venvPath = if ([System.IO.Path]::IsPathRooted($Venv)) {
    $Venv
} else {
    Join-Path $repoRoot $Venv
}
$python = Join-Path $venvPath "Scripts/python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Virtual environment Python not found: $python"
}
$modelDir = if ([System.IO.Path]::IsPathRooted($MiniCPMModelDir)) {
    [System.IO.Path]::GetFullPath($MiniCPMModelDir)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $repoRoot $MiniCPMModelDir))
}

$arguments = @(
    "-m", "silhouette_detector.model_store",
    "--repo-id", [string]$baseline.minicpm.repository,
    "--revision", [string]$baseline.minicpm.revision,
    "--destination", $modelDir,
    "--patch-resampler-list-import"
)
if (-not $Download) {
    $arguments += "--existing-snapshot"
}

Push-Location $repoRoot
try {
    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Could not prepare the pinned MiniCPM snapshot."
    }
} finally {
    Pop-Location
}

$manifestPath = Join-Path $modelDir "model-manifest.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.repo_id -ne $baseline.minicpm.repository) {
    throw "MiniCPM repository mismatch in $manifestPath"
}
if ($manifest.resolved_revision -ne $baseline.minicpm.revision) {
    throw "MiniCPM revision mismatch in $manifestPath"
}
if ($manifest.config_sha256 -ne $baseline.minicpm.config_sha256) {
    throw "MiniCPM config.json SHA256 does not match the validated baseline."
}

$expectedHashes = @(
    $baseline.minicpm.python_files_sha256_after_patch.PSObject.Properties
)
$actualHashes = @(
    $manifest.python_files_sha256.PSObject.Properties
)
if ($expectedHashes.Count -ne $actualHashes.Count) {
    throw "MiniCPM remote-code file count does not match the validated baseline."
}
foreach ($property in $expectedHashes) {
    $actual = $manifest.python_files_sha256.PSObject.Properties[$property.Name]
    if ($null -eq $actual -or $actual.Value -ne $property.Value) {
        throw "MiniCPM remote-code SHA256 mismatch: $($property.Name)"
    }
}

Write-Host "MiniCPM snapshot is pinned and verified:"
Write-Host "  repository: $($manifest.repo_id)"
Write-Host "  revision:   $($manifest.resolved_revision)"
Write-Host "  manifest:   $manifestPath"

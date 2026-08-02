[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$EvidenceDir,
    [string]$Destination = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$evidenceCandidate = if ([System.IO.Path]::IsPathRooted($EvidenceDir)) {
    $EvidenceDir
} else {
    Join-Path $repoRoot $EvidenceDir
}
if (-not (Test-Path -LiteralPath $evidenceCandidate -PathType Container)) {
    throw "Evidence directory not found: $evidenceCandidate"
}
$evidencePath = (Resolve-Path -LiteralPath $evidenceCandidate).Path

if ([string]::IsNullOrWhiteSpace($Destination)) {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $destinationPath = Join-Path $evidencePath "validation-bundle-$timestamp.zip"
} elseif ([System.IO.Path]::IsPathRooted($Destination)) {
    $destinationPath = [System.IO.Path]::GetFullPath($Destination)
} else {
    $destinationPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $Destination))
}
$destinationParent = Split-Path -Parent $destinationPath
New-Item -ItemType Directory -Path $destinationParent -Force | Out-Null
if (Test-Path -LiteralPath $destinationPath) {
    throw "Destination already exists: $destinationPath"
}

$staging = Join-Path ([System.IO.Path]::GetTempPath()) (
    "had-validation-" + [guid]::NewGuid().ToString("N")
)
New-Item -ItemType Directory -Path $staging | Out-Null

function Copy-RepositoryFile {
    param([Parameter(Mandatory = $true)][string]$RelativePath)
    $source = Join-Path $repoRoot $RelativePath
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        return
    }
    $target = Join-Path $staging $RelativePath
    New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
    Copy-Item -LiteralPath $source -Destination $target
}

try {
    foreach ($relativePath in @(
        ".github\workflows\ci.yml",
        "Dockerfile",
        "Dockerfile.gpu",
        "README.md",
        "models\README.md",
        "pyproject.toml",
        "tests\test_system.py",
        "tests\test_reproducibility.py",
        "src\silhouette_detector\attributes\minicpm.py",
        "src\silhouette_detector\model_store.py",
        "src\silhouette_detector\pipeline.py"
    )) {
        Copy-RepositoryFile -RelativePath $relativePath
    }

    foreach ($directory in @("requirements", "reproducibility", "scripts", "src", "tests")) {
        $sourceRoot = Join-Path $repoRoot $directory
        if (-not (Test-Path -LiteralPath $sourceRoot -PathType Container)) {
            continue
        }
        Get-ChildItem -LiteralPath $sourceRoot -Recurse -File | Where-Object {
            $_.FullName -notmatch "[\\/]__pycache__[\\/]" -and
            $_.Extension -notin @(".pyc", ".pyo")
        } | ForEach-Object {
            $relative = $_.FullName.Substring($repoRoot.Length + 1)
            Copy-RepositoryFile -RelativePath $relative
        }
    }

    $evidenceTarget = Join-Path $staging "evidence"
    New-Item -ItemType Directory -Path $evidenceTarget -Force | Out-Null
    Get-ChildItem -LiteralPath $evidencePath -Recurse -File | Where-Object {
        $_.Extension -ne ".zip"
    } | ForEach-Object {
        $relative = $_.FullName.Substring($evidencePath.Length).TrimStart("\", "/")
        $target = Join-Path $evidenceTarget $relative
        New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
        Copy-Item -LiteralPath $_.FullName -Destination $target
    }

    $metadataDir = Join-Path $staging "metadata"
    New-Item -ItemType Directory -Path $metadataDir -Force | Out-Null
    & git -C $repoRoot rev-parse HEAD |
        Set-Content -LiteralPath (Join-Path $metadataDir "git-head.txt") -Encoding UTF8
    & git -C $repoRoot status --short --branch |
        Set-Content -LiteralPath (Join-Path $metadataDir "git-status.txt") -Encoding UTF8

    $manifestItems = @(
        Get-ChildItem -LiteralPath $staging -Recurse -File | ForEach-Object {
            [ordered]@{
                path = $_.FullName.Substring($staging.Length + 1).Replace("\", "/")
                size_bytes = $_.Length
                sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLower()
            }
        }
    )
    $manifest = [ordered]@{
        schema_version = 1
        created_at = (Get-Date).ToUniversalTime().ToString("o")
        evidence_source = $evidencePath
        excludes = @("model weights", "MiniCPM snapshot", "virtual environments", "secrets")
        files = $manifestItems
    }
    $manifest | ConvertTo-Json -Depth 6 |
        Set-Content -LiteralPath (Join-Path $staging "bundle-manifest.json") -Encoding UTF8

    Compress-Archive -Path (Join-Path $staging "*") -DestinationPath $destinationPath
} finally {
    if (Test-Path -LiteralPath $staging -PathType Container) {
        Remove-Item -LiteralPath $staging -Recurse -Force
    }
}

Write-Host "Validation bundle: $destinationPath"
Write-Host "The archive excludes model weights, virtual environments, caches, and secrets."

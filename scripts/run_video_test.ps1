[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("resnet", "minicpm")]
    [string]$Backend,

    [Parameter(Mandatory = $true)]
    [string]$Video,

    [string]$Yolo = "",
    [string]$ResNetCheckpoint = "",
    [string]$TransformerCheckpoint = "",
    [string]$MiniCPMModelDir = "",
    [string]$Device = "",
    [string]$Venv = "",
    [string]$OutputDir = "",
    [int]$Timeout = 3600,
    [switch]$SkipUnit,
    [switch]$WithRuff,
    [switch]$AllowUnverifiedModelCode
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$baselinePath = Join-Path $repoRoot "reproducibility\windows-cuda-baseline.json"
if (-not (Test-Path -LiteralPath $baselinePath -PathType Leaf)) {
    throw "Reproducibility baseline not found: $baselinePath"
}
$baseline = Get-Content -LiteralPath $baselinePath -Raw | ConvertFrom-Json

function Resolve-InputFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value,
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [string[]]$Candidates = @()
    )

    if (-not [string]::IsNullOrWhiteSpace($Value)) {
        $candidate = if ([System.IO.Path]::IsPathRooted($Value)) {
            $Value
        } else {
            Join-Path $repoRoot $Value
        }
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            throw "$Label not found: $candidate"
        }
        return (Resolve-Path -LiteralPath $candidate).Path
    }

    foreach ($name in $Candidates) {
        $candidate = Join-Path (Join-Path $repoRoot "models") $name
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    throw "$Label was not provided and no known filename exists under models/."
}

function Resolve-InputDirectory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value,
        [Parameter(Mandatory = $true)]
        [string]$Label
    )
    if ([string]::IsNullOrWhiteSpace($Value)) {
        throw "$Label is required."
    }
    $candidate = if ([System.IO.Path]::IsPathRooted($Value)) {
        $Value
    } else {
        Join-Path $repoRoot $Value
    }
    if (-not (Test-Path -LiteralPath $candidate -PathType Container)) {
        throw "$Label not found: $candidate"
    }
    return (Resolve-Path -LiteralPath $candidate).Path
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

$videoPath = Resolve-InputFile -Value $Video -Label "Test video"
$yoloPath = Resolve-InputFile `
    -Value $Yolo `
    -Label "YOLO ONNX model" `
    -Candidates @("yolov8s_576x1024_v2.onnx", "yolo.onnx")
$yoloHash = Get-Sha256 -Path $yoloPath
if ($yoloHash -ne $baseline.yolo.checkpoint_sha256) {
    throw (
        "YOLO checkpoint SHA256 mismatch. Expected " +
        "$($baseline.yolo.checkpoint_sha256), got $yoloHash."
    )
}

if ([string]::IsNullOrWhiteSpace($Device)) {
    $Device = if ($Backend -eq "resnet") { "cpu" } else { "cuda" }
}
$Device = $Device.ToLowerInvariant()
if ([string]::IsNullOrWhiteSpace($Venv)) {
    $Venv = if ($Backend -eq "resnet") {
        ".venv-cpu-py311"
    } else {
        ".venv-minicpm-torch280"
    }
}
$venvPath = if ([System.IO.Path]::IsPathRooted($Venv)) {
    $Venv
} else {
    Join-Path $repoRoot $Venv
}
$python = Join-Path $venvPath "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Virtual environment Python not found: $python"
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $outputPath = Join-Path $repoRoot "var\video-tests\$timestamp-$Backend"
} elseif ([System.IO.Path]::IsPathRooted($OutputDir)) {
    $outputPath = [System.IO.Path]::GetFullPath($OutputDir)
} else {
    $outputPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputDir))
}
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null

$arguments = @(
    (Join-Path $repoRoot "tests\test_system.py"),
    "--video", $videoPath,
    "--yolo", $yoloPath,
    "--device", $Device,
    "--backend", $Backend,
    "--timeout", $Timeout,
    "--output-dir", $outputPath
)
$resnetPath = $null
$resnetHash = $null
$transformerPath = $null
$transformerHash = $null
$modelDir = $null
$modelRevision = $null

if ($Backend -eq "resnet") {
    $resnetPath = Resolve-InputFile `
        -Value $ResNetCheckpoint `
        -Label "ResNet checkpoint" `
        -Candidates @("resnet_ens_11.19_e60_s0.782.pt", "resnet_attributes.pt")
    $resnetHash = Get-Sha256 -Path $resnetPath
    if ($resnetHash -ne $baseline.resnet.checkpoint_sha256) {
        throw (
            "ResNet checkpoint SHA256 mismatch. Expected " +
            "$($baseline.resnet.checkpoint_sha256), got $resnetHash."
        )
    }
    $arguments += @("--resnet", $resnetPath)
    if ([string]::IsNullOrWhiteSpace($TransformerCheckpoint)) {
        $arguments += "--skip-transformer"
    } else {
        $transformerPath = Resolve-InputFile `
            -Value $TransformerCheckpoint `
            -Label "Transformer checkpoint"
        $transformerHash = Get-Sha256 -Path $transformerPath
        if ($transformerHash -ne $baseline.transformer.checkpoint_sha256) {
            throw (
                "Transformer checkpoint SHA256 mismatch. Expected " +
                "$($baseline.transformer.checkpoint_sha256), got $transformerHash."
            )
        }
        $arguments += @("--transformer", $transformerPath)
    }
} else {
    if ($Device -eq "cpu") {
        throw "MiniCPM INT4 video inference requires CUDA; CPU validation uses ResNet."
    }
    $modelDir = Resolve-InputDirectory `
        -Value $MiniCPMModelDir `
        -Label "MiniCPM snapshot"
    $transformerPath = Resolve-InputFile `
        -Value $TransformerCheckpoint `
        -Label "Transformer checkpoint"
    $transformerHash = Get-Sha256 -Path $transformerPath
    if ($transformerHash -ne $baseline.transformer.checkpoint_sha256) {
        throw (
            "Transformer checkpoint SHA256 mismatch. Expected " +
            "$($baseline.transformer.checkpoint_sha256), got $transformerHash."
        )
    }
    $manifest = Join-Path $modelDir "model-manifest.json"
    if (-not $AllowUnverifiedModelCode -and -not (Test-Path -LiteralPath $manifest -PathType Leaf)) {
        throw (
            "MiniCPM snapshot has no model-manifest.json. Run " +
            "scripts\prepare_minicpm_snapshot.ps1 or pass -AllowUnverifiedModelCode explicitly."
        )
    }
    if (Test-Path -LiteralPath $manifest -PathType Leaf) {
        $modelManifest = Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json
        if ($modelManifest.PSObject.Properties.Name -contains "resolved_revision") {
            $modelRevision = $modelManifest.resolved_revision
        }
        if ($modelRevision -ne $baseline.minicpm.revision) {
            throw (
                "MiniCPM revision mismatch. Expected $($baseline.minicpm.revision), " +
                "got $modelRevision."
            )
        }
        if ($modelManifest.config_sha256 -ne $baseline.minicpm.config_sha256) {
            throw "MiniCPM manifest config.json SHA256 differs from the central baseline."
        }
        $expectedHashes = @(
            $baseline.minicpm.python_files_sha256_after_patch.PSObject.Properties
        )
        $actualHashes = @($modelManifest.python_files_sha256.PSObject.Properties)
        if ($expectedHashes.Count -ne $actualHashes.Count) {
            throw "MiniCPM manifest contains a different number of remote-code Python files."
        }
        foreach ($expectedHash in $expectedHashes) {
            $actualHash = @(
                $actualHashes | Where-Object { $_.Name -eq $expectedHash.Name }
            )
            if ($actualHash.Count -ne 1 -or $actualHash[0].Value -ne $expectedHash.Value) {
                throw "MiniCPM remote-code SHA256 mismatch: $($expectedHash.Name)"
            }
        }
    }
    $arguments += @(
        "--minicpm-model-dir", $modelDir,
        "--transformer", $transformerPath
    )
    if ($AllowUnverifiedModelCode) {
        $arguments += "--allow-unverified-model-code"
    }
}
if ($SkipUnit) {
    $arguments += "--skip-unit"
}
if ($WithRuff) {
    $arguments += "--with-ruff"
}

$inputManifest = [ordered]@{
    schema_version = 1
    created_at = (Get-Date).ToUniversalTime().ToString("o")
    backend = $Backend
    device = $Device
    environment_python = $python
    inputs = [ordered]@{
        video = [ordered]@{
            path = $videoPath
            size_bytes = (Get-Item -LiteralPath $videoPath).Length
            sha256 = Get-Sha256 -Path $videoPath
        }
        yolo = [ordered]@{
            path = $yoloPath
            size_bytes = (Get-Item -LiteralPath $yoloPath).Length
            sha256 = $yoloHash
        }
        resnet = if ($null -ne $resnetPath) {
            [ordered]@{
                path = $resnetPath
                size_bytes = (Get-Item -LiteralPath $resnetPath).Length
                sha256 = $resnetHash
            }
        } else { $null }
        transformer = if ($null -ne $transformerPath) {
            [ordered]@{
                path = $transformerPath
                size_bytes = (Get-Item -LiteralPath $transformerPath).Length
                sha256 = $transformerHash
            }
        } else { $null }
        minicpm = if ($null -ne $modelDir) {
            [ordered]@{
                path = $modelDir
                revision = $modelRevision
                manifest = $manifest
            }
        } else { $null }
    }
}
$inputManifest | ConvertTo-Json -Depth 8 |
    Set-Content -LiteralPath (Join-Path $outputPath "input-manifest.json") -Encoding UTF8

$logPath = Join-Path $outputPath "run.log"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$previousConsoleOutputEncoding = [Console]::OutputEncoding
$previousOutputEncoding = $OutputEncoding
$previousPythonUtf8 = $env:PYTHONUTF8
$previousPythonIoEncoding = $env:PYTHONIOENCODING
Push-Location $repoRoot
try {
    [Console]::OutputEncoding = $utf8NoBom
    $OutputEncoding = $utf8NoBom
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    & $python @arguments 2>&1 | Tee-Object -FilePath $logPath
    $testExitCode = $LASTEXITCODE
} finally {
    [Console]::OutputEncoding = $previousConsoleOutputEncoding
    $OutputEncoding = $previousOutputEncoding
    if ($null -eq $previousPythonUtf8) {
        Remove-Item Env:\PYTHONUTF8 -ErrorAction SilentlyContinue
    } else {
        $env:PYTHONUTF8 = $previousPythonUtf8
    }
    if ($null -eq $previousPythonIoEncoding) {
        Remove-Item Env:\PYTHONIOENCODING -ErrorAction SilentlyContinue
    } else {
        $env:PYTHONIOENCODING = $previousPythonIoEncoding
    }
    Pop-Location
}
if ($testExitCode -ne 0) {
    throw "Video test failed; see $logPath and $outputPath\report.json"
}

$reportPath = Join-Path $outputPath "report.json"
if (-not (Test-Path -LiteralPath $reportPath -PathType Leaf)) {
    throw "Video test did not create report.json: $reportPath"
}
$report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
if ($report.success -ne $true) {
    throw "Video test report is not successful: $reportPath"
}

Write-Host "Video test passed:"
Write-Host "  backend: $Backend"
Write-Host "  device:  $Device"
Write-Host "  report:  $reportPath"
Write-Host "  log:     $logPath"
Write-Host "  video:   $outputPath\result.mp4"

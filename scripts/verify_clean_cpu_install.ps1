[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Video,
    [string]$Yolo = "",
    [string]$ResNetCheckpoint = "",
    [string]$TransformerCheckpoint = "",
    [string]$ReuseSeedVenv = "",
    [string]$ReuseCleanVenv = "",
    [string]$OutputDir = "",
    [int]$Timeout = 7200
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $outputPath = Join-Path $repoRoot "var\cpu-clean-validation-$timestamp"
} elseif ([System.IO.Path]::IsPathRooted($OutputDir)) {
    $outputPath = [System.IO.Path]::GetFullPath($OutputDir)
} else {
    $outputPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputDir))
}
if (Test-Path -LiteralPath $outputPath) {
    throw "Clean validation output already exists: $outputPath"
}
New-Item -ItemType Directory -Path $outputPath | Out-Null

$seedVenv = if ([string]::IsNullOrWhiteSpace($ReuseSeedVenv)) {
    Join-Path $repoRoot ".venv-cpu-seed-$timestamp"
} elseif ([System.IO.Path]::IsPathRooted($ReuseSeedVenv)) {
    [System.IO.Path]::GetFullPath($ReuseSeedVenv)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $repoRoot $ReuseSeedVenv))
}
$cleanVenv = if ([string]::IsNullOrWhiteSpace($ReuseCleanVenv)) {
    Join-Path $repoRoot ".venv-cpu-lock-$timestamp"
} elseif ([System.IO.Path]::IsPathRooted($ReuseCleanVenv)) {
    [System.IO.Path]::GetFullPath($ReuseCleanVenv)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $repoRoot $ReuseCleanVenv))
}
$missingBootstrapLock = Join-Path $outputPath "intentionally-missing-bootstrap.lock.txt"
$cpuLock = Join-Path $repoRoot "requirements\locks\windows-cpu-py311.lock.txt"
$setupScript = Join-Path $PSScriptRoot "setup_cpu_windows.ps1"
$videoScript = Join-Path $PSScriptRoot "run_video_test.ps1"
$lockScript = Join-Path $PSScriptRoot "export_windows_lock.ps1"
$bundleScript = Join-Path $PSScriptRoot "export_validation_bundle.ps1"
$reportPath = Join-Path $outputPath "cpu-clean-install-report.json"
$steps = [System.Collections.Generic.List[object]]::new()
$started = Get-Date

function Add-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Status,
        [string]$Detail = ""
    )
    $null = $steps.Add([ordered]@{
        name = $Name
        status = $Status
        detail = $Detail
        recorded_at = (Get-Date).ToUniversalTime().ToString("o")
    })
}

function Write-ValidationReport {
    param([string]$Status, [string]$ErrorMessage = "")
    $payload = [ordered]@{
        schema_version = 1
        status = $Status
        started_at = $started.ToUniversalTime().ToString("o")
        finished_at = (Get-Date).ToUniversalTime().ToString("o")
        seed_venv = $seedVenv
        clean_venv = $cleanVenv
        cpu_lock = $cpuLock
        cpu_lock_sha256 = if (Test-Path -LiteralPath $cpuLock -PathType Leaf) {
            (Get-FileHash -LiteralPath $cpuLock -Algorithm SHA256).Hash.ToLower()
        } else {
            $null
        }
        video = $Video
        yolo = $Yolo
        resnet_checkpoint = $ResNetCheckpoint
        transformer_checkpoint = $TransformerCheckpoint
        steps = @($steps)
        error = if ([string]::IsNullOrWhiteSpace($ErrorMessage)) { $null } else { $ErrorMessage }
    }
    $payload | ConvertTo-Json -Depth 8 |
        Set-Content -LiteralPath $reportPath -Encoding UTF8
}

try {
    if ([string]::IsNullOrWhiteSpace($ReuseSeedVenv)) {
        Write-Host "[1/6] Creating a fresh bootstrap CPU environment..."
        & $setupScript `
            -Venv $seedVenv `
            -LockFile $missingBootstrapLock `
            -RequireNew
        Add-Step -Name "Create bootstrap CPU environment" -Status "passed" -Detail $seedVenv
    } else {
        Write-Host "[1/6] Reusing the previously created bootstrap CPU environment..."
        $seedPython = Join-Path $seedVenv "Scripts\python.exe"
        if (-not (Test-Path -LiteralPath $seedPython -PathType Leaf)) {
            throw "Reusable bootstrap environment Python not found: $seedPython"
        }
        & $seedPython -m pip check
        if ($LASTEXITCODE -ne 0) {
            throw "pip check failed in reusable bootstrap environment: $seedVenv"
        }
        Add-Step -Name "Reuse bootstrap CPU environment" -Status "passed" -Detail $seedVenv
    }

    Write-Host "[2/6] Running a quick ResNet video pass before locking..."
    $seedParameters = @{
        Backend = "resnet"
        Video = $Video
        Yolo = $Yolo
        ResNetCheckpoint = $ResNetCheckpoint
        TransformerCheckpoint = $TransformerCheckpoint
        Venv = $seedVenv
        Device = "cpu"
        OutputDir = (Join-Path $outputPath "seed-resnet-video")
        Timeout = $Timeout
        SkipUnit = $true
    }
    & $videoScript @seedParameters
    Add-Step -Name "Bootstrap ResNet video" -Status "passed" `
        -Detail (Join-Path $outputPath "seed-resnet-video\report.json")

    Write-Host "[3/6] Exporting the complete Windows CPU lock..."
    & $lockScript -Target cpu -Venv $seedVenv
    if (-not (Test-Path -LiteralPath $cpuLock -PathType Leaf)) {
        throw "CPU lock was not created: $cpuLock"
    }
    $seedLockHash = (Get-FileHash -LiteralPath $cpuLock -Algorithm SHA256).Hash.ToLower()
    Copy-Item -LiteralPath $cpuLock -Destination (Join-Path $outputPath "seed-cpu.lock.txt")
    Add-Step -Name "Export CPU lock" -Status "passed" -Detail $seedLockHash

    if ([string]::IsNullOrWhiteSpace($ReuseCleanVenv)) {
        Write-Host "[4/6] Installing a second clean environment only from the CPU lock..."
        & $setupScript -Venv $cleanVenv -LockFile $cpuLock -RequireNew
        Add-Step -Name "Install clean CPU environment from lock" `
            -Status "passed" -Detail $cleanVenv
    } else {
        Write-Host "[4/6] Reusing the previously installed clean CPU environment..."
        $reusedCleanPython = Join-Path $cleanVenv "Scripts\python.exe"
        if (-not (Test-Path -LiteralPath $reusedCleanPython -PathType Leaf)) {
            throw "Reusable clean environment Python not found: $reusedCleanPython"
        }
        & $reusedCleanPython -m pip check
        if ($LASTEXITCODE -ne 0) {
            throw "pip check failed in reusable clean environment: $cleanVenv"
        }
        Add-Step -Name "Reuse clean CPU environment from lock" `
            -Status "passed" -Detail $cleanVenv
    }

    Write-Host "[5/6] Running unit tests and the full ResNet video pass in the clean environment..."
    $cleanParameters = @{
        Backend = "resnet"
        Video = $Video
        Yolo = $Yolo
        ResNetCheckpoint = $ResNetCheckpoint
        TransformerCheckpoint = $TransformerCheckpoint
        Venv = $cleanVenv
        Device = "cpu"
        OutputDir = (Join-Path $outputPath "clean-resnet-video")
        Timeout = $Timeout
    }
    & $videoScript @cleanParameters
    Add-Step -Name "Clean environment unit and ResNet video tests" -Status "passed" `
        -Detail (Join-Path $outputPath "clean-resnet-video\report.json")

    Write-Host "[6/6] Re-exporting and comparing the lock, then collecting evidence..."
    & $lockScript -Target cpu -Venv $cleanVenv
    $cleanLockHash = (Get-FileHash -LiteralPath $cpuLock -Algorithm SHA256).Hash.ToLower()
    if ($cleanLockHash -ne $seedLockHash) {
        throw "CPU lock changed after clean installation: $seedLockHash -> $cleanLockHash"
    }
    $cleanPython = Join-Path $cleanVenv "Scripts\python.exe"
    & $cleanPython -m pip freeze --all |
        Set-Content -LiteralPath (Join-Path $outputPath "clean-pip-freeze.txt") -Encoding UTF8
    & $cleanPython -m pip check 2>&1 |
        Tee-Object -FilePath (Join-Path $outputPath "clean-pip-check.txt")
    if ($LASTEXITCODE -ne 0) {
        throw "pip check failed in the clean CPU environment."
    }
    Copy-Item -LiteralPath $cpuLock -Destination (Join-Path $outputPath "verified-cpu.lock.txt")
    Add-Step -Name "Verify stable CPU lock and pip check" -Status "passed" -Detail $cleanLockHash

    Write-ValidationReport -Status "passed"
    & $bundleScript -EvidenceDir $outputPath
} catch {
    Add-Step -Name "Validation failure" -Status "failed" -Detail $_.Exception.Message
    Write-ValidationReport -Status "failed" -ErrorMessage $_.Exception.Message
    Write-Host "CPU validation failed. Partial evidence: $outputPath"
    try {
        & $bundleScript -EvidenceDir $outputPath
    } catch {
        Write-Warning "Could not package partial evidence: $($_.Exception.Message)"
    }
    throw
}

Write-Host "Clean Windows CPU installation is reproducible."
Write-Host "Report: $reportPath"
Write-Host "Lock:   $cpuLock"
Write-Host "Note: the two virtual environments are retained for inspection and can be deleted manually."

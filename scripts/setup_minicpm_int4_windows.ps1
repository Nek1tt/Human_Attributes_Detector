[CmdletBinding()]
param(
    [string]$PythonVersion = "3.11.9",
    [string]$Venv = ".venv-minicpm-torch280",
    [string]$CudaArch = "8.9",
    [int]$MaxJobs = 2,
    [string]$LockFile = "requirements/locks/windows-cuda-py311.lock.txt"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$autoGptqRepository = "https://github.com/RanchiZhao/AutoGPTQ.git"
$autoGptqCommit = "a9c8109ef450793e3d890c76a36f22177fcbbe28"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPath = if ([System.IO.Path]::IsPathRooted($Venv)) {
    $Venv
} else {
    Join-Path $repoRoot $Venv
}
$depsRoot = Join-Path $repoRoot ".deps"
$autoGptqPath = Join-Path $depsRoot "AutoGPTQ-minicpmo"
$pythonSeries = ($PythonVersion -split "\.")[0..1] -join "."
$lockPath = if ([System.IO.Path]::IsPathRooted($LockFile)) {
    $LockFile
} else {
    Join-Path $repoRoot $LockFile
}

function Assert-Command([string]$Name, [string]$InstallHint) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name was not found. $InstallHint"
    }
}

function Import-VisualStudioEnvironment {
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) {
        return
    }

    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio/Installer/vswhere.exe"
    if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
        throw (
            "MSVC was not found. Install Visual Studio 2022 Build Tools with the " +
            "Desktop development with C++ workload."
        )
    }
    $installationPath = (& $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath | Select-Object -First 1)
    if (-not $installationPath) {
        throw "Visual Studio 2022 C++ build tools were not found."
    }
    $devCmd = Join-Path $installationPath "Common7/Tools/VsDevCmd.bat"
    $environmentLines = cmd.exe /s /c "`"$devCmd`" -arch=x64 -host_arch=x64 >nul && set"
    foreach ($line in $environmentLines) {
        if ($line -match "^([^=]+)=(.*)$") {
            Set-Item -Path "Env:$($matches[1])" -Value $matches[2]
        }
    }
    Assert-Command "cl.exe" "Open an x64 Visual Studio Developer PowerShell and run again."
}

Set-Location $repoRoot
Assert-Command "git.exe" "Install Git for Windows."
Assert-Command "py.exe" "Install 64-bit Python $PythonVersion from python.org."
Assert-Command "nvcc.exe" "Install the CUDA Toolkit 12.8 (the PyTorch wheel alone is not enough)."
Import-VisualStudioEnvironment

$nvcc = (Get-Command nvcc.exe).Source
$nvccVersion = (& $nvcc --version | Out-String)
if ($nvccVersion -notmatch "release 12\.8") {
    throw "CUDA Toolkit 12.8 is required. nvcc reported:`n$nvccVersion"
}
$env:CUDA_PATH = Split-Path (Split-Path $nvcc -Parent) -Parent

if (-not (Test-Path -LiteralPath $venvPath -PathType Container)) {
    & py.exe "-$pythonSeries" -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the Python $PythonVersion virtual environment."
    }
}
$python = Join-Path $venvPath "Scripts/python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Virtual environment Python not found: $python"
}

$actualPython = (& $python -c "import platform; print(platform.python_version())").Trim()
if ($actualPython -ne $PythonVersion) {
    throw "Expected Python $PythonVersion, but the virtual environment uses $actualPython."
}

& $python -m pip uninstall -y gptqmodel
if ($LASTEXITCODE -ne 0) { throw "Could not remove GPTQModel from the environment." }

if (Test-Path -LiteralPath $lockPath -PathType Leaf) {
    Write-Host "Installing the complete pinned Windows CUDA lock: $lockPath"
    & $python -m pip install -r $lockPath
    if ($LASTEXITCODE -ne 0) { throw "Could not install the Windows CUDA lock." }
    & $python -m pip install --no-build-isolation --no-deps --editable $repoRoot
    if ($LASTEXITCODE -ne 0) { throw "Could not install the project checkout." }
} else {
    Write-Warning (
        "The complete Windows CUDA lock is not present yet. Bootstrapping the validated direct " +
        "versions; export and commit the full lock with scripts/export_windows_lock.ps1."
    )
    & $python -m pip install --upgrade pip setuptools wheel ninja
    if ($LASTEXITCODE -ne 0) { throw "Could not install Python build tools." }

    & $python -m pip install `
        "torch==2.8.0" `
        "torchvision==0.23.0" `
        "torchaudio==2.8.0" `
        --index-url "https://download.pytorch.org/whl/cu128"
    if ($LASTEXITCODE -ne 0) { throw "Could not install PyTorch 2.8.0 with CUDA 12.8." }

    & $python -m pip install -r (Join-Path $repoRoot "requirements/gpu.txt")
    if ($LASTEXITCODE -ne 0) { throw "Could not install GPU project dependencies." }

    & $python -m pip install `
        --upgrade-strategy only-if-needed `
        -r (Join-Path $repoRoot "requirements/minicpm-int4.txt")
    if ($LASTEXITCODE -ne 0) { throw "Could not install MiniCPM INT4 dependencies." }
}

New-Item -ItemType Directory -Force -Path $depsRoot | Out-Null
if (-not (Test-Path -LiteralPath $autoGptqPath -PathType Container)) {
    & git.exe clone $autoGptqRepository $autoGptqPath
    if ($LASTEXITCODE -ne 0) { throw "Could not clone the MiniCPM AutoGPTQ fork." }
}
if (-not (Test-Path -LiteralPath (Join-Path $autoGptqPath ".git") -PathType Container)) {
    throw "The AutoGPTQ destination exists but is not a Git checkout: $autoGptqPath"
}

& git.exe -C $autoGptqPath fetch origin minicpmo
if ($LASTEXITCODE -ne 0) { throw "Could not fetch the pinned AutoGPTQ commit." }
& git.exe -C $autoGptqPath checkout --detach $autoGptqCommit
if ($LASTEXITCODE -ne 0) { throw "Could not check out the pinned AutoGPTQ commit." }
$actualAutoGptqCommit = (& git.exe -C $autoGptqPath rev-parse HEAD).Trim()
if ($actualAutoGptqCommit -ne $autoGptqCommit) {
    throw "AutoGPTQ commit mismatch: expected $autoGptqCommit, found $actualAutoGptqCommit."
}

& (Join-Path $PSScriptRoot "patch_autogptq_torch28.ps1") -AutoGPTQPath $autoGptqPath
if ($LASTEXITCODE -ne 0) { throw "Could not patch AutoGPTQ CUDA kernels." }
$expectedModifiedFiles = @(
    "autogptq_extension/cuda_64/autogptq_cuda_kernel_64.cu",
    "autogptq_extension/cuda_256/autogptq_cuda_kernel_256.cu"
)
$actualModifiedFiles = @(& git.exe -C $autoGptqPath diff --name-only)
if ($LASTEXITCODE -ne 0) { throw "Could not inspect the AutoGPTQ patch." }
if ($actualModifiedFiles.Count -ne $expectedModifiedFiles.Count) {
    throw "AutoGPTQ contains unexpected tracked changes: $($actualModifiedFiles -join ', ')"
}
foreach ($expectedFile in $expectedModifiedFiles) {
    if ($expectedFile -notin $actualModifiedFiles) {
        throw "AutoGPTQ expected patched file is missing from git diff: $expectedFile"
    }
}
& git.exe -C $autoGptqPath diff --check
if ($LASTEXITCODE -ne 0) { throw "AutoGPTQ patch failed git diff --check." }

$env:TORCH_CUDA_ARCH_LIST = $CudaArch
$env:COMPILE_MARLIN = "0"
$env:MAX_JOBS = [string]$MaxJobs

& $python -m pip install `
    --no-build-isolation `
    --no-cache-dir `
    --no-deps `
    --editable $autoGptqPath
if ($LASTEXITCODE -ne 0) { throw "AutoGPTQ CUDA build failed." }

& (Join-Path $PSScriptRoot "verify_minicpm_env.ps1") -Venv $venvPath
if ($LASTEXITCODE -ne 0) { throw "Final environment verification failed." }

Write-Host ""
Write-Host "MiniCPM INT4 environment is ready. Activate it with:"
Write-Host "  $venvPath\Scripts\Activate.ps1"

[CmdletBinding()]
param(
    [string]$PythonVersion = "3.11.9",
    [string]$Venv = ".venv-cpu-py311",
    [string]$LockFile = "requirements/locks/windows-cpu-py311.lock.txt",
    [switch]$RequireNew
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPath = if ([System.IO.Path]::IsPathRooted($Venv)) {
    $Venv
} else {
    Join-Path $repoRoot $Venv
}
$lockPath = if ([System.IO.Path]::IsPathRooted($LockFile)) {
    $LockFile
} else {
    Join-Path $repoRoot $LockFile
}
$pythonSeries = ($PythonVersion -split "\.")[0..1] -join "."
if (-not (Get-Command py.exe -ErrorAction SilentlyContinue)) {
    throw "py.exe was not found. Install 64-bit Python $PythonVersion from python.org."
}
if ($RequireNew -and (Test-Path -LiteralPath $venvPath)) {
    throw "A clean environment was requested, but the target already exists: $venvPath"
}
if (-not (Test-Path -LiteralPath $venvPath -PathType Container)) {
    & py.exe "-$pythonSeries" -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the Python $PythonVersion virtual environment."
    }
}
$python = Join-Path $venvPath "Scripts/python.exe"
$actualPython = (& $python -c "import platform; print(platform.python_version())").Trim()
if ($actualPython -ne $PythonVersion) {
    throw "Expected Python $PythonVersion, but the virtual environment uses $actualPython."
}

Push-Location $repoRoot
try {
    if (Test-Path -LiteralPath $lockPath -PathType Leaf) {
        Write-Host "Installing the complete pinned Windows CPU lock: $lockPath"
        & $python -m pip install -r $lockPath
        if ($LASTEXITCODE -ne 0) { throw "Could not install the Windows CPU lock." }
        & $python -m pip install --no-build-isolation --no-deps --editable $repoRoot
        if ($LASTEXITCODE -ne 0) { throw "Could not install the project checkout." }
    } else {
        Write-Warning (
            "The complete Windows CPU lock is not present yet. Bootstrapping direct versions; " +
            "export and commit the full lock with scripts/export_windows_lock.ps1."
        )
        & $python -m pip install --upgrade pip setuptools wheel
        if ($LASTEXITCODE -ne 0) { throw "Could not install Python packaging tools." }
        & $python -m pip install `
            "torch==2.8.0" `
            "torchvision==0.23.0" `
            --index-url "https://download.pytorch.org/whl/cpu"
        if ($LASTEXITCODE -ne 0) { throw "Could not install CPU PyTorch 2.8.0." }
        & $python -m pip install -r (Join-Path $repoRoot "requirements/cpu.txt")
        if ($LASTEXITCODE -ne 0) { throw "Could not install CPU project dependencies." }
    }
    & $python -m pip check
    if ($LASTEXITCODE -ne 0) { throw "pip check found an inconsistent CPU environment." }
    $verification = @'
import platform
import torch
import torchvision

assert platform.python_version() == "3.11.9", platform.python_version()
assert torch.__version__.split("+")[0] == "2.8.0", torch.__version__
assert torchvision.__version__.split("+")[0] == "0.23.0", torchvision.__version__
assert torch.version.cuda is None, torch.version.cuda
assert not torch.cuda.is_available(), "CPU environment unexpectedly exposes CUDA"
print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CPU environment: OK")
'@
    $verification | & $python -
    if ($LASTEXITCODE -ne 0) { throw "CPU environment verification failed." }
} finally {
    Pop-Location
}

Write-Host "CPU environment is ready. Activate it with:"
Write-Host "  $venvPath\Scripts\Activate.ps1"

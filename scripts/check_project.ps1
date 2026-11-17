param([switch]$FullTraining)
$ErrorActionPreference = "Stop"
Push-Location (Join-Path $PSScriptRoot "..")
try {
    $arguments = @("scripts/check_project.py")
    if ($FullTraining) { $arguments += "--full-training" }
    $environmentDir = if ($env:UV_PROJECT_ENVIRONMENT) {
        $env:UV_PROJECT_ENVIRONMENT
    } elseif (Test-Path '.venv') {
        '.venv'
    } else {
        '.venv-clean'
    }
    $pythonPath = Join-Path $environmentDir 'Scripts/python.exe'
    if (-not (Test-Path $pythonPath)) { throw 'Run uv sync first' }
    & $pythonPath @arguments
    if ($LASTEXITCODE -ne 0) { throw "Project checks failed" }
} finally {
    Pop-Location
}

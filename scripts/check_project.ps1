param([switch]$FullTraining)
$ErrorActionPreference = "Stop"
Push-Location (Join-Path $PSScriptRoot "..")
try {
    $arguments = @("scripts/check_project.py")
    if ($FullTraining) { $arguments += "--full-training" }
    & .\.venv\Scripts\python.exe @arguments
    if ($LASTEXITCODE -ne 0) { throw "Project checks failed" }
} finally {
    Pop-Location
}

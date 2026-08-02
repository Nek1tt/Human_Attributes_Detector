[CmdletBinding()]
param(
    [string]$Venv = ".venv-minicpm-torch280"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPath = if ([System.IO.Path]::IsPathRooted($Venv)) {
    $Venv
} else {
    Join-Path $repoRoot $Venv
}
$python = Join-Path $venvPath "Scripts/python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Virtual environment Python not found: $python"
}

$check = @'
import importlib.util
import sys

import torch
import transformers
import torchvision
import auto_gptq
import autogptq_cuda_64
import autogptq_cuda_256

assert sys.version_info[:2] == (3, 11), sys.version
assert torch.__version__.split("+")[0] == "2.8.0", torch.__version__
assert torchvision.__version__.split("+")[0] == "0.23.0", torchvision.__version__
assert transformers.__version__ == "4.44.2", transformers.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available(), "CUDA is unavailable in PyTorch"
assert importlib.util.find_spec("gptqmodel") is None, "GPTQModel must not share this environment"

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("Transformers:", transformers.__version__)
print("AutoGPTQ:", auto_gptq.__version__)
print("GPU:", torch.cuda.get_device_name(0))
print("AutoGPTQ CUDA extensions: OK")
'@

# Windows PowerShell 5.1 can strip quotes embedded in a variable passed to
# a native executable via `-c`. Send the verification program through stdin
# so Python receives it verbatim.
$check | & $python -
if ($LASTEXITCODE -ne 0) {
    throw "MiniCPM environment verification failed."
}

& $python -m pip check
if ($LASTEXITCODE -ne 0) {
    throw "pip check found an inconsistent dependency set."
}

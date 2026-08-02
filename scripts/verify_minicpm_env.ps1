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
import importlib.metadata
import sys

required_modules = {
    "accelerate": "accelerate",
    "auto_gptq": "auto-gptq (patched MiniCPM fork)",
    "autogptq_cuda_64": "AutoGPTQ CUDA extension (64)",
    "autogptq_cuda_256": "AutoGPTQ CUDA extension (256)",
    "librosa": "librosa",
    "onnxruntime": "onnxruntime-gpu",
    "sentencepiece": "sentencepiece",
    "soundfile": "soundfile",
    "timm": "timm",
    "torch": "torch",
    "torchaudio": "torchaudio",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "vector_quantize_pytorch": "vector-quantize-pytorch",
    "vocos": "vocos",
}
missing = [
    display_name
    for module_name, display_name in required_modules.items()
    if importlib.util.find_spec(module_name) is None
]
if missing:
    raise RuntimeError("Missing MiniCPM dependencies: " + ", ".join(missing))

import accelerate
import auto_gptq
import autogptq_cuda_64
import autogptq_cuda_256
import librosa
import torch
import onnxruntime as ort
import sentencepiece
import soundfile
import timm
import torchaudio
import transformers
import torchvision
import vector_quantize_pytorch
import vocos

expected_distributions = {
    "accelerate": "1.2.1",
    "auto-gptq": "0.8.0.dev0+cu121",
    "librosa": "0.10.2.post1",
    "onnxruntime-gpu": "1.26.0",
    "sentencepiece": "0.2.0",
    "soundfile": "0.12.1",
    "timm": "0.9.10",
    "transformers": "4.44.2",
    "vector-quantize-pytorch": "1.18.5",
    "vocos": "0.1.0",
}
assert sys.version.split()[0] == "3.11.9", sys.version
assert torch.__version__ == "2.8.0+cu128", torch.__version__
assert torchvision.__version__ == "0.23.0+cu128", torchvision.__version__
assert torchaudio.__version__ == "2.8.0+cu128", torchaudio.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available(), "CUDA is unavailable in PyTorch"
preload_dlls = getattr(ort, "preload_dlls", None)
if callable(preload_dlls):
    preload_dlls()
assert "CUDAExecutionProvider" in ort.get_available_providers(), ort.get_available_providers()
assert importlib.util.find_spec("gptqmodel") is None, "GPTQModel must not share this environment"
for distribution, expected in expected_distributions.items():
    actual = importlib.metadata.version(distribution)
    assert actual == expected, f"{distribution}: expected {expected}, found {actual}"

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)
print("Transformers:", transformers.__version__)
print("AutoGPTQ:", auto_gptq.__version__)
print("ONNX Runtime:", ort.__version__)
print("ONNX Runtime providers:", ort.get_available_providers())
for distribution in expected_distributions:
    print(f"{distribution}:", importlib.metadata.version(distribution))
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM MiB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 3))
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

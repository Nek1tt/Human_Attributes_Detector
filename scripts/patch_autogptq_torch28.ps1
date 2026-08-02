[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$AutoGPTQPath
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$sourceRoot = (Resolve-Path $AutoGPTQPath).Path
$kernelFiles = @(
    "autogptq_extension/cuda_64/autogptq_cuda_kernel_64.cu",
    "autogptq_extension/cuda_256/autogptq_cuda_kernel_256.cu"
)
$oldExpression = "vec.type()"
$newExpression = "vec.scalar_type()"
$oldCount = 0
$newCount = 0
$contents = @{}

foreach ($relativePath in $kernelFiles) {
    $path = Join-Path $sourceRoot $relativePath
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "AutoGPTQ CUDA kernel not found: $path"
    }
    $text = [System.IO.File]::ReadAllText($path)
    $contents[$path] = $text
    $oldCount += ([regex]::Matches($text, [regex]::Escape($oldExpression))).Count
    $newCount += ([regex]::Matches($text, [regex]::Escape($newExpression))).Count
}

if ($oldCount -eq 0 -and $newCount -eq 16) {
    Write-Host "AutoGPTQ CUDA kernels are already patched (16/16)."
    exit 0
}
if ($oldCount -ne 16 -or $newCount -ne 0) {
    throw (
        "Unexpected AutoGPTQ source state: expected 16 '$oldExpression' and 0 " +
        "'$newExpression', found $oldCount and $newCount. Refusing a partial patch."
    )
}

$utf8WithoutBom = [System.Text.UTF8Encoding]::new($false)
foreach ($entry in $contents.GetEnumerator()) {
    $patched = $entry.Value.Replace($oldExpression, $newExpression)
    [System.IO.File]::WriteAllText($entry.Key, $patched, $utf8WithoutBom)
}

Write-Host "Patched all 16 AT_DISPATCH scalar-type calls in AutoGPTQ."

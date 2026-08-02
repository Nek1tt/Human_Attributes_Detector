# Reproducibility records

`windows-cuda-baseline.json` is the path-independent record of the Windows CPU and CUDA runs that
passed on 2026-08-02/03. It contains the exact MiniCPM and AutoGPTQ source identities, model
identities, effective Transformer architecture, package versions, measured runs, hardware, and
Docker base digests.

The captured machine used NVIDIA driver `591.86`, CUDA Toolkit `12.8` / nvcc `12.8.61`, cuDNN
`91002`, and an RTX 4060 Ti with `16379.375 MiB` visible to PyTorch. The complete portable freezes
are committed as `requirements/locks/windows-cpu-py311.lock.txt` (40 pins) and
`requirements/locks/windows-cuda-py311.lock.txt` (101 pins). The test suite checks them
automatically; users do not need to enter checksums during installation.

The passed run observed:

- MiniCPM embeddings `[7, 64, 3584]` and Transformer `input_proj.in_features=3584`;
- one backend load, followed by one cold and five warm inferences;
- 18.133 s model load, 1.596 s first inference, 779.276 ms warm mean;
- 9274.949 MiB maximum allocated VRAM;
- 0 MiB live allocated-memory growth over the five repeated inferences;
- all nine API keys and valid Russian labels;
- no CUDA OOM.

Machine-specific evidence is generated with `scripts/capture_reproducibility.ps1`. Keep the raw
report for each release candidate. A sanitized report may be committed here after reviewing local
paths.

Complete Windows locks must be generated on the target platform with
`scripts/export_windows_lock.ps1`; they must not be synthesized from the direct dependency list or
from a Linux resolver. Both committed locks are validated. The final end-to-end runs processed 33
frames and 66 detections: ResNet/CPU completed the video job in 14.094 s, while
MiniCPM/Transformer/CUDA completed it in 28.250 s with YOLO on `CUDAExecutionProvider`.

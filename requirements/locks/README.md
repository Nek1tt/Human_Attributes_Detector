# Windows lock files

The lock files in this directory are generated from clean, successfully verified Windows
environments. They contain the complete transitive `pip freeze --all` result, while the editable
project checkout and patched AutoGPTQ checkout are recorded separately by Git commit.

Regenerate a CUDA lock only after an intentional dependency change, followed by
`verify_minicpm_env.ps1` and a real MiniCPM video pass:

```powershell
.\scripts\export_windows_lock.ps1 -Target cuda
```

Generate the CPU lock from a separate CPU-only environment:

```powershell
.\scripts\verify_clean_cpu_install.ps1 `
  -Video test-data\person.mp4 `
  -Yolo models\yolov8s_576x1024_v2.onnx `
  -ResNetCheckpoint models\resnet_ens_11.19_e60_s0.782.pt
```

This command exports the lock from a fresh bootstrap environment, creates a second clean
environment only from that lock, reruns the system tests and ResNet video, and verifies that a
second export is byte-identical.

Never copy a Linux or a mixed CPU/CUDA freeze into these files. Review the diff before committing.

Current status:

- `windows-cuda-py311.lock.txt`: 101 exact pins exported from the verified Windows CUDA
  environment; includes `onnxruntime-gpu==1.26.0` and covers the MiniCPM + Transformer backend;
- `windows-cpu-py311.lock.txt`: 40 exact pins reproduced byte-for-byte from a second clean
  environment; covers YOLO ONNX + ResNet on CPU.

`scripts/export_windows_lock.ps1` writes UTF-8 without BOM and uses LF line endings so that a
dependency lock does not change only because it was exported on another Windows configuration.

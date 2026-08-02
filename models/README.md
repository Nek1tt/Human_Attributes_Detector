# Model files

Place private/large artifacts here. They are intentionally ignored by Git.

- `yolo.onnx` — object detector;
- `resnet_attributes.pt` — legacy nine-model ResNet checkpoint;
- `vision_attr_transformer.pt` — trained Transformer head;
- `minicpm-o-2_6-int4/` — local MiniCPM snapshot plus `model-manifest.json`.

Validated Transformer artifact:

- size: `585660899` bytes;
- SHA256: `099510a29497162f9afbab9cad1b5092fe246d70869bb90bd7b6e2ed2b6affe2`;
- MiniCPM embedding dimension / `input_proj.in_features`: `3584`.

Validated detector and ResNet artifacts:

- `yolov8s_576x1024_v2.onnx`: `44805580` bytes, SHA256
  `c0d4889317191548e8c18a31b4b91f8d2b28842102eebe4523f9f1593a9ea933`;
- `resnet_ens_11.19_e60_s0.782.pt`: `943757326` bytes, SHA256
  `0c0ae02e9a5990c6adbd1ac1d2ca0e6a88c21091b87ee7de621115b328447a91`.

The complete model provenance and expected remote-code hashes live in
`reproducibility/windows-cuda-baseline.json`. Run `scripts/prepare_minicpm_snapshot.ps1` before
enabling the MiniCPM backend; do not hand-write `model-manifest.json`.

Validation scripts compare these identities automatically. Do not commit data or weights unless
their license and the consent basis for images of people are documented.

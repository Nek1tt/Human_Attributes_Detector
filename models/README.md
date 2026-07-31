# Model files

Place private/large artifacts here. They are intentionally ignored by Git.

- `yolo.onnx` — object detector;
- `resnet_attributes.pt` — legacy nine-model ResNet checkpoint;
- `vision_attr_transformer.pt` — trained Transformer head;
- `minicpm-o-2_6-int4/` — local MiniCPM snapshot plus `model-manifest.json`.

Keep SHA-256 checksums and provenance for every checkpoint. Do not commit data or weights unless
their license and the consent basis for images of people are documented.

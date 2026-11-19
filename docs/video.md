# Видео API

Из корня проекта выполните [установку зависимостей и восстановление весов](../README.md#готовые-модели).
Положите ONNX-веса детектора людей в `models/yolo.onnx` или задайте путь через `HAD_DETECTOR_MODEL`.

```powershell
$env:HAD_ATTRIBUTE_CHECKPOINT = "models/original_resnet_peta.pt"
$env:HAD_DEVICE = "cpu"
uv run --no-sync had-api
```

Для Transformer выполните `uv run --no-sync had original-download`, задайте
`HAD_ATTRIBUTE_CHECKPOINT=models/original_transformer_peta.pt` и `HAD_DEVICE=cuda`.

Без API-ключа запросы принимаются с локального компьютера.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/jobs -F "file=@person.mp4"
curl http://127.0.0.1:8000/api/v1/jobs/JOB_ID
curl -o result.mp4 http://127.0.0.1:8000/api/v1/jobs/JOB_ID/result
```

Проверить распознавание девяти атрибутов и сохранение читаемого MP4 можно на локальном видео.

```bash
uv run --no-sync python scripts/check_peta_video.py --video person.mp4 --yolo models/yolo.onnx
```

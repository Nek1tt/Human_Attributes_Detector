# Видео и ранее обученные модели

Команды выполняются из корня репозитория. Старые веса не входят в Git.
Нужны YOLO ONNX, веса атрибутов и короткое видео с человеком.

## CPU: ResNet50

Для прежнего Windows-окружения нужен Python 3.11.9:

```powershell
.\scripts\setup_cpu_windows.ps1
.\scripts\run_resnet_video_test.ps1 -Video "test-data\person.mp4" -Yolo "models\yolo.onnx" -ResNetCheckpoint "models\resnet_attributes.pt"
```

Проверка установки в новом окружении:

```powershell
.\scripts\verify_clean_cpu_install.ps1 -Video "test-data\person.mp4" -Yolo "models\yolo.onnx" -ResNetCheckpoint "models\resnet_attributes.pt"
```

## CUDA: MiniCPM и Transformer

Нужны NVIDIA GPU, CUDA Toolkit 12.8 и Visual Studio Build Tools с C++.

```powershell
.\scripts\setup_minicpm_int4_windows.ps1
.\scripts\prepare_minicpm_snapshot.ps1 -MiniCPMModelDir "models\minicpm-o-2_6-int4" -Download
.\scripts\run_minicpm_video_test.ps1 -Video "test-data\person.mp4" -Yolo "models\yolo.onnx" -MiniCPMModelDir "models\minicpm-o-2_6-int4" -TransformerCheckpoint "models\vision_attr_transformer.pt"
```

Версия MiniCPM: `openbmb/MiniCPM-o-2_6-int4`, revision
`f347c848dd57a5dfdf5b6e32eb257101a6a8a07f`.

## Запуск видео API

```powershell
$env:HAD_ATTRIBUTE_BACKEND = "resnet"
$env:HAD_DEVICE = "cpu"
$env:HAD_RESNET_CHECKPOINT = "models/resnet_attributes.pt"
$env:HAD_DETECTOR_MODEL = "models/yolo.onnx"
had-api
```

Без API key принимаются запросы только с локального компьютера.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/jobs -F "file=@person.mp4"
curl http://127.0.0.1:8000/api/v1/jobs/JOB_ID
curl -o result.mp4 http://127.0.0.1:8000/api/v1/jobs/JOB_ID/result
```

Логи и MP4 видеотестов сохраняются в `var/video-tests`.

## Проверка сохранившихся весов

Старое окружение описано в `reproducibility/windows-cuda-baseline.json`.
Контрольные суммы исходных файлов:

| Файл        | SHA256                                                             |
| ----------- | ------------------------------------------------------------------ |
| YOLO        | `c0d4889317191548e8c18a31b4b91f8d2b28842102eebe4523f9f1593a9ea933` |
| Transformer | `099510a29497162f9afbab9cad1b5092fe246d70869bb90bd7b6e2ed2b6affe2` |
| ResNet50    | `0c0ae02e9a5990c6adbd1ac1d2ca0e6a88c21091b87ee7de621115b328447a91` |

```powershell
Get-FileHash models\yolo.onnx -Algorithm SHA256
.\scripts\capture_reproducibility.ps1 -TransformerCheckpoint "models\vision_attr_transformer.pt" -ResNetCheckpoint "models\resnet_attributes.pt" -AllowBaselineDifferences
.\scripts\export_windows_lock.ps1 -Target cuda
```

Исходные CPU/CUDA lock-файлы находятся в `requirements/locks`.
Новый учебный путь с PETA использует `uv.lock` из корня проекта.

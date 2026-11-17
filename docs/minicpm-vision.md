# Архив компактной архитектуры MiniCPM vision

Основное обучение исходной архитектуры описано в [original-training.md](original-training.md).

Модель определяет девять атрибутов исходного проекта. Классы взяты из PETA:
2 + 5 + 2 + 11 + 11 + 5 + 4 + 2 + 2 = 44 выхода.
Схема классов находится в `src/silhouette_detector/peta_data.py`.

## Архитектура

RGB 224×112 → визуальный энкодер MiniCPM → 128 токенов по 1152 признака →
проекция 256 → два Transformer-слоя, четыре головы внимания → среднее по токенам → 44 логита.
Языковая и аудиочасти MiniCPM, AutoGPTQ и ресемплер LLM не используются.

Исходные веса: [MiniCPM-o 2.6](https://huggingface.co/openbmb/MiniCPM-o-2_6),
revision `cf0b69a6118cd3b2c90bb37f3f4366e2fba802ab`.
Скачивается только визуальный энкодер, около 0.8 ГБ. Код проверяется по SHA256.
Attention использует PyTorch SDPA с исходными весами проекций.

## Setup

Нужна NVIDIA GPU. Проверенная конфигурация — RTX 4060 Ti 16 ГБ.

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-vision"
uv sync --locked --extra gpu --extra train --extra dev --extra vision --extra api
uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.get_device_name())"
```

В отдельном терминале из корня проекта:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-vision"
uv run --no-sync mlflow server --host 127.0.0.1 --port 8080 --workers 1 --backend-store-uri sqlite:///var/mlflow.db --default-artifact-root ./var/mlartifacts
```

## Train

```powershell
uv run --no-sync had download
uv run --no-sync python scripts/estimate_training.py
uv run --no-sync python scripts/train_minicpm_vision.py
```

Первый этап: пять эпох, энкодер заморожен, learning rate головы 0.0003.
Второй этап: пять эпох, дообучаются голова и два последних блока энкодера.
Learning rate головы 0.0001, энкодера 0.00001. BF16, batch 32.
Оба этапа используют все 11 060 обучающих и 3887 валидационных изображений.

Изменить длительность или batch:

```powershell
uv run --no-sync python scripts/train_minicpm_vision.py --head-epochs 5 --finetune-epochs 10 --batch-size 16
```

Файлы:

- `models/peta_vision_head.pt` — лучший checkpoint первого этапа;
- `models/peta_vision_finetuned.pt` — лучший checkpoint второго этапа;
- `plots/peta-head/` и `plots/peta-finetuned/` — журналы, графики и конфиги.

## Infer и оценка

```powershell
uv run --no-sync had infer inference.checkpoint=models/peta_vision_finetuned.pt inference.device=cuda inference.limit=10
uv run --no-sync had evaluate inference.checkpoint=models/peta_vision_finetuned.pt inference.device=cuda inference.output=var/peta-vision-test.json
```

Видео API:

```powershell
$env:HAD_ATTRIBUTE_BACKEND = "peta"
$env:HAD_ATTRIBUTE_CHECKPOINT = "models/peta_vision_finetuned.pt"
$env:HAD_DEVICE = "cuda"
$env:HAD_DETECTOR_MODEL = "C:/Human_Attributes_Detector/models/yolov8s_576x1024_v2.onnx"
uv run --no-sync had-api
```

Девять атрибутов возвращаются на русском. Возраст и тип верхней одежды
предсказываются моделью, как и остальные атрибуты.

## Место и время

Перед каждым этапом проверяется свободное место с запасом 4 ГиБ.
Один итоговый checkpoint занимает около 1.56 ГиБ. DVC сохраняет копии
в кеше и local remote. Два этапа требуют до 11–12 ГиБ дополнительного места
с учётом временного checkpoint. Готовые окружения повторно не устанавливаются.

Для пяти эпох каждого этапа замер на реальных батчах дал около 6 и 7 минут
чистого обучения с validation. С загрузкой и сохранением ориентир — 20–35 минут
вместе с бейзлайном. Видеопамять — около 2.1 и 2.7 ГиБ.
Длительность зависит от диска, свободной оперативной памяти и других программ.

Старый эксперимент Market-1501: `--dataset market`.
Его результаты находятся в `docs/minicpm-results.md`; набор классов отличается.

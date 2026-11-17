# MiniCPM vision + Transformer

Используется визуальная часть MiniCPM-o 2.6 и новый классификатор атрибутов.
Языковая модель, аудиомодули, ресемплер в пространство LLM и AutoGPTQ не нужны.
Исходные веса: [openbmb/MiniCPM-o-2_6](https://huggingface.co/openbmb/MiniCPM-o-2_6),
revision `cf0b69a6118cd3b2c90bb37f3f4366e2fba802ab`.

## Архитектура

1. RGB-кроп 224×112, нормализация со средним и отклонением 0.5.
2. Визуальный энкодер MiniCPM: 128 токенов по 1152 признака.
3. Проекция в 256 признаков.
4. Два Transformer-слоя с четырьмя головами внимания.
5. Усреднение токенов и классификация 12 атрибутов Market-1501.

Вместо обычного attention визуального энкодера используется PyTorch SDPA.
Веса проекций сохраняются, формула внимания проверяется тестом.
Голова содержит 1 885 223 параметра. Вся модель — 419 677 463 параметра.

## Установка

Нужна NVIDIA GPU с поддержкой CUDA. Для этой конфигурации используется RTX 4060 Ti 16 ГБ.
Отдельное окружение не меняет CPU-установку:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-vision"
uv sync --locked --extra gpu --extra train --extra dev --extra vision
uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.get_device_name())"
```

В отдельном терминале из корня проекта запустите MLflow:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-vision"
uv run --no-sync mlflow server --host 127.0.0.1 --port 8080 --workers 1 --backend-store-uri sqlite:///var/mlflow.db --default-artifact-root ./var/mlartifacts
```

## Обучение

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-vision"
uv run --no-sync python scripts/train_minicpm_vision.py
```

Скрипт скачивает только визуальные тензоры MiniCPM, проверяет исходный код
по SHA256 и сохраняет их в DVC. Полный многогигабайтный VLM не скачивается.

Два этапа по пять эпох:

- `minicpm-vision-head`: энкодер заморожен, обучается новая Transformer-голова;
- `minicpm-vision-finetuned`: загружается лучшая голова, размораживаются два последних
  блока энкодера. Learning rate головы — 0.0001, энкодера — 0.00001.

Оба этапа используют все 10 550 обучающих и 2386 валидационных изображений.
Выборки разделены по людям. Тестовые люди не участвуют в обучении.
Используется BF16 mixed precision и batch 32.

Изменить длительность и batch:

```powershell
uv run --no-sync python scripts/train_minicpm_vision.py --head-epochs 5 --finetune-epochs 10 --batch-size 32
```

Результаты:

- `models/minicpm_vision_head.pt` — лучшая модель первого этапа;
- `models/minicpm_vision_finetuned.pt` — лучшая модель второго этапа;
- `plots/minicpm-head/` и `plots/minicpm-finetuned/` — графики, метрики и конфиги;
- MLflow — loss, accuracy, macro F1, параметры и версия кода.

Это новые checkpoint-файлы с 12 атрибутами Market-1501.
Они используют обновлённую архитектуру и не загружаются в старую голову на девять атрибутов.

## Предсказание

```powershell
uv run --no-sync had infer inference.checkpoint=models/minicpm_vision_finetuned.pt inference.device=cuda inference.limit=10
```

Для подключения к видео используется общий backend:

```powershell
$env:HAD_ATTRIBUTE_BACKEND = "market"
$env:HAD_MARKET_CHECKPOINT = "models/minicpm_vision_finetuned.pt"
$env:HAD_DEVICE = "cuda"
uv run --no-sync had-api
```

Старый backend `minicpm` оставлен для прежних весов и схемы атрибутов.

## Место на диске

CUDA-окружение занимает около 8 ГБ. Сохраняются лучшие веса каждого этапа,
а не отдельная модель на каждую эпоху. После записи итоговых весов в DVC
временный checkpoint Lightning удаляется. Проверочные копии MLflow и ONNX
можно создавать отдельно, когда они нужны для запуска.

Результаты завершённого обучения: [minicpm-results.md](minicpm-results.md).

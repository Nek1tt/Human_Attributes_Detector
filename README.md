# Определение атрибутов людей

Автор: Абрамов Никита.

## Задача

Проект определяет внешние атрибуты человека: пол, возрастную группу,
одежду и сумки. Для видео есть пайплайн: YOLO находит людей,
SFSORT отслеживает их, классификатор обрабатывает кропы.
Результат видео — MP4 с подписями и JSONL с предсказаниями.

## Данные

Используется [Market-1501](https://zheng-lab-anu.github.io/Project/project_reid.html)
с [разметкой атрибутов](https://github.com/vana77/Market-1501_Attribute).
В датасете 32 668 изображений, 1501 человек.
751 человек относится к обучающей части, 750 — к тестовой.
20% обучающих людей выделяются в validation. Один человек не попадает
в разные части. Неразмеченные фоновые ID исключаются.

Фотографии скачиваются с открытого зеркала Hugging Face, разметка — из
репозитория авторов. Версии и SHA256 указаны в `configs/data/market.yaml`.
Данные хранятся через [DVC](https://dvc.org/doc), локальный remote — `.dvc-storage`.
На новом компьютере он восстанавливается из открытых источников.

## Модель

[ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
с предобученными признаками и линейным классификатором 12 атрибутов:
пол, возраст, длина волос, длина рукава, длина и тип нижней одежды,
головной убор, рюкзак, сумка, дамская сумка, цвет верха и низа.
По умолчанию обучается только классификатор.

Вход — RGB-кроп человека размером 128×64. Обработка входа сохраняется
в checkpoint. Неуказанный цвет исключается из обучения и метрик.
Обучение выполняет [Lightning](https://lightning.ai/docs/pytorch/stable/).

## Метрики

- CrossEntropy: средняя ошибка по атрибутам.
- Accuracy: доля верных предсказаний среди известных меток.
- Macro F1: среднее F1 по классам каждого атрибута, затем по атрибутам.

Loss и accuracy записываются для train и validation, F1 — для validation.
Лучший checkpoint выбирается по validation macro F1.
[MLflow](https://mlflow.org/docs/latest/) хранит метрики, параметры и Git commit.
Три графика, CSV-логи и конфигурация сохраняются в `plots/training/version_*/`.

## Этапы работы

1. Скачать фотографии и разметку, сохранить их в DVC.
2. Разделить данные по людям и подготовить изображения.
3. Обучить классификатор и сохранить лучший checkpoint.
4. Выполнить предсказание на новых кропах людей.
5. Экспортировать ONNX или подготовить модель для сервера.

Описание: [docs/proposal.md](docs/proposal.md).
Разбор всех критериев: [docs/criteria.md](docs/criteria.md).
Результаты проверок: [docs/checks.md](docs/checks.md).
PDF описания: [docs/proposal.pdf](docs/proposal.pdf).
Новая архитектура MiniCPM vision + Transformer: [docs/minicpm-vision.md](docs/minicpm-vision.md).

## Setup

Нужны Git, Python 3.11 или 3.12 и [uv](https://docs.astral.sh/uv/).
Все команды выполняются из корня репозитория.

```bash
git clone --recurse-submodules https://github.com/Nek1tt/Human_Attributes_Detector.git
cd Human_Attributes_Detector
python -m pip install uv
uv sync --locked --extra cpu --extra train --extra dev --extra export --extra api
uv run --no-sync pre-commit install
uv run --no-sync pre-commit run -a
```

`uv sync` создаёт `.venv` и устанавливает версии из `uv.lock`.
Для предсказания с готовым checkpoint достаточно extra `cpu`.
Для учебных команд с DVC нужен также `train`.
MiniCPM и Telegram-бот устанавливаются отдельно.

Перед сдачей код должен находиться в `main` или `master`.
Исходная версия была в `detector-remaster`.

## Train

Скачать и подготовить данные:

```bash
uv run --no-sync had download
```

В отдельном терминале запустите MLflow:

```bash
uv run --no-sync mlflow server --host 127.0.0.1 --port 8080 --workers 1 --backend-store-uri sqlite:///var/mlflow.db --default-artifact-root ./var/mlartifacts
```

Обучение:

```bash
uv run --no-sync had train
```

Команда вызывает DVC pull. При отсутствии local remote данные
восстанавливаются через `download_data()`. Начальные веса ResNet18
скачиваются в `.cache/torch`. Итог — `models/market_attributes.pt`,
модель также сохраняется в DVC.

Настройки меняются через [Hydra](https://hydra.cc/docs/intro/):

```bash
uv run --no-sync had train training.epochs=10 training.learning_rate=0.0005
uv run --no-sync had train model.freeze_backbone=false
```

Группы конфигов в `configs/`: data, preprocess, model, training,
logging, inference и production. Адрес MLflow — `logging.tracking_uri`.

## Production preparation

Для предсказания нужны пакет, зависимости `cpu` и checkpoint.
Для передачи модели скопируйте `.dvc-storage` и выполните `dvc pull`
либо обучите модель заново. Данные и веса в Git не хранятся.

ONNX-экспорт и проверка совпадения с PyTorch:

```bash
uv run --no-sync had export
```

Файл — `models/market_attributes.onnx`. Вход `images`: нормализованный
FP32-тензор `[batch, 3, 128, 64]`. Выход `logits`: `[batch, 39]`.
Порядок атрибутов и классов задан в `market_model.LABELS`.

На компьютере с NVIDIA TensorRT и `trtexec` в PATH:

```bash
uv run --no-sync had tensorrt
```

Результат — `models/market_attributes.engine`. Engine собирается на целевой GPU.

## Infer

Вход — JPEG/PNG с одним человеком, обрезанный по его границам.
Примеры находятся в DVC-датасете, папка
`data/market/Market-1501-v15.09.15/bounding_box_train`.

На тестовой части:

```bash
uv run --no-sync had infer inference.limit=10
```

На своём изображении:

```bash
uv run --no-sync had infer inference.image=person.jpg
```

Результат — `var/predictions.jsonl`: строка на изображение,
поля `image` и `attributes`. Названия классов на английском.
Возраст: `young`, `teenager`, `adult`, `old`.
`inference.confidence_threshold` задаёт порог уверенности;
ниже порога возвращается `unknown`.

### MLflow Serving

```bash
uv run --no-sync had package
uv run --no-sync mlflow models serve -m var/serving-model --env-manager local --host 127.0.0.1 --port 5001
uv run --no-sync python scripts/request_prediction.py person.jpg
```

В запросе JSON с колонкой `image`, содержащей изображение в Base64.
Для сравнения с локальной моделью добавьте `--check-with models/market_attributes.pt`.
При повторной упаковке укажите новую папку:
`had package production.mlflow_model=var/serving-model-v2`.

### Triton

После ONNX-экспорта:

```bash
uv run --no-sync had triton
docker run --rm -p 8000:8000 -v "${PWD}/var/triton-models:/models" nvcr.io/nvidia/tritonserver:26.10-py3 tritonserver --model-repository=/models
uv run --no-sync python scripts/request_triton.py person.jpg
```

Клиент нормализует изображение и переводит выходные логиты в атрибуты.

### Видео и старые веса

Backends `resnet` и `minicpm` сохранены.
Установка и видеотесты — в [docs/video.md](docs/video.md).
Новая модель подключается к тому же видеопайплайну:

```powershell
$env:HAD_ATTRIBUTE_BACKEND = "market"
$env:HAD_MARKET_CHECKPOINT = "models/market_attributes.pt"
$env:HAD_DETECTOR_MODEL = "models/yolo.onnx"
uv run --no-sync had-api
```

YOLO-веса нужны отдельно. Видео API сохраняет девять старых ключей.
Возраст и тип верхней одежды возвращаются как «не определен»:
в Market-1501 нет классов, совпадающих со старой схемой.
Полные 12 атрибутов возвращают `had infer` и MLflow Serving.

## Самостоятельные проверки

```bash
uv run --no-sync pre-commit run -a
uv run --no-sync python -m unittest discover -s tests -v
```

Все команды с коротким обучением на реальных данных
(MLflow должен быть запущен):

```bash
uv run --no-sync python scripts/check_project.py
```

То же с полным обучением:

```bash
uv run --no-sync python scripts/check_project.py --full-training
```

Для PowerShell есть `scripts/check_project.ps1` с флагом `-FullTraining`.
Короткая проверка использует отдельные `models/smoke_attributes.pt` и `var/smoke-plots`.

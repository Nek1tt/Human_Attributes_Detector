# Определение атрибутов людей

Автор: Абрамов Никита.

Проект принимает кроп человека и определяет девять атрибутов: пол,
возрастную группу, головной убор, цвет и тип верхней и нижней одежды,
наличие рюкзака и сумки. Для видео используется YOLO → SFSORT → классификатор.
Результат — видео с подписями и JSONL с атрибутами каждого трека.

## Данные и классы

Используется [PETA](https://mmlab.ie.cuhk.edu.hk/projects/PETA.html).
Архив скачивается с сайта авторов, SHA256 закреплён в `configs/data/peta.yaml`.
Часть i-LID исключена: для её использования авторы указывают отдельное разрешение.
Остаются 18 523 изображения. Разделение по ID из исходной разметки и поддатасету:
11 060 изображений для train, 3887 для validation и 3576 для test. Seed — 42.
Изображения одного ID находятся только в одной части.

| Атрибут       | Классы                                                          |
| ------------- | --------------------------------------------------------------- |
| Пол           | Мужчина, женщина                                                |
| Возраст       | Младшая группа `personalLess15`, 16–30, 31–45, 46–60, старше 60 |
| Головной убор | Нет, есть; учитываются шляпа/шапка и платок                     |
| Цвет верха    | 11 цветов PETA                                                  |
| Цвет низа     | 11 цветов PETA                                                  |
| Тип верха     | Куртка, свитер, костюм, футболка, другая одежда                 |
| Тип низа      | Брюки, шорты, юбка, капри                                       |
| Рюкзак        | Нет, есть                                                       |
| Сумка         | Нет, есть; сумка через плечо, пакет, чемодан или багажная сумка |

Всего 44 выходных значения. Несколько цветов или противоречивые типы одежды
не сводятся к случайному классу: такие метки исключаются из loss и метрик.
Отсутствующий тип верха также исключается. Класс «другая одежда» берётся
только из явной метки `upperBodyOther`. Объединение классов — в `peta_data.py`.

## Модели

Бейзлайн — один предобученный ResNet18 и линейный классификатор.
Вход RGB 128×64, нормализация ImageNet. Обучается классификатор.

Основная модель — визуальный энкодер MiniCPM-o 2.6 и Transformer.
Вход 224×112 → 128 токенов по 1152 признака → проекция в 256 →
два Transformer-слоя с четырьмя головами внимания → усреднение → 44 логита.
Языковая и аудиочасти MiniCPM не используются.

Обе модели обучаются на одинаковом разбиении PETA и возвращают одинаковые
девять атрибутов. Старые веса на 60 классов и эксперимент Market-1501
на 39 классов поддерживаются отдельно.

Описание проекта: [proposal.md](docs/proposal.md), [PDF](docs/proposal.pdf).
Все критерии: [criteria.md](docs/criteria.md).
Обучение основной модели: [minicpm-vision.md](docs/minicpm-vision.md).
Результаты и проверки: [peta-results.md](docs/peta-results.md).

## Setup

Нужны Git, Python 3.11 или 3.12 и [uv](https://docs.astral.sh/uv/).
Команды выполняются из корня репозитория.

```bash
git clone --recurse-submodules https://github.com/Nek1tt/Human_Attributes_Detector.git
cd Human_Attributes_Detector
python -m pip install uv
uv sync --locked --extra cpu --extra train --extra dev --extra export --extra api
uv run --no-sync pre-commit install
uv run --no-sync pre-commit run -a
```

Для GPU-обучения используйте установку из `docs/minicpm-vision.md`.
Версии зависимостей закреплены в `uv.lock`.

## Train

Скачать и подготовить PETA:

```bash
uv run --no-sync had download
```

В отдельном терминале запустить MLflow:

```bash
uv run --no-sync mlflow server --host 127.0.0.1 --port 8080 --workers 1 --backend-store-uri sqlite:///var/mlflow.db --default-artifact-root ./var/mlartifacts
```

Обучить бейзлайн, пять эпох:

```bash
uv run --no-sync had train
```

Результат — `models/peta_attributes.pt`. По умолчанию энкодер заморожен.
Настройки задаются через [Hydra](https://hydra.cc/docs/intro/):

```bash
uv run --no-sync had train training.epochs=10 model.freeze_backbone=false
```

Обучение выполняется через Lightning, оптимизатор AdamW. Ошибка — среднее
CrossEntropy по известным атрибутам. Лучшая модель выбирается по validation
macro F1. MLflow сохраняет loss, accuracy, macro F1, гиперпараметры и Git commit.
Графики, CSV и конфигурация находятся в `plots/`.

Данные и веса хранятся через DVC, local remote — `.dvc-storage`.
Train и infer вызывают DVC pull. На новом компьютере датасет восстанавливается
из открытого архива. Веса передаются с DVC-хранилищем или получаются обучением.

## Production preparation

Экспорт бейзлайна в ONNX и сравнение с PyTorch для batch 1 и 2:

```bash
uv run --no-sync had export
```

Файл `models/peta_attributes.onnx`: вход `[batch, 3, 128, 64]`,
выход `[batch, 44]`. Порядок атрибутов и классов хранится в checkpoint.
Для основной модели в командах укажите
`inference.checkpoint=models/peta_vision_finetuned.pt`.

Упаковка для MLflow Serving:

```bash
uv run --no-sync had package
uv run --no-sync mlflow models serve -m var/serving-model --env-manager local --host 127.0.0.1 --port 5001
uv run --no-sync python scripts/request_prediction.py person.jpg --check-with models/peta_attributes.pt
```

Требуются пакет, checkpoint и зависимости соответствующей модели.
`had tensorrt` вызывает установленный `trtexec`.
`had triton` создаёт репозиторий ONNX-модели для NVIDIA Triton.
Эти два дополнительных способа требуют отдельной установки серверов.

## Infer

На тестовой части:

```bash
uv run --no-sync had infer inference.limit=10
```

На своём кропе человека:

```bash
uv run --no-sync had infer inference.image=person.jpg
```

Результат — `var/predictions.jsonl`, девять атрибутов на английском.
`inference.confidence_threshold` задаёт порог: ниже него возвращается `unknown`.

Полная оценка на тесте, включая precision, recall и F1 каждого класса:

```bash
uv run --no-sync had evaluate inference.output=var/peta-test.json
```

Для основной модели добавьте
`inference.checkpoint=models/peta_vision_finetuned.pt inference.device=cuda`.
Тестовая часть не используется для выбора checkpoint.

### Видео

Новые модели используют общий backend `peta`:

```powershell
$env:HAD_ATTRIBUTE_BACKEND = "peta"
$env:HAD_ATTRIBUTE_CHECKPOINT = "models/peta_attributes.pt"
$env:HAD_DETECTOR_MODEL = "C:/Human_Attributes_Detector/models/yolov8s_576x1024_v2.onnx"
$env:HAD_DEVICE = "cpu"
uv run --no-sync had-api
```

API возвращает те же девять атрибутов на русском. Примеры HTTP-запросов
и старые backends — в [video.md](docs/video.md).

## Самостоятельные проверки

```bash
uv run --no-sync pre-commit run -a
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync python scripts/check_project.py
```

Для последней команды нужен запущенный MLflow. Она выполняет короткое обучение
на настоящих данных, infer, ONNX-экспорт и проверку MLflow-модели.

Проверка всего видеопайплайна с обученной PETA-моделью:

```powershell
uv run --no-sync python scripts/check_peta_video.py --video person.mp4 --yolo models/yolo.onnx --checkpoint models/peta_attributes.pt
```

Для основной модели добавьте `--checkpoint models/peta_vision_finetuned.pt --device cuda`.
Скрипт проверяет читаемость выходного MP4 и наличие всех девяти атрибутов в JSONL.

Замер времени и памяти до полного GPU-обучения:

```bash
uv run --no-sync python scripts/estimate_training.py
```

Результат — `var/training-estimate.json`. Скрипт не сохраняет обученные модели.
Перед каждой тренировкой проверяется свободное место. Сохраняется один лучший
временный checkpoint; после записи итоговых весов в DVC он удаляется.

## Старые результаты

Старый датасет утерян. Его веса остаются совместимы с прежними backends,
но метрики на старых и новых данных не сравниваются напрямую.
Эксперимент Market-1501 описан отдельно в [minicpm-results.md](docs/minicpm-results.md).
Для его воспроизведения используется `data=market`.

# Human Attributes Detector: распознавание внешних атрибутов людей на видео

Никита Абрамов

## Постановка задачи

Система обнаруживает людей на видео, отслеживает их между кадрами и определяет девять внешних атрибутов: пол, возрастную группу, головной убор, цвет и тип верхней и нижней одежды, наличие рюкзака и сумки. Код находится в [репозитории проекта](https://github.com/Nek1tt/Human_Attributes_Detector).

### Формат входных и выходных данных

На вход подаётся видеофайл. Система находит людей и вырезает их изображения для классификации. На выходе получаются MP4 с рамками и подписями и JSONL с временем, track_id, координатами, уверенностью детектора и девятью атрибутами. Для каждого атрибута предусмотрен класс «не определен».

### Метрики

Рассчитываются сумма CrossEntropyLoss по атрибутам и общая accuracy. Для каждого атрибута и класса считаются accuracy, precision, recall, macro F1 и support. Метрики, параметры и Git commit записываются в [MLflow](https://mlflow.org/docs/latest/). Графики loss, accuracy и macro F1 сохраняются в plots.

### Валидация

Изображения [PETA](https://mmlab.ie.cuhk.edu.hk/projects/PETA.html) разделяются по ID людей внутри поддатасетов: 60% групп для train, 20% для validation и 20% для test, seed 42. Один ID находится только в одной части. Лучшая модель сохраняется по validation accuracy. Тестовая часть используется после выбора лучшего варианта. Видеотесты проверяют путь от видео до MP4 и JSONL.

### Данные

Используются кропы людей и разметка PETA. Поддатасет i-LID исключён, поскольку для него требуется отдельное разрешение. Остаются 18 523 изображения: 11 060 train, 3887 validation, 3576 test. Официальный архив проверяется по SHA256, данные и эмбеддинги хранятся через [DVC](https://dvc.org/doc).

Сохранены девять атрибутов. Новые списки классов содержат 53 выхода, включая «не определен» у каждого атрибута. Возрастные группы и типы одежды взяты из PETA. Цвета верха и низа имеют по 11 значений. Типы верха: куртка, свитер, костюм, футболка, другая одежда. Типы низа: брюки, шорты, юбка, капри. Сумка включает сумку через плечо, пакет и чемодан. Пропущенные и противоречивые категориальные метки переводятся в «не определен».

Перед обучением Transformer сохраняются исходные MiniCPM-эмбеддинги формы 1 × 64 × 3584. Основные сложности данных: дисбаланс классов, перекрытия, низкое разрешение и похожая одежда.

## Моделирование

### Бейзлайн

Девять независимых [ResNet50](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html), по одной сети на атрибут. RGB-кроп приводится к 224 × 224 и нормализуется статистиками ImageNet. Каждая сеть имеет прежний классификатор 2048 → 1024 → 512 → число классов. Все девять сетей обучаются, архитектура исходного кода сохранена.

### Основная модель

[MiniCPM-o 2.6 INT4](https://huggingface.co/openbmb/MiniCPM-o-2_6-int4) извлекает визуальные признаки. Используются его исходные VPM и Resampler, возвращающие 64 токена по 3584 признака. Неиспользуемые языковые и аудиовеса не загружаются. Эмбеддинги совпадают с исходной функцией MiniCPM.

Исходный VisionAttrTransformer переводит признаки 3584 → 768, применяет шесть Transformer-слоёв с 12 головами внимания, CLS-токен, attention pooling и девять отдельных классификаторов. Архитектура и размеры промежуточных слоёв сохранены. Обучается классификатор на сохранённых эмбеддингах, оптимизатор AdamW. В [Lightning](https://lightning.ai/docs/pytorch/stable/) проводятся эксперименты с learning rate и взвешиванием ошибки. Конфигурация задаётся через [Hydra](https://hydra.cc/docs/intro/).

Обнаружение людей выполняет YOLO ONNX, сопровождение - [SFSORT](https://github.com/Applied-Deep-Learning-Lab/SFSORT). Атрибуты сохраняются по track_id и используются в следующих кадрах.

## Внедрение

FastAPI принимает видео и возвращает идентификатор задачи. После обработки можно скачать MP4. Покадровые атрибуты сохраняются рядом с видео в JSONL. На CPU используется ResNet50, на CUDA доступны обе модели.

На тестовой части PETA ResNet50 получил accuracy **0.7640** и macro F1 **0.5668**, MiniCPM + Transformer — **0.7894** и **0.6093**. Для сдачи доступны [описание проекта в PDF](docs/proposal.pdf), [условия экспериментов](docs/original-training.md) и [результаты по атрибутам](docs/original-results.md).

## Setup

Нужны Git, Python 3.11 или 3.12 и [uv](https://docs.astral.sh/uv/).
Все команды выполняются из корня репозитория.

```bash
git clone --branch detector-remaster --recurse-submodules https://github.com/Nek1tt/Human_Attributes_Detector.git
cd Human_Attributes_Detector
python -m pip install uv
uv sync --locked --extra gpu --extra train --extra dev --extra vision --extra api --extra export
uv run --no-sync pre-commit install
uv run --no-sync pre-commit run -a
```

Для обучения используется NVIDIA GPU. Для CPU-инференса ResNet вместо GPU/vision
достаточно `uv sync --locked --extra cpu --extra train --extra dev --extra export --extra api`.
Версии зависимостей закреплены в `uv.lock`.

## Готовые модели

В `models/` лежат два DVC-указателя на обученные классификаторы. Для запуска готовых
весов на новом компьютере понадобится копия каталога `.dvc-storage/` с объектами этих
двух моделей в корне проекта. Восстановить файлы можно одной командой:

```bash
uv run --no-sync dvc pull models/original_resnet_peta.pt.dvc models/original_transformer_peta.pt.dvc
```

Чтобы подготовить для преподавателя хранилище только с этими весами, на машине
с обученными моделями выполните команды ниже. Передайте каталог `dist/course-weights`
и при получении поместите его в корень клона под именем `.dvc-storage`.

```bash
uv run --no-sync dvc remote add --local -f course dist/course-weights
uv run --no-sync dvc push -r course models/original_resnet_peta.pt.dvc models/original_transformer_peta.pt.dvc
```

ResNet50 после этого готов к предсказанию на кропах. Для Transformer отдельно
загрузите визуальный энкодер MiniCPM из закреплённой ревизии Hugging Face
в `models/original-minicpm-visual/` (около 0.91 ГиБ):

```bash
uv run --no-sync had original-download
```

Если DVC-хранилище недоступно, классификаторы можно обучить на PETA по инструкции ниже.

## Train

В отдельном терминале запустить MLflow:

```bash
uv run --no-sync mlflow server --host 127.0.0.1 --port 8080 --workers 1 --backend-store-uri sqlite:///var/mlflow.db --default-artifact-root ./var/mlartifacts
```

Скачать PETA для обучения:

```bash
uv run --no-sync had download
```

Для обучения Transformer дополнительно подготовить эмбеддинги:

```bash
uv run --no-sync had original-download
uv run --no-sync had original-extract
```

Два варианта обучения каждой исходной архитектуры:

```bash
uv run --no-sync python scripts/train_original.py --skip-extraction
```

Для воспроизведения опубликованных метрик нужны прежние веса автора.
Скрипт принимает пути к ним через `--resnet-weights` и `--transformer-weights`.
Без них обучение стартует с ImageNet-весов ResNet50 и случайной инициализации Transformer.

По умолчанию выполняется до четырёх эпох на эксперимент. Меняются learning rate
и взвешивание ошибки. Сохраняется лучший checkpoint по validation accuracy.
Лучший Transformer дополнительно дообучается с меньшим learning rate.
Тестовая часть оценивается после выбора лучшего варианта каждой архитектуры.
Журналы и три графика каждого эксперимента сохраняются в `plots/original-*/`.

Один запуск ResNet с настройками по умолчанию:

```bash
uv run --no-sync had train
```

Один запуск Transformer:

```bash
uv run --no-sync had train model=original_transformer training.batch_size=32
```

[Lightning](https://lightning.ai/docs/pytorch/stable/) выполняет обучение и validation.
Ошибка — сумма CrossEntropy по девяти атрибутам, оптимизатор AdamW.
[Hydra](https://hydra.cc/docs/intro/) управляет данными, обработкой, моделью и обучением.

Данные, эмбеддинги и обученные классификаторы версионируются через DVC.

## Production preparation

Веса лучших моделей:

- `models/original_resnet_peta.pt`;
- `models/original_transformer_peta.pt` и `models/original-minicpm-visual`.

Экспорт ResNet в ONNX и сравнение с PyTorch для batch 1 и 2:

```bash
uv run --no-sync had export
```

Вход `[batch, 3, 224, 224]`, выход `[batch, 53]`.
`had export model=original_transformer` экспортирует классификатор:
вход `embeddings` формы `[batch, 1, 64, 3584]`. Перед ним нужны исходные MiniCPM-эмбеддинги.

MLflow Serving принимает обычные кропы и включает извлечение признаков:

```bash
uv run --no-sync had package model=original_transformer
uv run --no-sync mlflow models serve -m var/serving-model --env-manager local --host 127.0.0.1 --port 5001
uv run --no-sync python scripts/request_prediction.py person.jpg
```

По умолчанию сервер использует CPU. Для CUDA перед запуском сервера задайте
`HAD_DEVICE=cuda`. Для повторной упаковки укажите новую папку
через `production.mlflow_model`.
Дополнительные команды `had tensorrt` и `had triton` требуют установленного NVIDIA ПО.

При установленных TensorRT и Triton для ResNet:

```bash
uv run --no-sync had export
uv run --no-sync had tensorrt production.max_batch_size=8
uv run --no-sync had triton
tritonserver --model-repository=var/triton-models
uv run --no-sync python scripts/request_triton.py person.jpg
```

## Infer

Предсказание ResNet на кропе и на первых десяти тестовых изображениях:

```bash
uv run --no-sync had infer inference.image=person.jpg
uv run --no-sync had infer inference.limit=10
```

MiniCPM + Transformer:

```bash
uv run --no-sync had infer model=original_transformer inference.device=cuda inference.image=person.jpg
```

Результат — `var/predictions.jsonl`, девять атрибутов на английском.
`inference.confidence_threshold` задаёт дополнительный порог для ответа `unknown`.

Оценка на тесте:

```bash
uv run --no-sync had evaluate inference.device=cuda inference.batch_size=8 inference.output=var/resnet-test.json
uv run --no-sync had evaluate model=original_transformer inference.device=cuda inference.output=var/transformer-test.json
```

Отчёт содержит accuracy, precision, recall, F1, support и confusion matrix каждого класса.

### Видео

```powershell
$env:HAD_ATTRIBUTE_BACKEND = "peta"
$env:HAD_ATTRIBUTE_CHECKPOINT = "models/original_transformer_peta.pt"
$env:HAD_DETECTOR_MODEL = "models/yolo.onnx"
$env:HAD_DEVICE = "cuda"
uv run --no-sync had-api
```

Для видео отдельно положите ONNX-веса детектора людей по пути `HAD_DETECTOR_MODEL`.
Для ResNet укажите `models/original_resnet_peta.pt`, устройство CPU или CUDA.
API возвращает девять атрибутов на русском. Для локального запуска без ключа запросы выглядят так. Если задан `HAD_API_KEY`, добавьте к каждому запросу заголовок `X-API-Key`.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/jobs -F "file=@person.mp4"
curl http://127.0.0.1:8000/api/v1/jobs/JOB_ID
curl -o result.mp4 http://127.0.0.1:8000/api/v1/jobs/JOB_ID/result
```

## Самостоятельные проверки

```bash
uv run --no-sync pre-commit run -a
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync python scripts/check_project.py
```

Для общего скрипта нужен запущенный MLflow. CPU-окружение выполняет короткое
обучение исходных ResNet50, infer, ONNX и проверку MLflow-пакета.

Видеопайплайн:

```bash
uv run --no-sync python scripts/check_peta_video.py --video person.mp4 --yolo models/yolo.onnx --checkpoint models/original_transformer_peta.pt --device cuda
```

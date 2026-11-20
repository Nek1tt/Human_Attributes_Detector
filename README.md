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

Нужны Git и Python 3.11 или 3.12. Все команды выполняются из корня репозитория
в активированном виртуальном окружении. Для NVIDIA GPU используется CUDA 12.8.

```powershell
git clone --branch detector-remaster --recurse-submodules https://github.com/Nek1tt/Human_Attributes_Detector.git
cd Human_Attributes_Detector
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pre-commit install
pre-commit run -a
```

В Linux окружение активируется командой `source .venv/bin/activate` вместо
PowerShell-команды активации. Установка через `pip` одинакова.

Для CPU-инференса и проверок ResNet установите в новом окружении CPU-профиль:

```bash
python -m pip install -r requirements-cpu.txt
```

`requirements.txt` содержит GPU-зависимости для обеих моделей, обучения, API и
проверок. `requirements-cpu.txt` содержит CPU-зависимости ResNet и проверок.
Минимальные профили API в `requirements/api-cpu.txt` и `requirements/api-gpu.txt`
используются в Dockerfile. Каждый профиль также устанавливает сам пакет проекта.

Версии в requirements закреплены и выгружены из `uv.lock`; зависимости остаются
под управлением [uv](https://docs.astral.sh/uv/) и `pyproject.toml`.
После изменения зависимостей разработчик обновляет lockfile и выполняет:

```bash
uv lock
python scripts/export_requirements.py
python scripts/export_requirements.py --check
```

## Готовые модели

Финальные классификаторы доступны в
[GitHub Release detector-remaster-2026-11-20](https://github.com/Nek1tt/Human_Attributes_Detector/releases/tag/detector-remaster-2026-11-20).
Релиз содержит оба checkpoint, детектор `yolo.onnx`, два отчёта `benchmark.json`
и описание среды замеров.
Размер и SHA-256 весов закреплены в [манифесте](docs/release-weights.json).

Загрузка готовых моделей после установки зависимостей из Setup:

```powershell
python scripts/download_release_weights.py
had original-download
```

Загрузчик проверяет SHA-256 и сохраняет существующие файлы с другим содержимым.
Детектор `models/yolo.onnx` хранится прямо в ветке `detector-remaster` и загружается
вместе с кодом при `git clone`. Загрузчик также проверяет его SHA-256 и может
восстановить файл из релиза. В замерах использован этот же
`yolov8s_576x1024_v2.onnx`: вход
`[1, 3, 576, 1024]`, один класс `person`, SHA-256
`c0d4889317191548e8c18a31b4b91f8d2b28842102eebe4523f9f1593a9ea933`.
Метаданные детектора сохранены без изменений и указывают AGPL-3.0.
В репозитории находятся [текст лицензии](licenses/yolo-agpl-3.0.txt) и
[описание артефакта](docs/detector.json) со ссылкой на исходный код Ultralytics 8.3.48.

После подготовки детектора запустите обе модели на одном видео командами
из раздела «Самостоятельные проверки» ниже либо запустите API из раздела «Видео».
Transformer использует отдельно загружаемый визуальный энкодер MiniCPM;
он не входит в checkpoint классификатора и в файлы релиза.

В `models/` лежат два DVC-указателя на обученные классификаторы. Для запуска готовых
весов на новом компьютере понадобится копия каталога `.dvc-storage/` с объектами этих
двух моделей в корне проекта. Восстановить файлы можно одной командой:

```bash
dvc pull models/original_resnet_peta.pt.dvc models/original_transformer_peta.pt.dvc
```

Чтобы подготовить для преподавателя хранилище только с этими весами, на машине
с обученными моделями выполните команды ниже. Передайте каталог `dist/course-weights`
и при получении поместите его в корень клона под именем `.dvc-storage`.

```bash
dvc remote add --local -f course dist/course-weights
dvc push -r course models/original_resnet_peta.pt.dvc models/original_transformer_peta.pt.dvc
```

ResNet50 после этого готов к предсказанию на кропах. Для Transformer отдельно
загрузите визуальный энкодер MiniCPM из закреплённой ревизии Hugging Face
в `models/original-minicpm-visual/` (около 0.91 ГиБ):

```bash
had original-download
```

Если DVC-хранилище недоступно, классификаторы можно обучить на PETA по инструкции ниже.

## Train

В отдельном терминале запустить MLflow:

```bash
mlflow server --host 127.0.0.1 --port 8080 --workers 1 --backend-store-uri sqlite:///var/mlflow.db --default-artifact-root ./var/mlartifacts
```

Скачать PETA для обучения:

```bash
had download
```

Для обучения Transformer дополнительно подготовить эмбеддинги:

```bash
had original-download
had original-extract
```

Два варианта обучения каждой исходной архитектуры:

```bash
python scripts/train_original.py --skip-extraction
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
had train
```

Один запуск Transformer:

```bash
had train model=original_transformer training.batch_size=32
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
had export
```

Вход `[batch, 3, 224, 224]`, выход `[batch, 53]`.
`had export model=original_transformer` экспортирует классификатор:
вход `embeddings` формы `[batch, 1, 64, 3584]`. Перед ним нужны исходные MiniCPM-эмбеддинги.

MLflow Serving принимает обычные кропы и включает извлечение признаков:

```bash
had package model=original_transformer
mlflow models serve -m var/serving-model --env-manager local --host 127.0.0.1 --port 5001
python scripts/request_prediction.py person.jpg
```

По умолчанию сервер использует CPU. Для CUDA перед запуском сервера задайте
`HAD_DEVICE=cuda`. Для повторной упаковки укажите новую папку
через `production.mlflow_model`.
Дополнительные команды `had tensorrt` и `had triton` требуют установленного NVIDIA ПО.

При установленных TensorRT и Triton для ResNet:

```bash
had export
had tensorrt production.max_batch_size=8
had triton
tritonserver --model-repository=var/triton-models
python scripts/request_triton.py person.jpg
```

## Infer

Предсказание ResNet на кропе и на первых десяти тестовых изображениях:

```bash
had infer inference.image=person.jpg
had infer inference.limit=10
```

MiniCPM + Transformer:

```bash
had infer model=original_transformer inference.device=cuda inference.image=person.jpg
```

Результат — `var/predictions.jsonl`, девять атрибутов на английском.
`inference.confidence_threshold` задаёт дополнительный порог для ответа `unknown`.

Оценка на тесте:

```bash
had evaluate inference.device=cuda inference.batch_size=8 inference.output=var/resnet-test.json
had evaluate model=original_transformer inference.device=cuda inference.output=var/transformer-test.json
```

Отчёт содержит accuracy, precision, recall, F1, support и confusion matrix каждого класса.

### Видео

```powershell
$env:HAD_ATTRIBUTE_BACKEND = "peta"
$env:HAD_ATTRIBUTE_CHECKPOINT = "models/original_transformer_peta.pt"
$env:HAD_DETECTOR_MODEL = "models/yolo.onnx"
$env:HAD_DEVICE = "cuda"
had-api
```

Для видео используется включённый в ветку `models/yolo.onnx`.
`HAD_DETECTOR_MODEL` позволяет указать другой совместимый ONNX-детектор.
Для ResNet укажите `models/original_resnet_peta.pt`, устройство CPU или CUDA.
API возвращает девять атрибутов на русском. Для локального запуска без ключа запросы выглядят так. Если задан `HAD_API_KEY`, добавьте к каждому запросу заголовок `X-API-Key`.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/jobs -F "file=@person.mp4"
curl http://127.0.0.1:8000/api/v1/jobs/JOB_ID
curl -o result.mp4 http://127.0.0.1:8000/api/v1/jobs/JOB_ID/result
```

## Самостоятельные проверки

```bash
pre-commit run -a
python -m unittest discover -s tests -v
python scripts/check_project.py
```

Для общего скрипта нужен запущенный MLflow. CPU-окружение выполняет короткое
обучение исходных ResNet50, infer, ONNX и проверку MLflow-пакета.

Видеопайплайн:

```bash
python scripts/check_peta_video.py --video person.mp4 --yolo models/yolo.onnx --checkpoint models/original_transformer_peta.pt --device cuda
```

Для сравнения моделей на одном видео запустите из PowerShell в GPU-окружении:

```powershell
New-Item -ItemType Directory -Force var/benchmark | Out-Null
curl.exe -fL https://raw.githubusercontent.com/opencv/opencv/4.10.0/samples/data/vtest.avi -o var/benchmark/vtest.avi
foreach ($model in 'resnet', 'transformer') {
    python scripts/check_peta_video.py --video var/benchmark/vtest.avi --yolo models/yolo.onnx --checkpoint "models/original_${model}_peta.pt" --device cuda --output "var/benchmark/$model"
    if ($LASTEXITCODE -ne 0) { throw "Video check failed for $model" }
}
```

В каждом каталоге результата появится `benchmark.json`. Время в секундах включает чтение видео, детекцию на CPU, трекинг, классификацию и запись MP4 и JSONL. Загрузка моделей выполняется до запуска таймера. Пик памяти показывает объём видеопамяти, занятой тензорами PyTorch, включая веса модели. Каждый классификатор запускается в отдельном процессе.

Фактический прогон 20 ноября 2026 года на RTX 4060 Ti 16 ГБ, Windows 11,
Python 3.12.10, PyTorch 2.8.0+cu128:

| Классификатор         | Обработка, с | Пик PyTorch, МиБ |
| --------------------- | -----------: | ---------------: |
| Девять ResNet50       |       149.29 |          1394.41 |
| MiniCPM + Transformer |       145.91 |          1422.03 |

Один прогон каждой модели, без отдельного прогрева; видео 768×576, 795 кадров,
10 FPS, обработаны 398 кадров при целевых 5 FPS. В обоих результатах 2355 записей
о людях; классификация выполняется один раз на трек, а не на каждом кадре.
Пик измерен через `torch.cuda.max_memory_allocated` и не включает память
драйвера или ONNX Runtime. Счётчик сбрасывается перед загрузкой классификатора.
[Исходные JSON и среда замера](docs/benchmarks/rtx4060ti) также приложены к релизу.

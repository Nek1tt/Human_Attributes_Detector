# Определение атрибутов людей

Автор: Абрамов Никита.

Проект определяет девять атрибутов человека: пол, возрастную группу, головной убор,
цвет и тип верхней и нижней одежды, наличие рюкзака и сумки.
Для видео используется YOLO → SFSORT → классификатор. Результат — MP4 с подписями
и JSONL с атрибутами каждого трека.

## Архитектуры из описания проекта

Бейзлайн — **девять независимых ResNet50**, по одной на атрибут.
RGB-кроп 224×224, нормализация ImageNet, классификаторы 2048 → 1024 → 512 → классы.

Основная модель — **MiniCPM-o 2.6 INT4 + исходный VisionAttrTransformer**:

- VPM и Resampler MiniCPM возвращают 64 токена размерности 3584;
- проекция 3584 → 768;
- шесть Transformer-слоёв, 12 голов внимания;
- CLS-токен, attention pooling и девять отдельных классификаторов.

Используются классы `ResNetEnsemble` и `VisionAttrTransformer` из исходного кода.
Изменены только списки выходных классов под новый датасет.
Языковая и аудиочасти MiniCPM не вызываются при извлечении этих признаков.
Загрузчик сохраняет исходные визуальные компоненты и выдаёт те же эмбеддинги.

## Данные

Используется [PETA](https://mmlab.ie.cuhk.edu.hk/projects/PETA.html).
В нём есть возраст, типы и цвета одежды, аксессуары и сумки.
Market-1501 не содержит нужных типов верхней одежды.
В исходной разметке PA100K отсутствуют цвета одежды:
их отдельно добавляли авторы [UPAR](https://openaccess.thecvf.com/content/WACV2023/supplemental/Specker_UPAR_Unified_Pedestrian_WACV_2023_supplemental.pdf).

Часть i-LID исключена из-за отдельного разрешения на её использование.
Остаются 18 523 изображения: 11 060 train, 3887 validation и 3576 test.
ID из разметки делятся внутри поддатасетов с seed 42. Один ID находится только в одной части.

| Атрибут       | Классы PETA без «не определен»                                  |
| ------------- | --------------------------------------------------------------- |
| Пол           | Мужчина, женщина                                                |
| Возраст       | Младшая группа personalLess15, 16–30, 31–45, 46–60, старше 60   |
| Головной убор | Нет, есть: шляпа/шапка или платок                               |
| Цвет верха    | 11 цветов                                                       |
| Цвет низа     | 11 цветов                                                       |
| Тип верха     | Куртка, свитер, костюм, футболка, другая одежда                 |
| Тип низа      | Брюки, шорты, юбка, капри                                       |
| Рюкзак        | Нет, есть                                                       |
| Сумка         | Нет, есть: сумка через плечо, пакет, чемодан или багажная сумка |

У каждого атрибута также есть класс «не определен». Всего 53 выхода.
Противоречивые и неуказанные категориальные метки относятся к этому классу.
Вычисление классов описано в `peta_data.py` и `original_features.py`.

Описание: [proposal.md](docs/proposal.md), [PDF](docs/proposal.pdf).
Разбор критериев: [criteria.md](docs/criteria.md).
Обучение и эксперименты: [original-training.md](docs/original-training.md).
Результаты: [original-results.md](docs/original-results.md).

## Setup

Нужны Git, Python 3.11 или 3.12 и [uv](https://docs.astral.sh/uv/).
Все команды выполняются из корня репозитория.

```bash
git clone --recurse-submodules https://github.com/Nek1tt/Human_Attributes_Detector.git
cd Human_Attributes_Detector
git switch detector-remaster
python -m pip install uv
uv sync --locked --extra gpu --extra train --extra dev --extra vision --extra api
uv run --no-sync pre-commit install
uv run --no-sync pre-commit run -a
```

Для обучения используется NVIDIA GPU. Для CPU-инференса ResNet вместо GPU/vision
достаточно `uv sync --locked --extra cpu --extra train --extra dev --extra export --extra api`.
Версии зависимостей закреплены в `uv.lock`.

## Train

В отдельном терминале запустить MLflow:

```bash
uv run --no-sync mlflow server --host 127.0.0.1 --port 8080 --workers 1 --backend-store-uri sqlite:///var/mlflow.db --default-artifact-root ./var/mlartifacts
```

Подготовить данные и исходные эмбеддинги:

```bash
uv run --no-sync had download
uv run --no-sync had original-download
uv run --no-sync had original-extract
```

Два варианта обучения каждой исходной архитектуры:

```bash
uv run --no-sync python scripts/train_original.py --skip-extraction
```

Если сохранились прежние веса, их можно использовать как начальные:

```powershell
uv run --no-sync python scripts/train_original.py --skip-extraction --resnet-weights "C:/Human_Attributes_Detector/models/resnet_ens_11.19_e60_s0.782.pt" --transformer-weights "C:/Human_Attributes_Detector/models/MiniCPM-o 2.6int4 weights.pt"
```

По умолчанию выполняется до четырёх эпох на эксперимент. Меняются learning rate
и взвешивание ошибки. Сохраняется лучший checkpoint по validation accuracy.
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

Данные, исходные визуальные веса, эмбеддинги и checkpoint хранятся через DVC.
Local remote — `.dvc-storage`. На новом компьютере датасет и визуальные веса
скачиваются из закреплённых открытых источников. Готовые веса передаются
с DVC-хранилищем или получаются повторным обучением.

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

Для повторной упаковки задайте новую папку через `production.mlflow_model`.
Дополнительные команды `had tensorrt` и `had triton` требуют установленного NVIDIA ПО.

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
$env:HAD_DETECTOR_MODEL = "C:/Human_Attributes_Detector/models/yolov8s_576x1024_v2.onnx"
$env:HAD_DEVICE = "cuda"
uv run --no-sync had-api
```

Для ResNet укажите `models/original_resnet_peta.pt`, устройство CPU или CUDA.
API возвращает девять атрибутов на русском. HTTP-запросы описаны в [video.md](docs/video.md).

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

Старый датасет утерян. Его веса и backends сохранены.
Предыдущие компактные модели описаны в [архиве эксперимента](docs/peta-results.md).

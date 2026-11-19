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

На тестовой части PETA ансамбль ResNet50 получил accuracy **0.7640** и macro F1 **0.5668**.
У MiniCPM + Transformer эти показатели составили **0.7894** и **0.6093**.
В отчёте приведены [метрики по каждому атрибуту и графики обучения](docs/original-results.md).
Для курса доступны [описание проекта в PDF](docs/proposal.pdf), [исходный текст](docs/proposal.md)
и [условия экспериментов](docs/original-training.md).

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

# Результаты исходных моделей на PETA

18 ноября 2026. RTX 4060 Ti 16 ГБ, Python 3.12, PyTorch 2.8, BF16.

Использованы девять исходных ResNet50 и исходный VisionAttrTransformer: 3584 → 768, шесть слоёв, 12 голов внимания. MiniCPM VPM и Resampler заморожены. Классы обновлены под PETA: девять атрибутов, 53 выхода. Начальные веса взяты из прежних моделей автора.

Данные: 11 060 train, 3887 validation, 3576 test. ID людей между частями не пересекаются. Seed 42. Лучший checkpoint выбирается по validation accuracy.

## Эксперименты

| Модель                | Вариант  | Эпох выполнено | Лучшая эпоха | Validation accuracy | Validation macro F1 | Время, мин |
| --------------------- | -------- | -------------: | -----------: | ------------------: | ------------------: | ---------: |
| Девять ResNet50       | standard |              4 |            3 |              0.7832 |              0.5623 |       51.7 |
| Девять ResNet50       | balanced |              4 |            4 |              0.7613 |              0.5574 |       51.3 |
| MiniCPM + Transformer | standard |              3 |            1 |              0.7983 |              0.6079 |        2.0 |
| MiniCPM + Transformer | balanced |              4 |            4 |              0.7855 |              0.6232 |        2.5 |
| MiniCPM + Transformer | refined  |              3 |            1 |              0.7976 |              0.6240 |        2.0 |

standard — обычная CrossEntropy. balanced — веса классов по обратному квадратному корню частоты в train. refined — дообучение лучшего Transformer с меньшим learning rate. Настройки указаны в [original-training.md](original-training.md).

Лучшие по accuracy — standard для обеих архитектур. Взвешивание ошибки и дообучение повысили macro F1 Transformer, но accuracy осталось ниже выбранного варианта.

## Итоговая оценка на test

| Модель                | Accuracy | Macro F1 | Файл весов                            |
| --------------------- | -------: | -------: | ------------------------------------- |
| Девять ResNet50       |   0.7640 |   0.5668 | `models/original_resnet_peta.pt`      |
| MiniCPM + Transformer |   0.7894 |   0.6093 | `models/original_transformer_peta.pt` |

## Девять атрибутов на test

| Атрибут       | ResNet accuracy | ResNet macro F1 | Transformer accuracy | Transformer macro F1 |
| ------------- | --------------: | --------------: | -------------------: | -------------------: |
| Пол           |          0.8328 |          0.8328 |               0.8744 |               0.8733 |
| Возраст       |          0.5968 |          0.3811 |               0.6219 |               0.4001 |
| Головной убор |          0.9312 |          0.7165 |               0.9256 |               0.7261 |
| Цвет верха    |          0.6658 |          0.5522 |               0.6971 |               0.6577 |
| Цвет низа     |          0.7204 |          0.3704 |               0.7212 |               0.3910 |
| Тип верха     |          0.6538 |          0.4146 |               0.6890 |               0.4647 |
| Тип низа      |          0.9192 |          0.4067 |               0.9211 |               0.4322 |
| Рюкзак        |          0.8965 |          0.8197 |               0.8865 |               0.8105 |
| Сумка         |          0.6591 |          0.6069 |               0.7673 |               0.7279 |

Macro F1 — среднее F1 по классам атрибута, затем среднее по девяти атрибутам. Классы без примеров и предсказаний исключаются.

## Графики и полные результаты

Графики показывают обучение двух вариантов standard, выбранных по validation accuracy.
В JSON для каждого класса приведены precision, recall, F1, support и матрица ошибок.

Сохранены исходные журналы пяти экспериментов:

- [ResNet standard](../plots/original-resnet-standard/training/version_0/);
- [ResNet balanced](../plots/original-resnet-balanced/training/version_0/);
- [Transformer standard](../plots/original-transformer-standard/training/version_0/);
- [Transformer balanced](../plots/original-transformer-balanced/training/version_0/);
- [Transformer refined](../plots/original-transformer-refined/training/version_0/).

Каждый каталог содержит `metrics.csv`, `config.yaml`, `hparams.yaml`,
`code-version.txt` и три графика. Версия кода и признак `git_dirty` относятся
к моменту обучения и сохранены без изменений. Итоги запусков собраны в
[original-experiments.json](../plots/original-experiments.json), замеры батчей
и оценка ресурсов — в [original-resources.json](../plots/original-resources.json).

| Модель      | Loss                                          | Accuracy                                              | Macro F1                                              | Отчёт на test                                  |
| ----------- | --------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------- | ---------------------------------------------- |
| ResNet50    | [Loss](../plots/results/resnet/loss.png)      | [Accuracy](../plots/results/resnet/accuracy.png)      | [Macro F1](../plots/results/resnet/macro_f1.png)      | [JSON](../plots/results/resnet/test.json)      |
| Transformer | [Loss](../plots/results/transformer/loss.png) | [Accuracy](../plots/results/transformer/accuracy.png) | [Macro F1](../plots/results/transformer/macro_f1.png) | [JSON](../plots/results/transformer/test.json) |

## Время и место

Пять запусков обучения заняли суммарно около 110 минут. Перед Transformer один раз сохранены MiniCPM-эмбеддинги всех изображений. Сам классификатор обучается на 64 токенах: 11 060 примеров, до четырёх эпох на запуск.

Замер времени на обучающем батче: ResNet — 0.500 с на обучающий batch из 8 изображений, Transformer — 0.088 с на batch из 32. Одна эпоха занимает примерно 12.9 минуты и 33 секунды соответственно.

Визуальные веса MiniCPM занимают 0.91 ГиБ, эмбеддинги — 7.9 ГиБ.

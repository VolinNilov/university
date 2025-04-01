# Проект: Обучение и инференс модели YOLOv8 для обнаружения пауков

Этот проект направлен на обучение модели YOLOv8 для обнаружения пауков на изображениях. Проект включает:
- Подготовку датасета (изображения и метки).
- Обучение модели с использованием библиотеки Ultralytics.
- Инференс модели на тестовых данных.
- Создание видео из обработанных изображений.

## Содержание
1. [Структура проекта](#структура-проекта)
2. [Обучение модели](#обучение-модели)
3. [Инференс модели](#инференс-модели)

## Структура проекта
``` bach
C:.
├───dataset/
│   ├───spider_dataset/
│   │   ├───test/
│   │   │   ├───images/
│   │   │   └───labels/
│   │   ├───train/
│   │   │   ├───images/
│   │   │   └───labels/
│   │   └───valid/
│   │       ├───images/
│   │       └───labels/
│   └───spider_dataset_102/
│       ├───train/
│       │   ├───images/
│       │   └───labels/
│       └───val/
│           ├───images/
│           └───labels/
├───models/
│   ├───vemous_spiders_big_yolov8n/
│   │   └───weights/
│   └───vemous_spiders_yolov8n/
│       └───weights/
├───inference/
├───inference.py
├───main.py
├───output_video.mp4
└───yolov8n.pt
```

## Обучение модели
Для обучения модели используйте файл main.py. Этот файл содержит конфигурацию для обучения модели YOLOv8.

**Основные параметры:**
Параметры используемые при обучении модели (одинаковые как для модели обученной на малой выборке - 102 картинки - **vemous_spiders_yolov8n**, так и для большой модели - train: 1403; val: 51; test: 51 - **vemous_spiders_big_yolov8n**):
``` yaml
# Задача, которую решает модель (обнаружение объектов)
task: detect

# Режим работы: обучение модели
mode: train

# Используемая предобученная модель (веса YOLOv8n)
model: yolov8n.pt

# Путь к файлу конфигурации датасета (data.yaml), содержащему пути к изображениям и меткам
data: path/to/data.yaml

# Количество эпох для обучения модели
epochs: 500

# Время ограничения обучения (null = без ограничения)
time: null

# Количество эпох без улучшения, после которых обучение будет остановлено
patience: 60

# Размер батча (-1 = автоматический выбор)
batch: -1

# Размер входных изображений (в пикселях)
imgsz: 640

# Флаг сохранения модели после обучения
save: true

# Периодичность сохранения контрольных точек (checkpoints) (-1 = только последняя эпоха)
save_period: -1

# Кэширование данных в оперативной памяти (false = отключено)
cache: false

# Устройство для обучения (cuda = GPU, cpu = процессор)
device: cuda

# Количество потоков для загрузки данных
workers: 0

# Имя проекта (папка для сохранения результатов)
project: models

# Имя эксперимента (подпапка внутри папки проекта)
name: name

# Разрешить перезапись существующей папки с результатами
exist_ok: false

# Использовать предобученные веса для инициализации модели
pretrained: true

# Оптимизатор для обучения (AdamW обеспечивает хороший баланс между скоростью и стабильностью)
optimizer: AdamW

# Вывод подробной информации о процессе обучения
verbose: true

# Фиксированное значение seed для воспроизводимости результатов
seed: 0

# Включение детерминированных вычислений (для воспроизводимости)
deterministic: true

# Обучение модели как одноклассовой задачи (false = многоклассовая)
single_cls: false

# Использование прямоугольных изображений (без изменения соотношения сторон)
rect: false

# Использование косинусоидального расписания для скорости обучения
cos_lr: false

# Количество эпох перед завершением обучения, в течение которых мозаика отключается
close_mosaic: 10

# Возобновление обучения с контрольной точки
resume: false

# Использование автоматического микширования точности (AMP) для ускорения обучения
amp: true

# Доля данных, используемых для обучения (1.0 = все данные)
fraction: 1.0

# Профилирование производительности (false = отключено)
profile: false

# Количество слоев, которые будут заморожены при обучении
freeze: null

# Использование нескольких масштабов изображений для обучения
multi_scale: false

# Перекрытие масок для задач сегментации
overlap_mask: true

# Соотношение масок для задач сегментации
mask_ratio: 4

# Вероятность отключения нейронов для предотвращения переобучения
dropout: 0.0

# Проведение валидации во время обучения
val: true

# Набор данных для валидации (val = валидационный набор)
split: val

# Сохранение результатов в формате JSON
save_json: false

# Сохранение гибридных контрольных точек
save_hybrid: false

# Порог уверенности для обнаружения объектов
conf: null

# Порог Intersection over Union (IoU) для подавления немаксимумов
iou: 0.7

# Максимальное количество обнаруженных объектов на изображении
max_det: 300

# Использование полупрецизионных вычислений (half-precision)
half: false

# Использование DNN для вывода
dnn: false

# Генерация графиков и диаграмм после обучения
plots: true

# Источник данных для инференса (null = не используется при обучении)
source: null

# Шаг видео для обработки (1 = каждый кадр)
vid_stride: 1

# Буферизация потока видео (false = отключено)
stream_buffer: false

# Визуализация процесса обучения (false = отключено)
visualize: false

# Применение аугментации данных при инференсе
augment: false

# Агностическое подавление немаксимумов (agnostic NMS)
agnostic_nms: false

# Фильтрация классов для обнаружения
classes: null

# Использование высококачественных масок для задач сегментации
retina_masks: false

# Встраивание дополнительных данных
embed: null

# Показывать результаты в реальном времени
show: false

# Сохранение кадров видео
save_frames: false

# Сохранение результатов в текстовом формате
save_txt: false

# Сохранение уверенности модели вместе с результатами
save_conf: false

# Сохранение обрезанных областей объектов
save_crop: false

# Отображение меток классов на изображениях
show_labels: true

# Отображение уверенности модели на изображениях
show_conf: true

# Отображение рамок вокруг объектов
show_boxes: true

# Толщина линий для рамок (null = автоматический выбор)
line_width: null

# Формат экспорта модели (torchscript = для использования в PyTorch)
format: torchscript

# Экспорт модели в Keras (false = отключено)
keras: false

# Оптимизация модели для мобильных устройств
optimize: false

# Использование INT8-квантизации для оптимизации
int8: false

# Динамическая оснастка для экспорта модели
dynamic: false

# Упрощение модели для экспорта
simplify: true

# Версия ONNX Opset для экспорта
opset: null

# Рабочая область для экспорта
workspace: null

# Применение Non-Maximum Suppression (NMS)
nms: false

# Начальная скорость обучения
lr0: 0.01

# Конечная скорость обучения
lrf: 0.01

# Коэффициент импульса для оптимизатора
momentum: 0.937

# Коэффициент регуляризации весов
weight_decay: 0.0005

# Количество эпох для разогрева (warmup)
warmup_epochs: 3.0

# Импульс для разогрева
warmup_momentum: 0.8

# Скорость обучения для смещения (bias) при разогреве
warmup_bias_lr: 0.1

# Вес для потерь по ограничивающим рамкам
box: 7.5

# Вес для потерь по классам
cls: 0.5

# Вес для потерь по распределению расстояний
dfl: 1.5

# Вес для потерь по позе (pose estimation)
pose: 12.0

# Вес для потерь по объектам
kobj: 1.0

# Базовый размер батча для нормализации потерь
nbs: 64

# Изменение оттенка цвета (hue) при аугментации
hsv_h: 0.015

# Изменение насыщенности цвета (saturation) при аугментации
hsv_s: 0.7

# Изменение яркости цвета (value) при аугментации
hsv_v: 0.4

# Угол поворота изображений при аугментации
degrees: 0.0

# Перемещение изображений по осям X и Y при аугментации
translate: 0.1

# Масштабирование изображений при аугментации
scale: 0.5

# Сдвиг изображений по осям X и Y при аугментации
shear: 0.0

# Применение перспективных искажений
perspective: 0.0

# Вероятность переворота изображений по вертикали
flipud: 0.0

# Вероятность переворота изображений по горизонтали
fliplr: 0.5

# Инверсия каналов BGR
bgr: 0.0

# Вероятность применения мозаики (mosaic augmentation)
mosaic: 1.0

# Вероятность применения микширования изображений (mixup augmentation)
mixup: 0.0

# Вероятность применения копирования и вставки объектов
copy_paste: 0.0

# Режим копирования и вставки объектов
copy_paste_mode: flip

# Автоматическая аугментация данных
auto_augment: randaugment

# Вероятность случайного стирания частей изображения
erasing: 0.4

# Доля обрезки изображения
crop_fraction: 1.0

# Файл конфигурации модели (null = используется стандартный)
cfg: null

# Файл конфигурации трекера объектов
tracker: botsort.yaml

# Путь для сохранения результатов обучения
save_dir: path
```

Обе модели обучились и имеют следующие характеристики:
- Модель на малой выборке данных - **vemous_spiders_yolov8n**:
    - **Confusion Matrix Normalized** [более подробно про эту характеристику можно прочитать тут, в справочных материал](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/definitions/confusion_matrix_normalized.md)
        ![Confusion Matrix Normalized](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/models/vemous_spiders_yolov8n/confusion_matrix_normalized.png)
    
    - **Confusion Matrix** [более подробно про эту характеристику можно прочитать тут, в справочных материал](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/definitions/confusion_matrix.md)
        ![Confusion Matrix](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/models/vemous_spiders_yolov8n/confusion_matrix.png)
    
    - **Lasbles Correlogram** [более подробно про эту характеристику можно прочитать тут, в справочных материал](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/definitions/labels_correlogram.md)
        ![Lasbles Correlogram](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/models/vemous_spiders_yolov8n/labels_correlogram.jpg)
    
    - **F1** [более подробно про эту характеристику можно прочитать тут, в справочных материал](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/definitions/F1_curve.md)
        ![F1](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/models/vemous_spiders_yolov8n/F1_curve.png)
    
    - **PR** [более подробно про эту характеристику можно прочитать тут, в справочных материал](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/definitions/PR_curve.md)
        ![PR](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/models/vemous_spiders_yolov8n/PR_curve.png)
    
    - **P** [более подробно про эту характеристику можно прочитать тут, в справочных материал](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/definitions/P_curve.md)
        ![P](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/models/vemous_spiders_yolov8n/P_curve.png)

    - **R** [более подробно про эту характеристику можно прочитать тут, в справочных материал](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/definitions/R_curve.md)
        ![R](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/models/vemous_spiders_yolov8n/R_curve.png)

    Результат характеристик для модели **vemous_spiders_yolov8n**:
    ![vemous_spiders_yolov8n](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/models/vemous_spiders_yolov8n/results.png)

- Модель на большой выборке данных - **vemous_spiders_big_yolov8n**:
    - **Confusion Matrix Normalized** [более подробно про эту характеристику можно прочитать тут, в справочных материал](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/definitions/confusion_matrix_normalized.md)
        ![Confusion Matrix Normalized](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/models/vemous_spiders_big_yolov8n/confusion_matrix_normalized.png)
    
    - **Confusion Matrix** [более подробно про эту характеристику можно прочитать тут, в справочных материал](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/definitions/confusion_matrix.md)
        ![Confusion Matrix](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/models/vemous_spiders_big_yolov8n/confusion_matrix.png)
    
    - **Lasbles Correlogram** [более подробно про эту характеристику можно прочитать тут, в справочных материал](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/definitions/labels_correlogram.md)
        ![Lasbles Correlogram](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/models/vemous_spiders_big_yolov8n/labels_correlogram.jpg)
    
    - **F1** [более подробно про эту характеристику можно прочитать тут, в справочных материал](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/definitions/F1_curve.md)
        ![F1](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/models/vemous_spiders_big_yolov8n/F1_curve.png)
    
    - **PR** [более подробно про эту характеристику можно прочитать тут, в справочных материал](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/definitions/PR_curve.md)
        ![PR](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/models/vemous_spiders_big_yolov8n/PR_curve.png)
    
    - **P** [более подробно про эту характеристику можно прочитать тут, в справочных материал](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/definitions/P_curve.md)
        ![P](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/models/vemous_spiders_big_yolov8n/P_curve.png)

    - **R** [более подробно про эту характеристику можно прочитать тут, в справочных материал](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/definitions/R_curve.md)
        ![R](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/models/vemous_spiders_big_yolov8n/R_curve.png)

    Результат характеристик для модели **vemous_spiders_big_yolov8n**:
    ![vemous_spiders_big_yolov8n](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/models/vemous_spiders_big_yolov8n/results.png)


## Инференс модели
Инференс модели **vemous_spidersyolov8n**: 
![vemous_spidersyolov8n](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/definitions/illustrations/vemous_spiders_yolov8n.gif)
***Неудачный опыт, как видно на кадрах выше, у модели плохо получается детектировать объекты на изображении***

Инференс модели **vemous_spiders_big_yolov8n**:
![vemous_spiders_big_yolov8n](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/definitions/illustrations/vemous_spiders_big_yolov8n.gif)
***Удачный опыт, как видно на кадрах выше, у модели хорошо получается детектировать объекты на изображении. Видны оба класса и можно сделать вывод, что за исключением двойного обнаружения, модель ведёт себя прекрасно и выполняет свою работу)***
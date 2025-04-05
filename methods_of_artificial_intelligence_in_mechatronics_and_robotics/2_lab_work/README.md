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
│   │   │   |   ├───...
│   │   │   └───labels/
│   │   │       └───...
│   │   ├───train/
│   │   │   ├───images/
│   │   │   |   ├───...
│   │   │   └───labels/
│   │   │       └───...
│   │   └───valid/
│   │   │   ├───images/
│   │   │   |   ├───...
│   │   │   └───labels/
│   │   │       └───...
│   └───spider_dataset_102/
│       ├───train/
│       │   ├───images/
│       │   |   ├───...
│       │   └───labels/
│       │       └───...
│       └───val/
│       │   ├───images/
│       │   |   ├───...
│       │   └───labels/
│       │       └───...
├───models/
│   ├───vemous_spiders_big_yolov8n/
│   │   └───weights/
│   │       └───...
│   └───vemous_spiders_yolov8n/
│       └───weights/
│           └───...
├───definitions/
│   ├───illustrations/
│   └───...
├───inference.py
├───main.py
├───output_video.mp4
└───yolov8n.pt
```

## Обучение модели
Для обучения модели используйте файл main.py.
Также более подробно про характеристики модели можно прочитать тут:
- **Confusion Matrix Normalized** [более подробно про эту характеристику можно прочитать тут, в справочных материал](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/definitions/confusion_matrix_normalized.md)

- **Confusion Matrix** [более подробно про эту характеристику можно прочитать тут, в справочных материал](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/definitions/confusion_matrix.md)

- **Lasbles Correlogram** [более подробно про эту характеристику можно прочитать тут, в справочных материал](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/definitions/labels_correlogram.md)

- **F1** [более подробно про эту характеристику можно прочитать тут, в справочных материал](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/definitions/F1_curve.md)

- **PR** [более подробно про эту характеристику можно прочитать тут, в справочных материал](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/definitions/PR_curve.md)

- **P** [более подробно про эту характеристику можно прочитать тут, в справочных материал](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/definitions/P_curve.md)

- **R** [более подробно про эту характеристику можно прочитать тут, в справочных материал](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/definitions/R_curve.md)


## Инференс модели

Результат обучения для модели **spider_3_class**:
![spider_3_class](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/models/spiders_3_class/results.png)

Инференс модели **spider_3_class**: 

![spider_3_class без NMS](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/output_video/spiders_3_class/gif/spiders_3_class_without_nms.gif)
![spider_3_class c NMS](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/output_video/spiders_3_class/gif/spiders_3_class_with_nms.gif)

***Распознование 3 классов пауков: Black Widow, Blue Tarantula, Bold Jumper***
***Видео можно посмотреть тут: [видео](https://github.com/VolinNilov/university/tree/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/output_video/spiders_3_class/video)***


## Побочные модели. Модели распознающие ядовитых и не ядовитых пауков:

### Инференс модели [**vemous_spiders_yolov8n**](https://github.com/VolinNilov/university/tree/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/models/vemous_spiders_yolov8n): 

Результат обучения для модели **vemous_spiders_yolov8n**:

![vemous_spiders_yolov8n results](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/models/vemous_spiders_yolov8n/results.png)
![vemous_spiders_yolov8n](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/definitions/illustrations/vemous_spiders_yolov8n.gif)

***Неудачный опыт, как видно на кадрах выше, у модели плохо получается детектировать объекты на изображении***

### Инференс модели [**vemous_spiders_big_yolov8n**](https://github.com/VolinNilov/university/tree/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/models/vemous_spiders_big_yolov8n):

![vemous_spiders_big_yolov8n results](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/models/vemous_spiders_big_yolov8n/results.png)
![vemous_spiders_big_yolov8n](https://github.com/VolinNilov/university/blob/main/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/definitions/illustrations/vemous_spiders_big_yolov8n.gif)

***Удачный опыт, как видно на кадрах выше, у модели хорошо получается детектировать объекты на изображении. Видны оба класса и можно сделать вывод, что за исключением двойного обнаружения, модель ведёт себя прекрасно и выполняет свою работу)***


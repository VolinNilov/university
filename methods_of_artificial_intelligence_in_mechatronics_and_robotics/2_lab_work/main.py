from ultralytics import YOLO
import torch

torch.backends.openmp.enabled = False

#device = "cuda" if torch.cuda.is_available() else "cpu"

model = YOLO("yolov8n.pt")

model.train(
    data="D:/Projects/own/university/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/dataset/spiders_label_studio/data.yaml",
    epochs=500,                     # Количество эпох
    batch=-1,                       # Размер батча (или "auto" для автоматического выбора)
    imgsz=640,                      # Размер изображений
    device="cuda",                   # Автовыбор устройства
    optimizer="AdamW",              # Оптимизатор (SGD, Adam, AdamW)
    patience=60,                    # Количество эпох без улучшения перед остановкой
    workers=0,                      # Количество потоков для загрузки данных
    mosaic=1.0,                     # Использование Mosaic аугментации
    save=True,                      # Сохранение модели после обучения
    project="models",                # Папка для сохранения модели
    name="spiders_label_studio_yolov8n"   # Имя сохраненной модели
)
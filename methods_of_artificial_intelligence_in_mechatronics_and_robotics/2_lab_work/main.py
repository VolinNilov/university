from ultralytics import YOLO
import torch

torch.backends.openmp.enabled = False

#device = "cuda" if torch.cuda.is_available() else "cpu"

model = YOLO("yolov8n.pt")

model.train(
    data="C:/Users/1/Desktop/Учёба/Караваев/Магистратура/1 Лабораторная работа/dataset/spider_dataset/data.yaml",       # Укажите путь к файлу конфигурации датасета
    task=
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
    name="vemous_spiders_big_yolov8n"   # Имя сохраненной модели
)
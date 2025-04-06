import cv2
import torch
import os
import glob
import shutil
import re
import numpy as np
from ultralytics import YOLO
from typing import List
from dotenv import load_dotenv
import imageio  # Для создания GIF

# Глобальные переменные для цвета подложки и текста
BACKGROUND_COLOR = (0, 0, 0)  # BGR формат (чёрный)
TEXT_COLOR = (255, 255, 255)  # Белый цвет для текста

# Загрузка переменных окружения
load_dotenv()

# Извлечение имени датасета из пути
dataset_name = re.split(r'[\\/]', os.getenv("dataset_dir"))[-1]


class InatorBase:
    def __init__(self, model_path: str, test_images_path: str):
        """
        Конструктор класса. Инициализирует модель YOLO и устройство (GPU или CPU).
        :param model_path: Путь к весам модели.
        :param test_images_path: Путь к тестовым изображениям.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path).to(self.device)
        self.test_images_path = test_images_path
        self.processed_images = []

    def process_images(self, processed_dir: str) -> List[str]:
        """
        Обрабатывает изображения и сохраняет их в указанную директорию.
        Возвращает список путей к обработанным изображениям.
        :param processed_dir: Директория для сохранения обработанных изображений.
        :return: Список путей к обработанным изображениям.
        """
        image_paths = glob.glob(os.path.join(self.test_images_path, "*.jpg"))
        os.makedirs(processed_dir, exist_ok=True)  # Создаем директорию для сохранения

        processed_image_paths = []
        for img_path in image_paths:
            image = cv2.imread(img_path)
            results = self.model(image)

            if results and len(results) > 0:
                image_with_boxes = results[0].plot()
            else:
                image_with_boxes = image

            output_path = os.path.join(processed_dir, os.path.basename(img_path))
            cv2.imwrite(output_path, image_with_boxes)
            processed_image_paths.append(output_path)

        self.processed_images = processed_image_paths
        return processed_image_paths

    def create_video(self, fps: int, output_video_path: str) -> str:
        """
        Создаёт видео из обработанных изображений и сохраняет его в output_video_path.
        Возвращает путь к созданному видеофайлу.
        :param fps: Количество кадров в секунду.
        :param output_video_path: Путь для сохранения видеофайла.
        :return: Путь к созданному видеофайлу.
        """
        if not self.processed_images:
            raise ValueError("Нет обработанных изображений. Сначала вызовите process_images().")

        # Размеры подложки и обработанного изображения
        background_size = (900, 900)
        processed_image_size = (640, 640)

        # Вычисляем координаты для размещения обработанного изображения по центру
        x_offset = (background_size[0] - processed_image_size[0]) // 2
        y_offset = (background_size[1] - processed_image_size[1]) // 2

        # Создаем директорию для видео, если её нет
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

        # Инициализация видеозаписи
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, background_size)

        for img_path in self.processed_images:
            # Читаем обработанное изображение
            processed_image = cv2.imread(img_path)
            processed_image = cv2.resize(processed_image, processed_image_size)

            # Создаем чёрную подложку
            background = np.full((background_size[1], background_size[0], 3), BACKGROUND_COLOR, dtype=np.uint8)
            background[y_offset:y_offset + processed_image_size[1], x_offset:x_offset + processed_image_size[0]] = processed_image

            # Получаем информацию о классах и confidence
            results = self.model(cv2.imread(img_path))
            if results and len(results) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                scores = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy()

                text_lines = []
                for score, cls_id in zip(scores, class_ids):
                    class_name = self.model.names[int(cls_id)]
                    text_lines.append(f"{class_name}: {score:.2f}")

                # Добавляем текст в левый верхний угол
                y_text = 30
                for line in text_lines:
                    cv2.putText(
                        background,
                        line,
                        (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        TEXT_COLOR,
                        2
                    )
                    y_text += 30

            # Записываем кадр в видео
            video_writer.write(background)

        video_writer.release()
        return output_video_path


class InatorNMS(InatorBase):
    def process_images(self, processed_dir: str, iou_threshold: float = 0.5, conf_threshold: float = 0.3) -> List[str]:
        """
        Обрабатывает изображения с применением NMS и сохраняет их в указанную директорию.
        Возвращает список путей к обработанным изображениям.
        :param processed_dir: Директория для сохранения обработанных изображений.
        :param iou_threshold: Порог IOU для фильтрации боксов.
        :param conf_threshold: Минимальный порог уверенности для боксов.
        :return: Список путей к обработанным изображениям.
        """
        image_paths = glob.glob(os.path.join(self.test_images_path, "*.jpg"))
        os.makedirs(processed_dir, exist_ok=True)

        processed_image_paths = []
        for img_path in image_paths:
            image = cv2.imread(img_path)
            results = self.model(image)

            if results and len(results) > 0:
                # Применяем NMS к результатам
                boxes = results[0].boxes.xyxy.cpu().numpy()
                scores = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy()

                nms_results = cv2.dnn.NMSBoxes(
                    bboxes=boxes.tolist(),
                    scores=scores.tolist(),
                    score_threshold=conf_threshold,
                    nms_threshold=iou_threshold
                )

                # Проверяем, что результат не пустой
                if isinstance(nms_results, tuple):
                    indices = [] if len(nms_results) == 0 else nms_results[0]
                else:
                    indices = nms_results

                indices = indices.flatten() if hasattr(indices, 'flatten') else indices

                # Если есть индексы, фильтруем боксы
                if len(indices) > 0:
                    filtered_boxes = [boxes[i] for i in indices]
                    filtered_scores = [scores[i] for i in indices]
                    filtered_class_ids = [class_ids[i] for i in indices]

                    for box, score, cls_id in zip(filtered_boxes, filtered_scores, filtered_class_ids):
                        x1, y1, x2, y2 = map(int, box)
                        label = f"{self.model.names[int(cls_id)]} {score:.2f}"
                        color = (0, 255, 0)
                        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                else:
                    print(f"Нет подходящих боксов для изображения: {img_path}")

            output_path = os.path.join(processed_dir, os.path.basename(img_path))
            cv2.imwrite(output_path, image)
            processed_image_paths.append(output_path)

        self.processed_images = processed_image_paths
        return processed_image_paths


class InatorGif:
    def __init__(self, processed_images: List[str]):
        """
        Конструктор класса. Инициализирует список обработанных изображений.
        :param processed_images: Список путей к обработанным изображениям.
        """
        self.processed_images = processed_images

    def create_gif(self, output_gif_path: str, duration: float = 0.5) -> str:
        """
        Создает GIF из обработанных изображений и сохраняет его в указанный путь.
        :param output_gif_path: Путь для сохранения GIF-файла.
        :param duration: Длительность показа каждого кадра в секундах.
        :return: Путь к созданному GIF-файлу.
        """
        if not self.processed_images:
            raise ValueError("Нет обработанных изображений для создания GIF.")

        # Создаем директорию для GIF, если её нет
        os.makedirs(os.path.dirname(output_gif_path), exist_ok=True)

        # Чтение изображений и создание GIF
        images = [imageio.imread(img_path) for img_path in self.processed_images]
        imageio.mimsave(output_gif_path, images, duration=duration)

        return output_gif_path


def process_and_create_video(inator, data_path, video_path, use_nms=False):
    """
    Обрабатывает изображения и создает видео.
    :param inator: Экземпляр класса InatorBase или InatorNMS.
    :param data_path: Директория для сохранения обработанных изображений.
    :param video_path: Путь для сохранения видеофайла.
    :param use_nms: Флаг использования NMS.
    """
    print("[START] Старт обработки изображений:")
    if use_nms:
        processed_images = inator.process_images(data_path, iou_threshold=0.4, conf_threshold=0.3)
    else:
        processed_images = inator.process_images(data_path)
    print("[END] Конец обработки изображений.")

    print("[START] Старт создания видео:")
    video_path = inator.create_video(fps=1, output_video_path=video_path)
    print(f"[END] Видео создано: {video_path}")


def process_and_create_gif(inator, data_path, gif_path, use_nms=False):
    """
    Обрабатывает изображения и создает GIF.
    :param inator: Экземпляр класса InatorBase или InatorNMS.
    :param data_path: Директория для сохранения обработанных изображений.
    :param gif_path: Путь для сохранения GIF-файла.
    :param use_nms: Флаг использования NMS.
    """
    print("[START] Старт обработки изображений:")
    if use_nms:
        processed_images = inator.process_images(data_path, iou_threshold=0.4, conf_threshold=0.3)
    else:
        processed_images = inator.process_images(data_path)
    print("[END] Конец обработки изображений.")

    print("[START] Старт создания GIF:")
    gif_creator = InatorGif(processed_images)
    gif_path = gif_creator.create_gif(output_gif_path=gif_path, duration=0.5)
    print(f"[END] GIF создан: {gif_path}")


def main():
    model_name = os.getenv("model_name")
    dataset_dir = os.getenv("dataset_dir")
    dataset_name = re.split(r'[\\/]', dataset_dir)[-1]
    
    # Определение путей
    inference_with_nms_dir = f"inference/{model_name}/with_nms/{dataset_name}"
    inference_without_nms_dir = f"inference/{model_name}/without_nms/{dataset_name}"
    
    # Новая структура для видео и GIF
    output_base_dir = f"output_video/{dataset_name}"
    output_video_dir = f"{output_base_dir}/video"
    output_gif_dir = f"{output_base_dir}/gif"
    
    # Удаление старых данных
    for path in [inference_with_nms_dir, inference_without_nms_dir, output_base_dir]:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
                print(f"Папка {path} удалена")
            else:
                os.remove(path)
                print(f"Файл {path} удален")
    
    # Создание необходимых директорий
    os.makedirs(inference_with_nms_dir, exist_ok=True)
    os.makedirs(inference_without_nms_dir, exist_ok=True)
    os.makedirs(output_video_dir, exist_ok=True)
    os.makedirs(output_gif_dir, exist_ok=True)
    
    test_images_path = f"dataset/{dataset_name}/test/images"
    model_path = f"models/{model_name}/weights/best.pt"
    
    # Обработка без NMS
    print("=" * 150)
    print(f"Запуск создания видео и GIF для модели {model_name} без NMS")
    print("=" * 150)
    inator = InatorBase(model_path, test_images_path)
    process_and_create_video(
        inator,
        inference_without_nms_dir,
        f"{output_video_dir}/{dataset_name}_without_nms.mp4"
    )
    process_and_create_gif(
        inator,
        inference_without_nms_dir,
        f"{output_gif_dir}/{dataset_name}_without_nms.gif"
    )
    
    # Обработка с NMS
    print("=" * 150)
    print(f"Запуск создания видео и GIF для модели {model_name} с NMS")
    print("=" * 150)
    inator_nms = InatorNMS(model_path, test_images_path)
    process_and_create_video(
        inator_nms,
        inference_with_nms_dir,
        f"{output_video_dir}/{dataset_name}_with_nms.mp4",
        use_nms=True
    )
    process_and_create_gif(
        inator_nms,
        inference_with_nms_dir,
        f"{output_gif_dir}/{dataset_name}_with_nms.gif",
        use_nms=True
    )

if __name__ == "__main__":
    main()
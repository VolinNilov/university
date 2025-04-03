import cv2
import torch
import os
import glob
from ultralytics import YOLO
from typing import List
import shutil

class Inator:
    def __init__(self, model_path: str, test_images_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path).to(self.device)
        self.test_images_path = test_images_path
        self.processed_images = []

    def process_images(self, processed_dir: str, iou_threshold: float = 0.5, conf_threshold: float = 0.3) -> List[str]:
        """
        Обрабатывает изображения из test_images_path с использованием модели YOLO.
        
        Возвращает список путей к обработанным изображениям.
        Добавлены параметры для NMS:
        - iou_threshold: порог IOU для фильтрации боксов.
        - conf_threshold: минимальный порог уверенности для боксов.
        """
        image_paths = glob.glob(os.path.join(self.test_images_path, "*.jpg"))
        output_dir = processed_dir
        os.makedirs(output_dir, exist_ok=True)
        
        processed_image_paths = []
        
        for img_path in image_paths:
            image = cv2.imread(img_path)
            results = self.model(image)

            if results and len(results) > 0:
                # Применяем NMS к результатам
                boxes = results[0].boxes.xyxy.cpu().numpy()  # Координаты боксов
                scores = results[0].boxes.conf.cpu().numpy()  # Уверенность
                class_ids = results[0].boxes.cls.cpu().numpy()  # ID классов

                # Фильтрация боксов с помощью NMS
                indices = cv2.dnn.NMSBoxes(
                    bboxes=boxes.tolist(), 
                    scores=scores.tolist(), 
                    score_threshold=conf_threshold, 
                    nms_threshold=iou_threshold
                )

                # Оставляем только боксы, прошедшие NMS
                filtered_boxes = [boxes[i] for i in indices.flatten()]
                filtered_scores = [scores[i] for i in indices.flatten()]
                filtered_class_ids = [class_ids[i] for i in indices.flatten()]

                # Рисуем отфильтрованные боксы на изображении
                for box, score, cls_id in zip(filtered_boxes, filtered_scores, filtered_class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{self.model.names[int(cls_id)]} {score:.2f}"
                    color = (0, 255, 0)  # Цвет рамки
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            output_path = os.path.join(output_dir, os.path.basename(img_path))
            cv2.imwrite(output_path, image)
            processed_image_paths.append(output_path)
        
        self.processed_images = processed_image_paths
        return processed_image_paths

    def create_video(self, fps: int, output_video_path: str) -> str:
        """
        Создаёт видео из обработанных изображений и сохраняет его в output_video_path.
        Возвращает путь к созданному видеофайлу.
        """
        if not self.processed_images:
            raise ValueError("Нет обработанных изображений. Сначала вызовите process_images().")
        
        first_image = cv2.imread(self.processed_images[0])
        height, width, _ = first_image.shape
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        for img_path in self.processed_images:
            frame = cv2.imread(img_path)
            video_writer.write(frame)
        
        video_writer.release()
        return output_video_path

def main():
    data_path = "inference"
    file_path = "output_video.mp4"

    if os.path.exists(data_path) and os.path.isdir(data_path):
        shutil.rmtree(data_path)
        print(f"Папка {data_path} удалена")
    
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Файл {file_path} удалён")

    infer = Inator(
        model_path="models/spiders_label_studio_yolov8n/weights/best.pt",
        test_images_path="dataset/spider_dataset/test/images"
    )
    
    processed_images = infer.process_images(
        processed_dir=data_path,
        iou_threshold=0.5,  # Порог IOU для NMS
        conf_threshold=0.5  # Минимальная уверенность для боксов
    )

    video_path = infer.create_video(
        fps=1,
        output_video_path=file_path
    )

    print(f"\nВидео создано: {video_path}")

if __name__ == "__main__":
    main()

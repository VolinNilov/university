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

    def process_images(self, processed_dir: str) -> List[str]:
        """
        Обрабатывает изображения из test_images_path с использованием модели YOLO.
        
        Возвращает список путей к обработанным изображениям.
        """
        image_paths = glob.glob(os.path.join(self.test_images_path, "*.jpg"))
        output_dir = processed_dir
        os.makedirs(output_dir, exist_ok=True)
        
        processed_image_paths = []
        
        for img_path in image_paths:
            image = cv2.imread(img_path)
            results = self.model(image)
            
            if results and len(results) > 0:
                image_with_boxes = results[0].plot()
            else:
                image_with_boxes = image
            
            output_path = os.path.join(output_dir, os.path.basename(img_path))
            cv2.imwrite(output_path, image_with_boxes)
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
        processed_dir = data_path
    )

    video_path = infer.create_video(
        fps=1,
        output_video_path=file_path
    )

    print(f"\nВидео создано: {video_path}")

if __name__ == "__main__":
    main()
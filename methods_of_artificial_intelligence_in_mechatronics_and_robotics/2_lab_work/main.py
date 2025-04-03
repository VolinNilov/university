from ultralytics import YOLO
import torch
import os
from collections import defaultdict
import matplotlib.pyplot as plt

torch.backends.openmp.enabled = False

dataset_dir = "D:/Projects/university/methods_of_artificial_intelligence_in_mechatronics_and_robotics/2_lab_work/dataset/spiders_label_studio"
model_name = "spiders_label_studio_yolov8n"

def analyze_dataset(labels_dir):
    class_counts = defaultdict(int)
    for label_file in os.listdir(labels_dir):
        label_path = os.path.join(labels_dir, label_file)
        if os.path.isfile(label_path):
            with open(label_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
    return class_counts

labels_dir = os.path.join(dataset_dir, "train", "labels")
class_counts = analyze_dataset(labels_dir)

print("Количество аннотаций по классам:")
for class_id, count in sorted(class_counts.items()):
    print(f"Класс {class_id}: {count} аннотаций")

classes = list(class_counts.keys())
counts = list(class_counts.values())

plt.figure(figsize=(10, 6))
plt.bar(classes, counts, color='skyblue')
plt.xlabel('Классы')
plt.ylabel('Количество аннотаций')
plt.title('Распределение аннотаций по классам')
plt.xticks(classes)
plt.grid(axis='y', linestyle='--', alpha=0.7)

output_plot_path = os.path.join(f"models/{model_name}", "class_distribution.png")
plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
print(f"График сохранен в файл: {output_plot_path}")

plt.show()

model = YOLO("yolov8n.pt")

model.train(
    data=os.path.join(dataset_dir, "data.yaml"),
    epochs=1000,
    batch=-1,
    imgsz=640,
    device="cuda",
    optimizer="AdamW",
    patience=150,
    workers=0,
    mosaic=1.0,
    save=True,
    project="models",
    name=model_name
)
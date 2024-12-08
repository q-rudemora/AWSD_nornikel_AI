import os
import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from pathlib import Path
import shutil

def prepare_dataset():
    """Подготовка датасета в формате YOLO для сегментации"""
    dataset_dir = Path("datasets")
    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "masks"
    
    # Создаем директории для YOLO
    yolo_dir = Path("yolo_dataset")
    if yolo_dir.exists():
        shutil.rmtree(yolo_dir)
    
    for split in ['train', 'val']:
        (yolo_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (yolo_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Получаем список изображений
    image_files = list(images_dir.glob("*.jpg"))
    np.random.seed(42)  # для воспроизводимости
    np.random.shuffle(image_files)
    
    # Разделяем на train и val (80/20)
    split_idx = int(len(image_files) * 0.8)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    def process_image(img_path, split):
        img_name = img_path.name
        mask_path = masks_dir / img_name.replace('.jpg', '.png')
        
        if not mask_path.exists():
            return
        
        # Копируем изображение
        dst_img = yolo_dir / split / 'images' / img_name
        shutil.copy2(img_path, dst_img)
        
        # Читаем маску
        mask = cv2.imread(str(mask_path))
        if mask is None:
            return
            
        # Конвертируем маску в бинарный формат
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = (mask > 0).astype(np.uint8)
        
        # Находим контуры загрязнений
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Получаем размеры изображения
        img_height, img_width = mask.shape
        label_path = yolo_dir / split / 'labels' / (img_name.replace('.jpg', '.txt'))
        
        with open(label_path, 'w') as f:
            for contour in contours:
                # Преобразуем контур в полигон с нормализованными координатами
                polygon = contour.reshape(-1, 2)
                # Нормализуем координаты
                polygon = polygon.astype(np.float32)
                polygon[:, 0] = polygon[:, 0] / img_width
                polygon[:, 1] = polygon[:, 1] / img_height
                
                # Записываем в формате YOLO: class x1 y1 x2 y2 ...
                points = polygon.reshape(-1).tolist()
                if len(points) >= 6:  # Минимум 3 точки для полигона
                    f.write("0 " + " ".join(map(str, points)) + "\n")
    
    print(f"Подготовка train набора ({len(train_files)} изображений)...")
    for img_path in train_files:
        process_image(img_path, 'train')
    
    print(f"Подготовка validation набора ({len(val_files)} изображений)...")
    for img_path in val_files:
        process_image(img_path, 'val')
    
    # Создаем yaml файл для обучения
    dataset_yaml = {
        'path': str(yolo_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'names': {0: 'pollution'},
        'nc': 1
    }
    
    yaml_path = yolo_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f)
    
    print(f"Датасет подготовлен и сохранен в {yolo_dir}")
    print(f"Конфигурация сохранена в {yaml_path}")
    
    return yaml_path

def train_model():
    """Обучение модели YOLOv8 для сегментации"""
    print("Подготовка датасета...")
    dataset_yaml = prepare_dataset()
    
    print("Загрузка предобученной модели...")
    model = YOLO('yolov8n-seg.pt')
    
    print("Начало обучения...")
    results = model.train(
        data=str(dataset_yaml),
        epochs=25,
        imgsz=640,
        batch=2,
        device='cpu',
        patience=10,
        save=True,
        project='runs/train',
        name='pollution_detector',
        verbose=True
    )
    
    # Копируем лучшую модель
    best_model = Path('runs/train/pollution_detector/weights/best.pt')
    if best_model.exists():
        shutil.copy2(best_model, 'baseline.pt')
        print(f"Лучшая модель сохранена как baseline.pt")
    else:
        print("Предупреждение: файл лучшей модели не найден")

if __name__ == "__main__":
    train_model()

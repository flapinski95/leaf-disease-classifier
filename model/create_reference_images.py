import os
import shutil
import random

SOURCE_DIR = '../dataset/plant_dataset/valid'
TARGET_DIR = './reference_images'
IMAGES_PER_CLASS = 5 

os.makedirs(TARGET_DIR, exist_ok=True)

for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    target_class_path = os.path.join(TARGET_DIR, class_name)
    os.makedirs(target_class_path, exist_ok=True)

    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    selected = random.sample(images, min(IMAGES_PER_CLASS, len(images)))

    for img in selected:
        shutil.copy2(os.path.join(class_path, img), os.path.join(target_class_path, img))

print(f"Wygenerowano folder '{TARGET_DIR}' z maks. {IMAGES_PER_CLASS} obrazami na klasę.")
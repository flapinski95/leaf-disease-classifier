import os
import shutil
import random

TRAIN_DIR = '../dataset/plant_dataset/train'
TEST_DIR = '../dataset/plant_dataset/test'
IMAGES_PER_CLASS = 5  

for class_name in os.listdir(TRAIN_DIR):
    train_class_path = os.path.join(TRAIN_DIR, class_name)
    test_class_path = os.path.join(TEST_DIR, class_name)

    if not os.path.isdir(train_class_path):
        continue

    os.makedirs(test_class_path, exist_ok=True)

    images = [f for f in os.listdir(train_class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    selected = random.sample(images, min(IMAGES_PER_CLASS, len(images)))

    for image in selected:
        src = os.path.join(train_class_path, image)
        dst = os.path.join(test_class_path, image)
        shutil.copy(src, dst)

print("✅ Wygenerowano test-set: po", IMAGES_PER_CLASS, "obrazów na klasę.")
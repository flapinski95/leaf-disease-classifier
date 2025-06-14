import os
import shutil

TRAIN_DIR = '../dataset/plant_dataset/train'
TEST_DIR = '../dataset/plant_dataset/test'

train_classes = set(os.listdir(TRAIN_DIR))
test_classes = set(os.listdir(TEST_DIR))

extra_classes = test_classes - train_classes

for class_name in extra_classes:
    path = os.path.join(TEST_DIR, class_name)
    print(f"🗑 Usuwam nadmiarowy folder: {class_name}")
    shutil.rmtree(path)

print("✅ Test set oczyszczony – pasuje do klas treningowych.")
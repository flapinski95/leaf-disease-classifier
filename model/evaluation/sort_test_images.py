import os
import shutil

TRAIN_DIR = '../dataset/plant_dataset/train'
TEST_DIR = '../dataset/plant_dataset/test'
DEST_DIR = TEST_DIR 

train_classes = os.listdir(TRAIN_DIR)

for class_name in train_classes:
    class_path = os.path.join(DEST_DIR, class_name)
    os.makedirs(class_path, exist_ok=True)

for filename in os.listdir(TEST_DIR):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    matched_class = None
    for class_name in train_classes:
        if class_name.lower().replace("___", "").replace("_", "").replace(" ", "") in filename.lower().replace("_", "").replace(" ", ""):
            matched_class = class_name
            break

    if matched_class:
        label_folder = os.path.join(DEST_DIR, matched_class)
    else:
        label_folder = os.path.join(DEST_DIR, 'unknown')
        os.makedirs(label_folder, exist_ok=True)

    src = os.path.join(TEST_DIR, filename)
    dst = os.path.join(label_folder, filename)
    shutil.move(src, dst)

print("✅ Posortowano zdjęcia i uzupełniono brakujące foldery.")
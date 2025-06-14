import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Argumenty CLI
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, choices=["mobilenet", "resnet", "efficientnet"])
args = parser.parse_args()
model_name = args.model

# Konfiguracja
DATASET_PATH = '../dataset/plant_dataset'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
INPUT_SHAPE = (*IMAGE_SIZE, 3)
EPOCHS_INITIAL = 30
EPOCHS_FINE_TUNE = 10
RESULTS_DIR = f"results/{model_name}"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Model bazowy
def get_base_model(name, input_shape):
    if name == "mobilenet":
        return MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    elif name == "resnet":
        return ResNet50(input_shape=input_shape, include_top=False, weights="imagenet")
    elif name == "efficientnet":
        return EfficientNetB0(input_shape=input_shape, include_top=False, weights="imagenet")

# Augmentacja
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    brightness_range=(0.8, 1.2),
    horizontal_flip=True
)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'train'),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'valid'),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Budowanie modelu
base_model = get_base_model(model_name, INPUT_SHAPE)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(f'{RESULTS_DIR}/best_model.h5', monitor='val_accuracy', save_best_only=True)
]

# Trening początkowy
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS_INITIAL,
    callbacks=callbacks
)

# Fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

fine_tune_history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS_FINE_TUNE,
    callbacks=callbacks
)

# Zapis modelu
model.save(f'{RESULTS_DIR}/model')
model.save(f'{RESULTS_DIR}/model.h5')

# Wykresy
def plot_and_save(history, key, title, filename):
    plt.figure()
    plt.plot(history.history[key], label=f'train_{key}')
    plt.plot(history.history[f'val_{key}'], label=f'val_{key}')
    plt.title(title)
    plt.xlabel('Epoka')
    plt.ylabel(key.capitalize())
    plt.legend()
    plt.savefig(f'{RESULTS_DIR}/{filename}')
    plt.close()

plot_and_save(history, 'accuracy', f'{model_name} - Dokładność', 'accuracy.png')
plot_and_save(history, 'loss', f'{model_name} - Strata', 'loss.png')

# Ewaluacja
y_true = valid_generator.classes
y_pred_probs = model.predict(valid_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
labels = list(valid_generator.class_indices.keys())

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(14, 10))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.title(f"{model_name} - Macierz pomyłek")
plt.xlabel("Predykcja")
plt.ylabel("Rzeczywista klasa")
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/confusion_matrix.png')

with open(f'{RESULTS_DIR}/classification_report.txt', 'w') as f:
    f.write(classification_report(y_true, y_pred, target_names=labels))
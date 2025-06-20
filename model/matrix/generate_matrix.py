import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

# --- Ścieżki ---
MODEL_PATH = '../saved_models/best_model.keras'
TEST_DIR = '../../dataset/plant_dataset/test'
OUT_DIR = '../../model/matrix/output'
os.makedirs(OUT_DIR, exist_ok=True)

# --- Parametry ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# --- Generator danych testowych ---
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

model = load_model(MODEL_PATH)

y_true = test_generator.classes
filenames = test_generator.filenames
class_names = list(test_generator.class_indices.keys())

preds = model.predict(test_generator)
y_pred = np.argmax(preds, axis=1)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(16, 16))
disp.plot(ax=ax, xticks_rotation=90, cmap="Blues", colorbar=True)
plt.title("Macierz Pomyłek")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.pdf"))

pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(os.path.join(OUT_DIR, "confusion_matrix.csv"))

report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
pd.DataFrame(report).transpose().to_csv(os.path.join(OUT_DIR, "classification_report.csv"))
print(classification_report(y_true, y_pred, target_names=class_names))

errors = []
for i in range(len(y_true)):
    if y_pred[i] != y_true[i]:
        errors.append({
            'file': filenames[i],
            'true': class_names[y_true[i]],
            'predicted': class_names[y_pred[i]]
        })

pd.DataFrame(errors).to_csv(os.path.join(OUT_DIR, "misclassified.csv"), index=False)

from collections import Counter
confused_pairs = [ (class_names[y_true[i]], class_names[y_pred[i]]) for i in range(len(y_true)) if y_true[i] != y_pred[i] ]
common_errors = Counter(confused_pairs).most_common(5)

print("\n🔝 TOP 5 Pomyłek:")
for (true, pred), count in common_errors:
    print(f"{true} → {pred}: {count} razy")
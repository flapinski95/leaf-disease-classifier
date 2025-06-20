import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
TEST_DIR = '../dataset/plant_dataset/test'  

mobilenet = load_model('../model/saved_models/best_model_mobilenet.keras')
resnet = load_model('../model/saved_models/resnet.h5')
efficientnet = load_model('../model/saved_models/efficientnet.h5')

datagen = ImageDataGenerator(rescale=1./255)
test_gen = datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

y_true = test_gen.classes
class_names = list(test_gen.class_indices.keys())

pred_mob = mobilenet.predict(test_gen, verbose=1)
pred_res = resnet.predict(test_gen, verbose=1)
pred_eff = efficientnet.predict(test_gen, verbose=1)

combinations = {
    "MobileNet": pred_mob,
    "MobileNet + ResNet": 0.7 * pred_mob + 0.3 * pred_res,
    "MobileNet + ResNet + EfficientNet": (0.5 * pred_mob + 0.25 * pred_res + 0.25 * pred_eff),
}

for name, pred in combinations.items():
    y_pred = np.argmax(pred, axis=1)
    print(f"\n🔍 {name}")
    print("✅ Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print(classification_report(y_true, y_pred, target_names=class_names))
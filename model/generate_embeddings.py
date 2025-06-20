import os
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle

REFERENCE_DIR = './reference_images'
OUTPUT_FILE = './embeddings/embedding_db.pkl'

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

model = load_model('./saved_models/best_model.keras')
embedding_model = Model(inputs=model.input, outputs=model.layers[-2].output)

embedding_db = []

for class_name in os.listdir(REFERENCE_DIR):
    class_dir = os.path.join(REFERENCE_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue
    for filename in os.listdir(class_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        path = os.path.join(class_dir, filename)
        img = load_img(path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        embedding = embedding_model.predict(img_array, verbose=0)[0]
        embedding_db.append({
            'label': class_name,
            'embedding': embedding
        })

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(embedding_db, f)

print(f"✅ Zapisano {len(embedding_db)} embeddingów do {OUTPUT_FILE}")
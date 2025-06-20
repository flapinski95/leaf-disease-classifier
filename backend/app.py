from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import pickle
import joblib

from embedding_model import load_embedding_model
from utils.verify import verify_prediction
from utils.fuzzy_confidence2 import evaluate_fuzzy_trust

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

model = load_model('../model/saved_models/best_model_mobilenet.keras')
embedding_model = load_embedding_model()
binary_health_model = joblib.load('../model/binary/binary_health_classifier.joblib')

with open('../model/embeddings/embedding_db.pkl', 'rb') as f:
    embedding_db = pickle.load(f)
ref_embeddings = np.array([e['embedding'] for e in embedding_db])
ref_labels = [e['label'] for e in embedding_db]

class_names = [ 'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus' ]

confusion_pairs = {
    'Apple___Apple_scab': ['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Pepper,_bell___Bacterial_spot'],
    'Corn_(maize)___Northern_Leaf_Blight': ['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot'],
    'Potato___Late_blight': ['Potato___Early_blight'],
    'Strawberry___Leaf_scorch': ['Potato___Early_blight'],
    'Tomato___Late_blight': ['Tomato___Early_blight']
}

def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Brak pliku'}), 400

    file = request.files['image']
    img_bytes = file.read()
    input_tensor = preprocess(img_bytes)

    prediction = model.predict(input_tensor)[0]
    top_indices = prediction.argsort()[-3:][::-1]
    results = [{'label': class_names[i], 'confidence': float(prediction[i])} for i in top_indices]
    top_label = results[0]['label']
    top_confidence = results[0]['confidence']
    top_labels = [r['label'] for r in results]

    query_embedding = embedding_model.predict(input_tensor)
    top_sim_labels, avg_sim = verify_prediction(query_embedding, ref_embeddings, ref_labels, top_n=5)

    trust_score = evaluate_fuzzy_trust(top_confidence, avg_sim)

    prob_healthy = binary_health_model.predict_proba(query_embedding)[0][1]

    warnings = []

    if trust_score < 50:
        warnings.append("System fuzzy sugeruje niskie zaufanie do tej diagnozy.")
    elif trust_score < 75:
        warnings.append("ℹSystem fuzzy ocenia tę predykcję jako średnio wiarygodną.")
    else:
        warnings.append("System fuzzy potwierdza wysokie zaufanie do predykcji.")

    if top_label not in top_sim_labels and avg_sim > 0.8:
        warnings.append(f"Predykcja '{top_label}' nie pasuje do podobnych wizualnie klas: {top_sim_labels}. Możliwa pomyłka.")

    if top_confidence < 0.6:
        warnings.append("Model jest niepewny swojej predykcji – wynik może być błędny.")

    if top_label in confusion_pairs:
        mistaken_with = confusion_pairs[top_label]
        if any(label in top_labels for label in mistaken_with):
            warnings.append(f"'{top_label}' często bywa mylona z: {mistaken_with}.")

    if "healthy" in top_label and prob_healthy < 0.6:
        warnings.append(f"Choć główny model rozpoznał '{top_label}', binarny klasyfikator sugeruje, że to może być choroba (pewność healthy = {prob_healthy:.2f}).")

    return jsonify({
        'predictions': results,
        'warning': "\n".join(warnings) if warnings else None,
        'trust_score': round(trust_score, 2),
        'similar_classes': top_sim_labels,
        'prob_healthy_binary': round(prob_healthy, 2)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
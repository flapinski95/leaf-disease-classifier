from tensorflow.keras.models import load_model, Model

def load_embedding_model():
    full_model = load_model('../model/saved_models/best_model_mobilenet.keras')
    return Model(inputs=full_model.input, outputs=full_model.layers[-2].output)
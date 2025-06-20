import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def verify_prediction(query_embedding, reference_embeddings, reference_labels, top_n=5):
    similarities = cosine_similarity(query_embedding, reference_embeddings)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    top_labels = [reference_labels[i] for i in top_indices]
    avg_similarity = similarities[top_indices].mean()
    return top_labels, avg_similarity
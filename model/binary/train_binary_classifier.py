import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

with open('../embeddings/embedding_db.pkl', 'rb') as f:
    embedding_db = pickle.load(f)

X = np.array([e['embedding'] for e in embedding_db])
y = np.array([1 if 'healthy' in e['label'] else 0 for e in embedding_db])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['diseased', 'healthy']))

joblib.dump(clf, 'binary_health_classifier.joblib')
print("Zapisano: binary_health_classifier.joblib")
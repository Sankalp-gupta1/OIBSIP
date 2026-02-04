import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# BASE PATH (IMPORTANT)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "data", "iris.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "iris_model.pkl")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

# Drop ID column if exists
if "Id" in df.columns:
    df.drop("Id", axis=1, inplace=True)

# Features & Target
X = df.drop("Species", axis=1)
y = df["Species"]

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# PIPELINE (ADVANCED MODEL)
# =========================
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=10, gamma="scale"))
])

# Train
pipeline.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
y_pred = pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks(range(3), le.classes_)
plt.yticks(range(3), le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =========================
# SAVE MODEL (GUARANTEED)
# =========================
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

joblib.dump({
    "model": pipeline,
    "label_encoder": le
}, MODEL_PATH)

print("\n✅ Model saved successfully at:")
print(MODEL_PATH)

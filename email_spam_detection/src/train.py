import pandas as pd
import re
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("data/spam.csv", encoding="latin1")
df = df[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# =========================
# 2. Text Cleaning
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['text'] = df['text'].apply(clean_text)

# =========================
# 3. Train Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    df['text'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# =========================
# 4. TF-IDF Vectorizer
# =========================
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    max_df=0.95,
    min_df=2,
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =========================
# 5. Model Training
# =========================
model = MultinomialNB(alpha=0.1)
model.fit(X_train_vec, y_train)

# =========================
# 6. Predictions
# =========================
y_pred = model.predict(X_test_vec)
y_prob = model.predict_proba(X_test_vec)[:, 1]

# =========================
# 7. Metrics
# =========================
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n✅ Accuracy:", acc)
print("\n📌 Classification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# 8. Confusion Matrix Plot
# =========================
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0,1], ["Ham", "Spam"])
plt.yticks([0,1], ["Ham", "Spam"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i][j], ha="center", va="center")

plt.show()

# =========================
# 9. ROC Curve
# =========================
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# =========================
# 10. Save Model
# =========================
joblib.dump(model, "model/spam_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("\n💾 Model & Vectorizer Saved Successfully!")

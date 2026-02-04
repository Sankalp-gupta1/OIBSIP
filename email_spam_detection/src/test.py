import joblib

model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

msg = ["Congratulations! You won a free iPhone"]
vec = vectorizer.transform(msg)
res = model.predict(vec)

print("Spam" if res[0] == 1 else "Not Spam")

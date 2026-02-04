import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "car_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "car_price_model.pkl")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

# Drop car name
if "Car_Name" in df.columns:
    df.drop("Car_Name", axis=1, inplace=True)

# Feature engineering
df["Car_Age"] = 2026 - df["Year"]
df.drop("Year", axis=1, inplace=True)

# Rename for consistency
df.rename(columns={
    "Kms_Driven": "Driven_kms",
    "Seller_Type": "Selling_type"
}, inplace=True)

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MODEL
# =========================
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test)

print("R2 Score :", r2_score(y_test, y_pred))
print("MAE      :", mean_absolute_error(y_test, y_pred))
print("RMSE     :", np.sqrt(mean_squared_error(y_test, y_pred)))

# =========================
# SAVE MODEL + FEATURES
# =========================
bundle = {
    "model": model,
    "features": X.columns.tolist()
}

pickle.dump(bundle, open(MODEL_PATH, "wb"))
print("✅ Model saved at:", MODEL_PATH)

# =========================
# GRAPH
# =========================
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.show()

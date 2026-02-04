import streamlit as st
import joblib
import numpy as np

# Load model
bundle = joblib.load("D:\iris_flower_detection\model\iris_model.pkl")
model = bundle["model"]
le = bundle["label_encoder"]

st.title("🌸 Iris Flower Detection App")

st.write("Enter flower measurements to predict species")

sl = st.number_input("Sepal Length (cm)", 4.0, 8.0, 5.1)
sw = st.number_input("Sepal Width (cm)", 2.0, 4.5, 3.5)
pl = st.number_input("Petal Length (cm)", 1.0, 7.0, 1.4)
pw = st.number_input("Petal Width (cm)", 0.1, 2.5, 0.2)

if st.button("Predict"):
    data = np.array([[sl, sw, pl, pw]])
    pred = model.predict(data)
    flower = le.inverse_transform(pred)

    st.success(f"🌼 Predicted Species: **{flower[0]}**")

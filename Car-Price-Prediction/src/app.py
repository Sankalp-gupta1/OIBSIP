import streamlit as st
import pandas as pd
import pickle
import os

# ===============================
# LOAD MODEL
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "car_price_model.pkl")

bundle = pickle.load(open(MODEL_PATH, "rb"))
model = bundle["model"]
feature_names = bundle["features"]

st.set_page_config(page_title="Car Price Prediction", layout="centered")

st.title("🚗 Car Price Prediction App")
st.write("Fill the details to predict the selling price of a car")

# ===============================
# USER INPUTS
# ===============================
year = st.number_input("Year of Purchase", 2000, 2026, 2015)
present_price = st.number_input("Present Price (in Lakhs)", value=5.0)
kms_driven = st.number_input("Kilometers Driven", value=50000)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Owner", [0, 1, 3])

car_age = 2026 - year

# ===============================
# INPUT DATAFRAME
# ===============================
input_dict = {
    "Present_Price": present_price,
    "Driven_kms": kms_driven,
    "Owner": owner,
    "Car_Age": car_age,
    "Fuel_Type_Diesel": 1 if fuel_type == "Diesel" else 0,
    "Fuel_Type_Petrol": 1 if fuel_type == "Petrol" else 0,
    "Selling_type_Individual": 1 if seller_type == "Individual" else 0,
    "Transmission_Manual": 1 if transmission == "Manual" else 0,
}

input_df = pd.DataFrame([input_dict])

# FIX FEATURE ORDER
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# ===============================
# PREDICTION
# ===============================
if st.button("Predict Price 💰"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Selling Price: ₹ {prediction:.2f} Lakhs")

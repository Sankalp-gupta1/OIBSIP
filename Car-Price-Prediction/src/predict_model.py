import streamlit as st
import pandas as pd
import pickle
import os

# ===============================
# LOAD MODEL
# ===============================
MODEL_PATH = os.path.join("model", "car_price_model.pkl")

model = pickle.load(open(MODEL_PATH, "rb"))

st.set_page_config(
    page_title="Car Price Prediction",
    layout="centered"
)

st.title("🚗 Car Price Prediction App")
st.write("Fill the details to predict the selling price of a car")

# ===============================
# USER INPUTS
# ===============================
year = st.number_input("Year of Purchase", min_value=2000, max_value=2026, value=2015)
present_price = st.number_input("Present Price (in Lakhs)", value=5.0)
kms_driven = st.number_input("Kilometers Driven", value=50000)

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])
owner = st.selectbox("Owner", [0, 1, 3])

# ===============================
# INPUT DATA PREPARATION
# ===============================
input_dict = {
    "Year": year,
    "Present_Price": present_price,
    "Kms_Driven": kms_driven,
    "Owner": owner,
    "Fuel_Type_Diesel": 1 if fuel_type == "Diesel" else 0,
    "Fuel_Type_Petrol": 1 if fuel_type == "Petrol" else 0,
    "Seller_Type_Individual": 1 if seller_type == "Individual" else 0,
    "Transmission_Manual": 1 if transmission == "Manual" else 0
}

input_df = pd.DataFrame([input_dict])

# ===============================
# PREDICTION
# ===============================
if st.button("Predict Price 💰"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Selling Price: ₹ {prediction:.2f} Lakhs")

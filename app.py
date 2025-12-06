import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and scaler
rf = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details below to predict the likelihood of heart disease.")

# Input fields for each feature
age = st.number_input("Age", 1, 120, 45)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 240)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise-induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope of Peak Exercise ST", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (0–3)", [0, 1, 2, 3])

# Convert input to DataFrame
input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]],
                          columns=["age", "sex", "cp", "trestbps", "chol", "fbs",
                                   "restecg", "thalach", "exang", "oldpeak",
                                   "slope", "ca", "thal"])

# Scale data using the same scaler as training
scaled_data = scaler.transform(input_data)

# Prediction
if st.button("Predict"):
    prediction = rf.predict(scaled_data)[0]
    probability = rf.predict_proba(scaled_data)[0][1]

    if prediction == 1:
        st.error(f"💔 High chance of heart disease. (Probability: {probability:.2f})")
    else:
        st.success(f"❤️ Low chance of heart disease. (Probability: {probability:.2f})")

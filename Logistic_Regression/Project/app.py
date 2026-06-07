import streamlit as st
import pickle
import numpy as np
from pathlib import Path

# Current app.py folder ka path
BASE_DIR = Path(__file__).parent

# Load model and scaler from same folder
model = pickle.load(open(BASE_DIR / "diabetes_model.pkl", "rb"))
scaler = pickle.load(open(BASE_DIR / "scaler.pkl", "rb"))

# Title
st.title("Diabetes Prediction System")

st.write("Enter Patient Details")

# User Inputs
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=1)

# Predict Button
if st.button("Predict"):

    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    # Apply scaling
    input_data = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_data)

    # Probability
    probability = model.predict_proba(input_data)

    # Output
    if prediction[0] == 1:
        st.error("Person is Diabetic")
    else:
        st.success("Person is Not Diabetic")

    st.write("Prediction Probability:")
    st.write(probability)
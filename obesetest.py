
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Load the trained assets with (1) suffix
@st.cache_resource
def load_assets():
    model = pickle.load(open('random_forest_classifier (1).pkl', 'rb'))
    scaler = pickle.load(open('scaler (1).pkl', 'rb'))
    le = pickle.load(open('label_encoder (1).pkl', 'rb'))
    return model, scaler, le

try:
    model, scaler, le = load_assets()
except FileNotFoundError:
    st.error("Required .pkl files with '(1)' suffix not found in the repository.")
    st.stop()

# 2. Exact column order from training
train_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'BMI',
              'Gender_Male', 'family_history_with_overweight_yes', 'FAVC_yes', 
              'CAEC_Frequently', 'CAEC_Sometimes', 'CAEC_no', 'SMOKE_yes', 'SCC_yes', 
              'CALC_Frequently', 'CALC_Sometimes', 'CALC_no', 'MTRANS_Bike', 
              'MTRANS_Motorbike', 'MTRANS_Public_Transportation', 'MTRANS_Walking']

st.title("Obesity Level Predictor")

# 3. User Inputs
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age", 1.0, 100.0, 25.0)
    height = st.number_input("Height (m)", 1.0, 2.5, 1.70)
    weight = st.number_input("Weight (kg)", 30.0, 250.0, 70.0)
    family_history = st.selectbox("Family history with overweight?", ["yes", "no"])
    favc = st.selectbox("Frequent high caloric food?", ["yes", "no"])
    fcvc = st.slider("Vegetable consumption", 1.0, 3.0, 2.0)
    ncp = st.slider("Number of meals", 1.0, 4.0, 3.0)

with col2:
    caec = st.selectbox("Food between meals", ["Sometimes", "Frequently", "Always", "no"])
    smoke = st.selectbox("Smoke?", ["yes", "no"])
    ch2o = st.slider("Water intake (L)", 1.0, 3.0, 2.0)
    scc = st.selectbox("Monitor calories?", ["yes", "no"])
    faf = st.slider("Physical activity", 0.0, 3.0, 1.0)
    tue = st.slider("Tech usage time", 0.0, 2.0, 1.0)
    calc = st.selectbox("Alcohol consumption", ["Sometimes", "Frequently", "Always", "no"])
    mtrans = st.selectbox("Transportation", ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"])

# 4. Prediction Logic
if st.button("Predict Weight Category", key="predict_btn"):
    # Calculate BMI
    bmi_val = weight / (height ** 2)

    # Map inputs to match the 24 encoded features
    input_dict = {
        'Age': age, 'Height': height, 'Weight': weight, 'FCVC': fcvc, 'NCP': ncp,
        'CH2O': ch2o, 'FAF': faf, 'TUE': tue, 'BMI': bmi_val,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'family_history_with_overweight_yes': 1 if family_history == 'yes' else 0,
        'FAVC_yes': 1 if favc == 'yes' else 0,
        'CAEC_Frequently': 1 if caec == 'Frequently' else 0,
        'CAEC_Sometimes': 1 if caec == 'Sometimes' else 0,
        'CAEC_no': 1 if caec == 'no' else 0,
        'SMOKE_yes': 1 if smoke == 'yes' else 0,
        'SCC_yes': 1 if scc == 'yes' else 0,
        'CALC_Frequently': 1 if calc == 'Frequently' else 0,
        'CALC_Sometimes': 1 if calc == 'Sometimes' else 0,
        'CALC_no': 1 if calc == 'no' else 0,
        'MTRANS_Bike': 1 if mtrans == 'Bike' else 0,
        'MTRANS_Motorbike': 1 if mtrans == 'Motorbike' else 0,
        'MTRANS_Public_Transportation': 1 if mtrans == 'Public_Transportation' else 0,
        'MTRANS_Walking': 1 if mtrans == 'Walking' else 0
    }

    # Create DataFrame and ensure EXACT column order
    input_df = pd.DataFrame([input_dict])[train_cols]

    # Scale the data using the loaded scaler
    scaled_data = scaler.transform(input_df.values)
    
    # Predict using the loaded model
    prediction = model.predict(scaled_data)
    final_label = le.inverse_transform(prediction)

    # Display Results
    st.markdown("--- ")
    st.success(f"Predicted Category: **{final_label[0]}**")
    st.info(f"Calculated BMI: **{bmi_val:.2f}**")

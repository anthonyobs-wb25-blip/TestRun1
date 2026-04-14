import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("random_forest_classifier.pkl", "rb"))

st.title("Obesity Prediction System")

st.header("Enter Your Information")

# ---- INPUTS ----
age = st.number_input("Age", 1, 100)
gender = st.selectbox("Gender", ["Male", "Female"])
height = st.number_input("Height (m)", 1.0, 2.5)
weight = st.number_input("Weight (kg)", 20.0, 200.0)

# BMI auto calculation
bmi = weight / (height ** 2)
st.write(f"BMI: {bmi:.2f}")

family_history = st.selectbox("Family History with Overweight", ["yes", "no"])
favc = st.selectbox("Frequent High Calorie Food", ["yes", "no"])

fcvc = st.slider("Vegetable Intake (FCVC)", 1.0, 3.0)
ncp = st.slider("Meals per Day (NCP)", 1.0, 5.0)
ch2o = st.slider("Water Intake (CH2O)", 1.0, 3.0)
faf = st.slider("Physical Activity (FAF)", 0.0, 3.0)

scc = st.selectbox("Monitor Calories (SCC)", ["yes", "no"])

mtrans = st.selectbox("Transport Mode", [
    "Walking", "Bike", "Motorbike", "Public_Transportation", "Automobile"
])

# ---- ENCODING ----
def preprocess():
    gender_map = {"Male": 1, "Female": 0}
    yes_no = {"yes": 1, "no": 0}
    mtrans_map = {
        "Walking": 0,
        "Bike": 1,
        "Motorbike": 2,
        "Public_Transportation": 3,
        "Automobile": 4
    }

    return np.array([[ 
        age,
        gender_map[gender],
        height,
        weight,
        bmi,
        yes_no[family_history],
        yes_no[favc],
        fcvc,
        ncp,
        ch2o,
        faf,
        yes_no[scc],
        mtrans_map[mtrans]
    ]])

# ---- PREDICTION ----
if st.button("Predict"):
    data = preprocess()
    result = model.predict(data)

    st.success(f"Predicted Class: {result[0]}")
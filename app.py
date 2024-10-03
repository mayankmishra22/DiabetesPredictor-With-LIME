import streamlit as st
import pandas as pd
from app.model import DiabetesModel
from app.explainer import Explainer
import shap
import numpy as np

# Load and train the model
diabetes_model = DiabetesModel()
X_train, X_test, y_train, y_test = diabetes_model.load_data()
model = diabetes_model.train_model()

# Explanation handler
explainer = Explainer(model)

# Streamlit web interface
st.title("Diabetes Prediction and Explanation")
st.sidebar.header("Input Patient Data")

# Input fields
pregnancies = st.sidebar.number_input("Pregnancies", 0, 20)
glucose = st.sidebar.number_input("Glucose Level", 0, 200)
blood_pressure = st.sidebar.number_input("Blood Pressure", 0, 122)
skin_thickness = st.sidebar.number_input("Skin Thickness", 0, 99)
insulin = st.sidebar.number_input("Insulin Level", 0, 846)
bmi = st.sidebar.number_input("BMI", 0.0, 67.1)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 2.42)
age = st.sidebar.number_input("Age", 21, 100)

# Prediction button
if st.sidebar.button("Predict"):
    # Prepare the input data
    input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
    
    # Convert input_data into DataFrame
    input_data_df = pd.DataFrame([input_data], columns=diabetes_model.columns)
    
    # Prediction
    prediction = model.predict(input_data_df)  # Ensure model is called correctly
    st.subheader("Prediction")
    st.write(f"Diabetes: {'Yes' if prediction[0] == 1 else 'No'}")
    
    # LIME Explanation
    st.subheader("LIME Explanation")
    try:
        lime_explanation = explainer.lime_local_explanation(input_data_df, X_train)
        st.write(lime_explanation)
    except Exception as e:
        st.error(f"Error generating LIME explanation: {e}")

# Diabetes Prediction and Explanation

This project focuses on predicting the likelihood of diabetes using Random Forest Algorithm and providing explainability through LIME (Local Interpretable Model-Agnostic Explanations). The model is deployed via a Streamlit web application, making it accessible for healthcare professionals to use and interpret.

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Model Details](#model-details)
- [Explainability](#explainability)
- [Acknowledgements](#acknowledgements)

## Project Overview
The goal of this project is to predict the likelihood of diabetes based on patient health data and explain the predictions using interpretable machine learning techniques. This project integrates **Random Forest** for prediction and **LIME** for model interpretability, providing healthcare professionals with insights into the key factors that led to the model's decision.

## Technologies Used
- **Python 3.8+**
- **scikit-learn**: For model training and evaluation
- **LIME**: For explainable AI
- **Streamlit**: For creating the web interface
- **pandas**: For data manipulation
- **numpy**: For numerical operations

Model Details
The model used in this project is a Random Forest Classifier. It is trained on the Pima Indians Diabetes dataset, which contains medical records of 768 patients and includes features such as:

Pregnancies
Glucose
Blood Pressure
Skin Thickness
Insulin
BMI (Body Mass Index)
Diabetes Pedigree Function
Age
Explainability
To make the model's predictions interpretable, the project uses LIME. For each prediction, LIME identifies which features contributed most to the model's decision, making the process transparent and allowing healthcare professionals to understand why the model predicted diabetes or not for a specific patient.

Acknowledgements
This project uses the Pima Indians Diabetes Dataset from the UCI Machine Learning Repository, and various open-source libraries. Special thanks to the creators of LIME and Streamlit for their powerful tools.

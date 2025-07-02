# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 19:15:01 2021
@author: siddhardhan (Modified by Gokulakannan)
"""

import numpy as np
import pickle
import streamlit as st

# Load the saved model from the correct path
model_path = 'C:/Users/Deepika/OneDrive/Desktop/Diabetes_Prediction_Using_Machine_Learning/trained_model.sav'
loaded_model = pickle.load(open(model_path, 'rb'))

# Prediction function
def diabetes_prediction(input_data):
    try:
        # Convert input data to float and numpy array
        input_data_as_numpy_array = np.asarray(input_data, dtype=float).reshape(1, -1)

        # Predict
        prediction = loaded_model.predict(input_data_as_numpy_array)

        # Fix interpretation based on your dataset:
        # 0 = Diabetic, 1 = Not Diabetic
        if prediction[0] == 0:
            return '⚠️ The person is diabetic'
        else:
            return '✅ The person is not diabetic'
    except ValueError:
        return "❌ Please enter valid numeric values."

# Main function for Streamlit app
def main():
    st.title('🩺 Diabetes Prediction Web App')

    # Input fields
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')

    diagnosis = ''

    if st.button('Get Diabetes Test Result'):
        user_input = [
            Pregnancies, Glucose, BloodPressure, SkinThickness,
            Insulin, BMI, DiabetesPedigreeFunction, Age
        ]
        diagnosis = diabetes_prediction(user_input)

    st.success(diagnosis)

# Run app
if __name__ == '__main__':
    main()

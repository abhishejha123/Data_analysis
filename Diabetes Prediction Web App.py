#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 23:11:36 2023
@author: rishabhsharma
"""

import numpy as np
import pickle
import streamlit as st


# Loading the saved model
loaded_model = pickle.load(open('/Users/rishabhsharma/Desktop/Diabetes Prediction/trained_model.sav', 'rb'))


# Creating a function for prediction
def diabetes_prediction(input_data):
    # Changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'


def main():
    # Add a sidebar
    st.sidebar.title('About')
    st.sidebar.info('This is a Diabetes Prediction Web App.')

    # Add a header
    st.header('Diabetes Prediction Web App')

    # Getting the input data from the user
    with st.beta_container():
        st.subheader('Input Parameters')
        col1, col2 = st.beta_columns(2)
        with col1:
            pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, step=1)
            glucose = st.number_input('Glucose Level', min_value=0, max_value=300, step=1)
            blood_pressure = st.number_input('Blood Pressure value', min_value=0, max_value=200, step=1)
            skin_thickness = st.number_input('Skin Thickness value', min_value=0, max_value=100, step=1)
        with col2:
            insulin = st.number_input('Insulin Level', min_value=0, max_value=1000, step=1)
            bmi = st.number_input('BMI value', min_value=0.0, max_value=100.0, step=0.1)
            diabetes_pedigree = st.number_input('Diabetes Pedigree Function value', min_value=0.0, max_value=10.0, step=0.01)
            age = st.number_input('Age of the Person', min_value=0, max_value=150, step=1)

    # Code for prediction
    diagnosis = ''

    # Creating a button for prediction
    if st.button('Diabetes Test Result'):
        input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]
        diagnosis = diabetes_prediction(input_data)

    # Display the result
    st.subheader('Prediction')
    if diagnosis:
        st.success(diagnosis)


if __name__ == '__main__':
    main()

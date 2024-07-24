import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained Decision Tree model
model = pickle.load(open('decision_tree_model.pkl', 'rb'))

# Define a function to make predictions
def predict_cancer(data):
    scaler = StandardScaler()
    data = scaler.fit_transform([data])
    prediction = model.predict(data)
    return prediction[0]

# Streamlit app
st.title("Cancer Prediction App")

# Input fields for user
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 18, 100, 50)
smoking = st.selectbox('Smoking', ['Non-smoker', 'Smoker'])
fatigue = st.selectbox('Fatigue', ['No fatigue', 'Fatigue'])
allergy = st.selectbox('Allergy', ['No allergy', 'Allergy'])

# Convert text inputs to numerical values
gender_num = 1 if gender == 'Female' else 0
smoking_num = 1 if smoking == 'Smoker' else 0
fatigue_num = 1 if fatigue == 'Fatigue' else 0
allergy_num = 1 if allergy == 'Allergy' else 0

# Create a button for prediction
if st.button('Predict'):
    input_data = [gender_num, age, smoking_num, fatigue_num, allergy_num]
    result = predict_cancer(input_data)
    if result == 1:
        st.write("The model predicts that you have cancer.")
    else:
        st.write("The model predicts that you do not have cancer.")

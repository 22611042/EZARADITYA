import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained Decision Tree model
model = pickle.load(open('decision_tree_model.pkl', 'rb'))

# Define a function to make predictions
def predict_cancer(data):
    # Instantiate the scaler
    scaler = StandardScaler()
    
    # Reshape the data to match the scaler's expected input
    data = [data]
    
    # Fit and transform the data (note: in a real scenario, you should fit the scaler on the training data only)
    scaled_data = scaler.fit_transform(data)
    
    # Make a prediction using the loaded model
    prediction = model.predict(scaled_data)
    
    # Return the prediction result
    return prediction[0]

# Streamlit app
st.title("Cancer Prediction App")

# Input fields for user
gender = st.selectbox('Gender', [0, 1], help="0: Male, 1: Female")
age = st.slider('Age', 18, 100, 50, help="Select your age.")
smoking = st.selectbox('Smoking', [0, 1], help="0: Non-smoker, 1: Smoker")
fatigue = st.selectbox('Fatigue', [0, 1], help="0: No fatigue, 1: Fatigue")
allergy = st.selectbox('Allergy', [0, 1], help="0: No allergy, 1: Allergy")

# Create a button for prediction
if st.button('Predict'):
    # Collect input data
    input_data = [gender, age, smoking, fatigue, allergy]
    
    # Get prediction
    result = predict_cancer(input_data)
    
    # Display the result
    if result == 1:
        st.write("The model predicts that you have cancer.")
    else:
        st.write("The model predicts that you do not have cancer.")

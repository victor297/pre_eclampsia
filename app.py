import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the model
model = joblib.load('pre_eclampsia_model.joblib')

# Load the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['high risk', 'low risk', 'mid risk'])  # Ensure the classes are in the correct order

# Streamlit app
st.title('Pre-eclampsia Risk Prediction')
st.subheader('Model Training Metrics')
st.image('xgb_log_loss.png')
st.image('xgb_feature_importance.png')
# User input
age = st.number_input('Age', min_value=0)
systolic_bp = st.number_input('Systolic BP', min_value=0)
diastolic_bp = st.number_input('Diastolic BP', min_value=0)
bs = st.number_input('Blood Sugar Level', min_value=0.0)
body_temp = st.number_input('Body Temperature', min_value=0.0)
heart_rate = st.number_input('Heart Rate', min_value=0)

# Predict button
if st.button('Predict'):
    # Create input data for prediction
    input_data = pd.DataFrame([[age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate]],
                              columns=['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate'])
    
    # Predict the risk level
    prediction = model.predict(input_data)
    
    # Print the prediction for debugging
    st.write(f'Prediction: {prediction}')
    
    # Ensure the prediction is an integer array
    prediction = prediction.astype(int)
    
    # Inverse transform the prediction to get the risk level
    risk_level = label_encoder.inverse_transform(prediction)[0]  # Access the first element directly
    
    st.write(f'The predicted risk level is: {risk_level}')

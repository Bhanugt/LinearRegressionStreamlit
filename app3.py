import streamlit as st
import pandas as pd
import joblib  # To load saved scaler and model

# Load the saved scaler and model
scaler = joblib.load('scaler.pkl')  # Ensure this file exists
model = joblib.load('model.pkl')

st.title("IBD Prediction App (Linear Regression)")
st.write("Enter patient details to predict IBD outcome:")

# User input
feature1 = st.number_input("Enter Feature1", min_value=0.0)
feature2 = st.number_input("Enter Feature2", min_value=0.0)

if st.button("Predict"):
    # Create input DataFrame with correct column names
    input_df = pd.DataFrame([[feature1, feature2]], columns=['Feature1', 'Feature2'])
    
    # Ensure input matches training data format
    try:
        input_scaled = scaler.transform(input_df)
    except ValueError as e:
        st.error("Feature mismatch error! Please check input format.")
        st.stop()

    # Make prediction
    prediction = model.predict(input_scaled)

    st.success(f"Predicted IBD Outcome: {prediction[0]}")

import streamlit as st
import joblib
import numpy as np

st.title("Salary Estimation App")

st.divider()

# Correcting the input function name and parameter names
years_at_company = st.number_input("Enter years at company", min_value=0, max_value=20)
satisfaction_level = st.number_input("Satisfaction Level", min_value=0.0, max_value=1.0)
average_monthly_hours = st.number_input("Average Monthly Hours", min_value=120, max_value=400)

X = [years_at_company, satisfaction_level, average_monthly_hours]

# Load the pre-trained scaler and model
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

# Button for prediction
predict_button = st.button("Press for predicting the salary")

st.divider()

if predict_button:
    st.balloons()
    
    # Ensuring the input is in the correct format for the model
    X_array = scaler.transform([X])
    
    # Predicting the salary
    prediction = model.predict(X_array)[0]
    
    # Displaying the prediction
    st.write(f"Salary prediction is {prediction:.2f}")
else:
    st.write("Please enter the values and press the predict button.")

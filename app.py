import streamlit as st
import joblib
import numpy as np

st.title("Salary Estimation App")

st.divider()

years_at_company = st.number_input("Enter years at company", min_value = 0, max_value = 20)
satisfaction_level = st.number_input("Satisfactiob Level", min_value = 0)
average_monthly_hours = st.number_input("Average Monthly Hours", min_value = 120, max_value = 400)

X = [years_at_company, satisfaction_level, average_monthly_hours]

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

predict_button = st.button("Press for presdicting the salary")

st.divider()

if predict_button:

    st.balloons()

    X1 = np.array(X)

    X_array = scaler.transform([X1])

    prediction = model.predict(X_array)[0]

    st.write(f"Salary prediction is {prediction}")

else:
    st.write("Please enter the values and press to the predict button")







#X orders, 'Years_at_Company', 'Satisfaction_Level', 'Average_Monthly_Hours'
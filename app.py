import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Calories Burn Prediction", page_icon="🔥", layout="centered")

st.title(" Calories Burn Prediction")
st.markdown("Predict how many calories you burn during exercise using Machine Learning.")

with open('model.pkl', 'rb') as f:
    model, poly = pickle.load(f)

st.markdown("---")
st.subheader("Enter Your Exercise Details")

col1, col2, col3 = st.columns(3)

with col1:
    duration = st.number_input("Duration (minutes)", min_value=1, max_value=300, value=30)
with col2:
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=100)
with col3:
    body_temp = st.number_input("Body Temperature (°C)", min_value=36.0, max_value=42.0, value=38.5, step=0.1)

st.markdown("---")

if st.button("Predict Calories ", use_container_width=True):
    input_data = np.array([[duration, body_temp, heart_rate]])
    input_poly = poly.transform(input_data)
    prediction = model.predict(input_poly)[0]

    st.success(f"### Estimated Calories Burned: **{prediction:.1f} kcal**")

    if prediction < 100:
        st.info("Light activity 🚶")
    elif prediction < 300:
        st.info("Moderate activity 🏃")
    else:
        st.info("Intense activity 💪")

st.markdown("---")
st.caption("Model: Polynomial Regression (degree=2) | R² = 0.9622 | Best among 7 algorithms")

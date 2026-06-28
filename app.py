import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Calories Burn Prediction", page_icon="🔥", layout="centered")

st.title("🔥 Calories Burn Prediction")
st.markdown("Predict how many calories you burn during exercise using Machine Learning.")

@st.cache_resource
def train_model():
    dataset1 = pd.read_csv('calories.csv')
    dataset2 = pd.read_csv('exercise.csv')
    dataset = pd.merge(dataset2, dataset1, on='User_ID')

    le = LabelEncoder()
    dataset['Gender'] = le.fit_transform(dataset['Gender'])
    dataset.drop('User_ID', axis=1, inplace=True)

    X = dataset[['Duration', 'Body_Temp', 'Heart_Rate']].values
    y = dataset['Calories'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=3)

    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    return model, poly

try:
    model, poly = train_model()

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

    if st.button("Predict Calories 🔥", use_container_width=True):
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
    st.caption("Model: Polynomial Regression (degree=2) | R² = 0.9622 | Best performing model among 7 algorithms")

except FileNotFoundError:
    st.error("Dataset files not found. Please make sure calories.csv and exercise.csv are in the same directory.")
    st.info("Download the dataset from: https://www.kaggle.com/datasets/fmendes/fmendesdat263xdemos")

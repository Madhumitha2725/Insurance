# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Train model with caching to improve performance
@st.cache_data
def train_model():
    df = pd.read_csv("Insurancepredc.csv")

    # Label Encoding
    le_sex = LabelEncoder()
    le_region = LabelEncoder()
    le_smoker = LabelEncoder()

    df['sex'] = le_sex.fit_transform(df['sex'])
    df['region'] = le_region.fit_transform(df['region'])
    df['smoker'] = le_smoker.fit_transform(df['smoker'])

    # Prepare features and target
    X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
    y = df['charges']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, le_sex, le_smoker, le_region

# Load model and encoders
model, le_sex, le_smoker, le_region = train_model()

# Streamlit page config
st.set_page_config(page_title="💰 Insurance Cost Predictor", layout="centered")
st.title("🏥 Medical Insurance Cost Predictor")
st.markdown("Predict your expected **medical insurance charges** by entering the details below.")

# Input UI
st.subheader("🔎 Personal & Lifestyle Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🎂 Age", min_value=1, max_value=120, value=30)
    sex = st.radio("👤 Sex", options=["male", "female"])
    bmi = st.number_input("⚖️ BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=25.0)

with col2:
    children = st.slider("👶 Number of Children", 0, 10, 0)
    smoker = st.selectbox("🚬 Do you smoke?", ["yes", "no"])
    region = st.selectbox("🌎 Region", ["southwest", "southeast", "northwest", "northeast"])

# Predict button
if st.button("🔮 Predict Insurance Charges"):
    # Encode inputs
    sex_encoded = le_sex.transform([sex])[0]
    smoker_encoded = le_smoker.transform([smoker])[0]
    region_encoded = le_region.transform([region])[0]

    input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])
    predicted_charge = model.predict(input_data)[0]

    st.success(f"💸 Estimated Insurance Cost: ₹{predicted_charge:,.2f}")

    # Show a tip if smoker
    if smoker == "yes":
        st.warning("💡 Tip: Quitting smoking can help lower your insurance costs!")

# Footer
st.markdown("---")
st.caption("🔧 Created with ❤️ using Streamlit | Project by Madhu Mitha")

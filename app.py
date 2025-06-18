# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Train model with caching
@st.cache_data
def train_model():
    df = pd.read_csv("Insurancepredc.csv")

    # Standardize cases
    df['sex'] = df['sex'].str.title()
    df['smoker'] = df['smoker'].str.title()
    df['region'] = df['region'].str.lower()

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

# Title and intro
st.title("🏥 Medical Insurance Cost Predictor")

# Moving banner
st.markdown("""
<marquee behavior="scroll" direction="left" scrollamount="8" style="background-color: #f0f8ff; color: #00008b; padding: 10px; font-size: 20px; font-weight: bold; border-radius: 10px;">
Welcome! Predict your expected **medical insurance charges** by entering the details below. Stay Healthy, Stay Informed 💪
</marquee>
""", unsafe_allow_html=True)

# Input form with wrapper box
with st.form("prediction_form", clear_on_submit=False):
    st.markdown("""
    <div style='border: 2px solid #4B8BBE; padding: 20px; border-radius: 10px; background-color: #f9f9f9;'>
    """, unsafe_allow_html=True)

    st.subheader("🧾 Enter Personal Information")

    age = st.slider("🎂 Age", 18, 100, 30)
    bmi = st.number_input("⚖️ BMI", min_value=10.0, max_value=50.0, value=25.0)
    children = st.number_input("👶 Number of Children", min_value=0, max_value=10, step=1)
    sex = st.radio("🧑 Sex", ['Male', 'Female'], horizontal=True)
    smoker = st.radio("🚬 Smoker", ['Yes', 'No'], horizontal=True)
    region = st.selectbox("🌍 Region", ['southwest', 'southeast', 'northwest', 'northeast'])

    st.markdown("</div>", unsafe_allow_html=True)

    submit = st.form_submit_button("🔍 Estimate Charges")

# Prediction
if submit:
    try:
        # Match encoding format
        sex_input = sex.title()
        smoker_input = smoker.title()
        region_input = region.lower()

        # Encode inputs
        sex_encoded = le_sex.transform([sex_input])[0]
        smoker_encoded = le_smoker.transform([smoker_input])[0]
        region_encoded = le_region.transform([region_input])[0]

        input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])
        predicted_charge = model.predict(input_data)[0]

        st.success(f"💸 Estimated Insurance Cost: ₹{predicted_charge:,.2f}")

        if smoker_input == "Yes":
            st.warning("🚭 Tip: Quitting smoking can help lower your insurance costs and improve your health!")
        else:
            st.info("👏 Awesome! Being a non-smoker helps reduce your medical risks and insurance charges!")

    except ValueError as e:
        st.error("🚫 Error: There was an issue with input encoding. Please ensure valid values.")

# Footer
st.markdown("---")
st.caption("🔧 Project by Madhu Mitha")

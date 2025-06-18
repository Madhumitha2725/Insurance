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

# Input UI - stacked vertically
st.subheader("🔎 Personal & Lifestyle Information")

age = st.slider("🎂 Age", 18, 100, 30)
sex = st.radio("👤 Sex", options=["male", "female"])
bmi = st.slider("⚖️ BMI (Body Mass Index)", 10.0, 50.0, 25.0)
children = st.slider("👶 Number of Children", 0, 10, 0)
smoker = st.selectbox("🚬 Do you smoke?", ["yes", "no"])
region = st.selectbox("🌎 Region", ["southwest", "southeast", "northwest", "northeast"])

# BMI Health Feedback
st.subheader("📊 BMI Category")
if bmi < 18.5:
    st.warning("📉 Underweight")
elif 18.5 <= bmi < 25:
    st.success("✅ Normal")
elif 25 <= bmi < 30:
    st.info("⚠️ Overweight")
else:
    st.error("❗ Obese - High health risk")

# Predict button
if st.button("🔮 Predict Insurance Charges"):
    # Encode inputs
    sex_encoded = le_sex.transform([sex])[0]
    smoker_encoded = le_smoker.transform([smoker])[0]
    region_encoded = le_region.transform([region])[0]

    input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])
    predicted_charge = model.predict(input_data)[0]

    # Show result
    st.success(f"💸 Estimated Insurance Cost: ₹{predicted_charge:,.2f}")

    # Tip if smoker
    if smoker == "yes":
        st.warning("💡 Tip: Quitting smoking can help lower your insurance costs!")

    # Summary
    st.subheader("📋 Summary of Your Inputs")
    st.markdown(f"- Age: **{age}**")
    st.markdown(f"- Sex: **{sex}**")
    st.markdown(f"- BMI: **{bmi}**")
    st.markdown(f"- Number of Children: **{children}**")
    st.markdown(f"- Smoker: **{smoker}**")
    st.markdown(f"- Region: **{region}**")

    # Chart - Compare with average
    avg_cost = 13200  # You can update this based on your dataset
    df = pd.DataFrame({
        'Type': ['Average Cost', 'Your Prediction'],
        'Amount': [avg_cost, predicted_charge]
    })
    st.subheader("📈 Comparison with Average")
    st.bar_chart(df.set_index('Type'))

    # Download as CSV
    st.subheader("📥 Download Your Prediction")
    result_df = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "BMI": bmi,
        "Children": children,
        "Smoker": smoker,
        "Region": region,
        "Predicted Insurance Cost (₹)": f"{predicted_charge:,.2f}"
    }])
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Result as CSV", csv, "insurance_result.csv", "text/csv")

# Footer
st.markdown("---")
st.caption("Project by Madhumitha")

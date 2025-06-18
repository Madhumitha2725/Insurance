import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and encoders
model = pickle.load(open('model.pkl', 'rb'))
encoders = pickle.load(open('label_encoders.pkl', 'rb'))

le_sex = encoders['sex']
le_smoker = encoders['smoker']
le_region = encoders['region']

# Page config
st.set_page_config(page_title="💰 Insurance Cost Predictor", layout="centered")

# Title and instructions
st.title("🏥 Medical Insurance Cost Prediction")
st.markdown("Predict your estimated **insurance charges** based on your lifestyle and personal details.")

st.divider()
st.subheader("📝 Enter Your Details Below")

# Inputs (Vertically stacked)
age = st.slider("🎂 Age", 18, 100, 30)
sex = st.radio("👤 Sex", le_sex.classes_)
bmi = st.slider("⚖️ BMI (Body Mass Index)", 10.0, 50.0, 25.0)
children = st.slider("👶 Number of Children", 0, 10, 0)
smoker = st.radio("🚬 Do you smoke?", le_smoker.classes_)
region = st.selectbox("🌍 Region", le_region.classes_)

# BMI Health Category Indicator
st.subheader("📊 BMI Category")
if bmi < 18.5:
    st.warning("📉 Underweight")
elif 18.5 <= bmi < 25:
    st.success("✅ Normal")
elif 25 <= bmi < 30:
    st.info("⚠️ Overweight")
else:
    st.error("❗ Obese")

st.divider()

# Prediction
if st.button("🔮 Predict Insurance Cost"):
    # Encode input
    sex_encoded = le_sex.transform([sex])[0]
    smoker_encoded = le_smoker.transform([smoker])[0]
    region_encoded = le_region.transform([region])[0]
    input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])
    predicted_cost = model.predict(input_data)[0]

    # Result
    st.success(f"💸 Estimated Insurance Cost: ₹{predicted_cost:,.2f}")

    # Tip if smoker
    if smoker == "yes":
        st.warning("💡 Tip: Quitting smoking may reduce your insurance cost significantly!")

    # Summary
    st.subheader("📋 Summary of Your Inputs")
    st.markdown(f"- Age: **{age}**")
    st.markdown(f"- Sex: **{sex}**")
    st.markdown(f"- BMI: **{bmi}**")
    st.markdown(f"- Number of Children: **{children}**")
    st.markdown(f"- Smoker: **{smoker}**")
    st.markdown(f"- Region: **{region}**")

    # Comparison Chart
    avg_cost = 13200  # Example average
    df = pd.DataFrame({
        'Type': ['Average Cost', 'Your Prediction'],
        'Amount': [avg_cost, predicted_cost]
    })
    st.subheader("📈 Comparison with Average")
    st.bar_chart(df.set_index('Type'))

    # Download prediction
    st.subheader("📥 Download Your Prediction")
    result_df = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "BMI": bmi,
        "Children": children,
        "Smoker": smoker,
        "Region": region,
        "Predicted Insurance Cost (₹)": f"{predicted_cost:,.2f}"
    }])
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Result as CSV", csv, "insurance_result.csv", "text/csv")

st.markdown("---")
st.caption("🔧 Built with ❤️ by Madhu Mitha | Powered by Streamlit & Machine Learning")

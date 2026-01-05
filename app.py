import os
import streamlit as st
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "credit_risk_pipeline.pkl")

st.write("Files in app directory:", os.listdir(BASE_DIR))

pipeline = joblib.load(MODEL_PATH)


# ----------------------------
# Load trained pipeline
# ----------------------------


st.set_page_config(page_title="Credit Risk Engine", layout="centered")

st.title("💳 Credit Risk Engine")
st.write("Predict Probability of Default and Lending Decision")

# ----------------------------
# User Inputs
# ----------------------------
st.header("Customer Information")

limit_bal = st.number_input(
    "Credit Limit",
    min_value=0,
    value=50000,
    step=5000
)

age = st.number_input(
    "Age",
    min_value=18,
    value=30
)

num_late_payments = st.number_input(
    "Number of Late Payments (last 6 months)",
    min_value=0,
    value=0
)

credit_utilisation = st.slider(
    "Credit Utilisation Ratio",
    min_value=0.0,
    max_value=1.0,
    value=0.3
)

payment_ratio = st.slider(
    "Payment-to-Bill Ratio",
    min_value=0.0,
    max_value=1.0,
    value=0.7
)

# ----------------------------
# Derived Features
# ----------------------------
avg_delay = num_late_payments  # simplified assumption
high_risk_flag = int(
    (num_late_payments >= 3) and (credit_utilisation > 0.8)
)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Evaluate Risk"):
    input_df = pd.DataFrame([{
        "num_late_payments": num_late_payments,
        "avg_delay": avg_delay,
        "credit_utilization": credit_utilisation,
        "payment_ratio": payment_ratio,
        "high_risk_flag": high_risk_flag,
        "LIMIT_BAL": limit_bal,
        "AGE": age
    }])

    pd_score = pipeline.predict_proba(input_df)[0][1]

    # ----------------------------
    # Risk Band Logic
    # ----------------------------
    if pd_score < 0.15:
        risk_band = "Low Risk"
        decision = "Approve"
    elif pd_score < 0.35:
        risk_band = "Medium Risk"
        decision = "Manual Review"
    else:
        risk_band = "High Risk"
        decision = "Reject"

    # ----------------------------
    # Display Results
    # ----------------------------
    st.subheader("Result")
    st.metric("Probability of Default (PD)", f"{pd_score:.2%}")
    st.write(f"**Risk Band:** {risk_band}")
    st.write(f"**Decision:** {decision}")

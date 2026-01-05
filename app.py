import streamlit as st
import os
import joblib
import sys

st.write("✅ Streamlit started")
st.write("Python version:", sys.version)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
st.write("Base directory:", BASE_DIR)

try:
    files = os.listdir(BASE_DIR)
    st.write("Files in directory:", files)
except Exception as e:
    st.error(f"❌ Could not list directory: {e}")

MODEL_PATH = os.path.join(BASE_DIR, "credit_risk_pipeline.pkl")
st.write("Model path:", MODEL_PATH)

try:
    pipeline = joblib.load(MODEL_PATH)
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error("❌ Model failed to load")
    st.exception(e)
    st.stop()



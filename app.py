import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "house_price_ann.keras")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.joblib")

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

st.markdown("""
<style>
.main { padding-top: 1rem; }
.hero { padding: 1.5rem; border-radius: 18px; background: linear-gradient(135deg, #2563eb, #7c3aed); color: white; margin-bottom: 1.5rem; }
.hero h1 { margin-bottom: 0.25rem; }
.result { padding: 1.5rem; border-radius: 16px; background: #eff6ff; border: 1px solid #bfdbfe; text-align: center; }
.price { font-size: 2.2rem; font-weight: 800; color: #1d4ed8; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🏠 House Price Prediction</h1>
<p>Deep Learning powered by an Artificial Neural Network (ANN)</p>
</div>
""", unsafe_allow_html=True)

if not (os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH)):
    st.warning("The model has not been trained yet. Run `python train_model.py` first.")
    st.stop()

@st.cache_resource
def load_assets():
    return load_model(MODEL_PATH), joblib.load(PREPROCESSOR_PATH), joblib.load(METADATA_PATH)

model, preprocessor, metadata = load_assets()

st.subheader("Enter house details")
col1, col2 = st.columns(2)
with col1:
    area = st.number_input("Area (sq ft)", min_value=300, max_value=10000, value=1500, step=50)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
    floors = st.number_input("Floors", min_value=1, max_value=5, value=2, step=1)
with col2:
    house_age = st.number_input("House age (years)", min_value=0, max_value=150, value=8, step=1)
    distance = st.number_input("Distance from city center (km)", min_value=0.1, max_value=100.0, value=8.0, step=0.5)
    location_type = st.selectbox("Location type", ["Urban", "Suburban", "Rural"])
    parking = st.selectbox("Parking available", ["Yes", "No"])

if st.button("Predict House Price", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        "area_sqft": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "floors": floors,
        "house_age": house_age,
        "city_distance_km": distance,
        "location_type": location_type,
        "parking": parking,
    }])
    processed = preprocessor.transform(input_df).astype("float32")
    prediction = float(model.predict(processed, verbose=0)[0][0])
    prediction = max(0, prediction)

    st.markdown(f"""
    <div class="result">
      <div>Estimated House Price</div>
      <div class="price">₹{prediction:,.0f}</div>
      <div>Approx. ₹{prediction / 100000:.2f} lakh</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.subheader("Model performance")
metrics = metadata.get("metrics", {})
m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"₹{metrics.get('mae', 0):,.0f}")
m2.metric("RMSE", f"₹{metrics.get('rmse', 0):,.0f}")
m3.metric("R² Score", f"{metrics.get('r2', 0):.3f}")
st.caption("Note: This demo uses a generated sample dataset. Replace the CSV with real local housing data for meaningful predictions.")

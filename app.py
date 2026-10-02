import streamlit as st
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Hydroelectric Power Predictor",
    page_icon="⚡",
    layout="centered"
)

# Load trained artifacts with caching
@st.cache_resource
def load_models():
    model = joblib.load("hydro_power_model.pkl")
    scaler = joblib.load("hydro_scaler.pkl")
    return model, scaler

try:
    model, scaler = load_models()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model files: {e}. Please ensure 'hydro_power_model.pkl' and 'hydro_scaler.pkl' are in the same folder.")

# UI Header
st.title("⚡ Hydroelectric Dam Power Predictor")
st.write("Estimate electricity generation (in Megawatts) using regularized Ridge Regression.")
st.write("---")

# Input columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Water Reservoir")
    level = st.slider("Reservoir Water Level (m)", 150.0, 320.0, 220.0, step=1.0)
    head = st.slider("Hydraulic Head (m)", 120.0, 280.0, 190.0, step=1.0)
    temp = st.slider("Water Temperature (°C)", 4.0, 26.0, 12.0, step=0.5)

with col2:
    st.subheader("Turbine Dynamics")
    flow = st.slider("Water Flow Rate (m³/s)", 10.0, 350.0, 140.0, step=5.0)
    gate = st.slider("Turbine Gate Opening (%)", 20.0, 100.0, 75.0, step=1.0)

st.write("---")

# Prediction action
if st.button("Calculate Power Output", use_container_width=True):
    if model_loaded:
        # Arrange features in training order:
        # [Reservoir_Water_Level_m, Hydraulic_Head_m, Water_Flow_Rate_m3s, Turbine_Gate_Opening_pct, Water_Temperature_C]
        input_data = np.array([[level, head, flow, gate, temp]])
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]

        st.success("Prediction Generated!")
        st.metric(label="Predicted Power Generation", value=f"{prediction:,.2f} MW")

        if prediction > 350.0:
            st.info("High Generation: Dam operating near peak capacity.")
        else:
            st.caption("Standard operational output range.")

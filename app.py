# ----------------------------------------------------------------------
# Streamlit Application: Heart Failure Prediction
# ----------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time

# Set Streamlit page configuration
st.set_page_config(
    page_title="Heart Failure Risk Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Load the Deployment Artifact ---

ARTIFACT_FILE = "heart_failure_deployment_artifact.pkl"

@st.cache_resource
def load_deployment_artifact(file_path):
    """Loads the model, scaler, and feature list from the pkl file."""
    if not os.path.exists(file_path):
        st.error(f"Error: Model artifact '{file_path}' not found.")
        st.stop()
    
    try:
        with open(file_path, 'rb') as f:
            artifact = pickle.load(f)
        return artifact
    except Exception as e:
        st.error(f"Error loading artifact: {e}")
        st.stop()

# Load model, scaler, and features
try:
    artifact = load_deployment_artifact(ARTIFACT_FILE)
    model = artifact['model']
    scaler = artifact['scaler']
    FEATURE_COLUMNS = artifact['features']
    metadata = artifact['metadata']
except Exception as e:
    # This block handles critical errors during artifact loading
    st.error("Application Initialization Failed. Check if 'heart_failure_deployment_artifact.pkl' is correct.")
    st.stop()

# --- 2. Application UI ---

st.title("💔 Heart Failure Risk Prediction")
st.markdown("Use the input form below to predict the probability of a death event due to heart failure.")

# Display model metadata
st.sidebar.header("Model Info")
st.sidebar.markdown(f"**Type:** {metadata.get('model_type', 'N/A')}")
st.sidebar.markdown(f"**Test Accuracy:** {metadata.get('accuracy_on_test_set', 0.0)*100:.2f}%")
st.sidebar.markdown("This app uses the features derived from `train_model.py`.")


# --- Input Form ---
with st.form("prediction_form"):
    
    col1, col2, col3 = st.columns(3)

    # Basic Info
    with col1:
        age = st.slider("Age (Years)", 40, 95, 60, 1)
        sex = st.selectbox("Sex", options=["Male", "Female"], index=0)
        smoking = st.selectbox("Smoking", options=["No", "Yes"], index=0)
        
    # Vital Metrics
    with col2:
        ejection_fraction = st.slider("Ejection Fraction (%)", 20, 80, 40, 5)
        serum_creatinine = st.slider("Serum Creatinine (mg/dL)", 0.5, 9.4, 1.3, 0.1)
        serum_sodium = st.slider("Serum Sodium (mEq/L)", 113, 148, 137, 1)

    # Clinical History
    with col3:
        anaemia = st.selectbox("Anaemia", options=["No", "Yes"], index=0)
        high_blood_pressure = st.selectbox("High Blood Pressure", options=["No", "Yes"], index=0)
        diabetes = st.selectbox("Diabetes", options=["No", "Yes"], index=0)
        # Assuming follow-up time is needed for prediction
        time_days = st.slider("Follow-up Period (Days)", 0, 300, 150, 1)
        creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase (mcg/L)", 23, 7861, 582, 1)
        platelets = st.number_input("Platelets (kiloplatelets/mL)", 25000, 850000, 263358, 1000)

    # Convert inputs to required numerical format (0/1)
    sex_val = 1 if sex == "Male" else 0
    smoking_val = 1 if smoking == "Yes" else 0
    anaemia_val = 1 if anaemia == "Yes" else 0
    high_blood_pressure_val = 1 if high_blood_pressure == "Yes" else 0
    diabetes_val = 1 if diabetes == "Yes" else 0
    
    submitted = st.form_submit_button("Predict Risk")

# --- 3. Prediction Logic ---

if submitted:
    
    # 3.1. Create DataFrame for Unprocessed Input
    raw_data = {
        'age': [age],
        'creatinine_phosphokinase': [creatinine_phosphokinase],
        'ejection_fraction': [ejection_fraction],
        'platelets': [platelets],
        'serum_creatinine': [serum_creatinine],
        'serum_sodium': [serum_sodium],
        'time': [time_days],
        # Binary features (must be in the order they appear in FEATURE_COLUMNS, or pandas handles it)
        'anaemia': [anaemia_val],
        'diabetes': [diabetes_val],
        'high_blood_pressure': [high_blood_pressure_val],
        'sex': [sex_val],
        'smoking': [smoking_val],
    }
    input_df = pd.DataFrame(raw_data)

    # 3.2. Apply Feature Engineering (Binning and One-Hot Encoding) - MUST match train_model.py
    
    # Define bins based on training script
    age_bins = [0, 50, 70, 100]
    ef_bins = [0, 30, 50, 100]
    ss_bins = [0, 135, 145, 200]
    
    # Apply binning
    input_df["age_group"] = pd.cut(input_df["age"], bins=age_bins, labels=["Young", "Middle", "Old"], right=False)
    input_df["ejection_fraction_group"] = pd.cut(input_df["ejection_fraction"], bins=ef_bins, labels=["Low", "Medium", "High"], right=False)
    input_df["serum_sodium_group"] = pd.cut(input_df["serum_sodium"], bins=ss_bins, labels=["Low", "Normal", "High"], right=False)

    # Drop original columns (they are included in raw_data, but dropped/encoded below)
    input_df = input_df.drop(columns=['age', 'ejection_fraction', 'serum_sodium'], errors='ignore')
    
    # Apply One-Hot Encoding
    input_encoded = pd.get_dummies(input_df, 
                                columns=["age_group", "ejection_fraction_group", "serum_sodium_group"],
                                drop_first=True)

    # 3.3. Align Feature Columns and Scale - CRUCIAL STEP
    
    # Find all columns that are in the full feature list but NOT in the user input.
    # These are the columns created by drop_first=True, which might be missing if the user input 
    # happened to fall into the dropped category (e.g., 'Young', 'Low', 'Normal').
    missing_cols = set(FEATURE_COLUMNS) - set(input_encoded.columns)

    # Add missing columns with a value of 0
    for c in missing_cols:
        input_encoded[c] = 0

    # Ensure the order of columns matches the training data exactly
    final_input_df = input_encoded[FEATURE_COLUMNS]

    # Scale the input data using the fitted scaler
    scaled_input = scaler.transform(final_input_df)

    # 3.4. Predict
    with st.spinner('Calculating risk...'):
        time.sleep(1) # Simulate computation time
        
        # SVC prediction returns 0 or 1.
        prediction = model.predict(scaled_input)[0] 
        
        # SVC does not natively support probability (predict_proba) unless initialized with probability=True.
        # We'll use the hard prediction (0 or 1) and provide an interpretation.
        
        if prediction == 1:
            st.error(
                """
                ### High Risk Predicted! 🚨
                Based on the provided clinical data, the model predicts a **High Risk** of a death event due to heart failure.
                """
            )
            st.markdown(
                """
                **Disclaimer:** This is an AI model prediction and should **not** replace professional medical advice. Please consult a healthcare professional immediately.
                """
            )
        else:
            st.success(
                """
                ### Low Risk Predicted ✅
                Based on the provided clinical data, the model predicts a **Low Risk** of a death event due to heart failure.
                """
            )
            st.markdown(
                """
                **Note:** While the risk is low according to the model, ongoing monitoring and professional medical consultation are always recommended.
                """
            )
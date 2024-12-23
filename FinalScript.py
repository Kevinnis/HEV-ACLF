import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sksurv
import shap
model = joblib.load("gbm_mod1.pkl")

# Define feature names
feature_names = ["INR", "TBIL", "Na", "HDL-C", "URA"]

# Streamlit user interface
st.set_page_config(
    page_title="HEV-ACLF Risk Predictor",
    page_icon=":microbe:",  # Adding a custom icon (microbe emoji)
    layout="centered",  # Center the content
)

st.title("HEV-ACLF Risk Predictor")
st.caption('This online tool was developed to predict the risk of hepatitis E virus-related acute-on-chronic liver failure among hospitalized patients with HEV infection')

# Customizing the appearance of the input form using streamlit's markdown styling
st.markdown("""
    <style>
        .stNumberInput {
            width: 100%;
            padding: 10px;
            font-size: 16px;
            border-radius: 5px;
            border: 1px solid #ccc;
        }
    </style>
""", unsafe_allow_html=True)

# Numerical input
INR = st.number_input("International normalized ratio (INR)", min_value=0.0, max_value=100.0, format="%.2f", key="NEU")
TBIL = st.number_input("Total bilirubin (TBIL) (μmol/L)", min_value=0.0, max_value=10000.0, format="%.2f", key="MONO")
Na = st.number_input("Na (mmol/L)", min_value=0.0, max_value=1000.0, format="%.2f", key="INR")
HDL = st.number_input("High-density lipoprotein-cholesterol (HDL-C) (mmol/L)", min_value=0.0, max_value=10000.0, format="%.2f", key="AST")
URA = st.number_input("Uric acid (URA) (μmol/L)", min_value=0.0, max_value=10000.0, format="%.2f", key="ALB")

# Z-score transformation (standardization)
INR = (INR - 1.244769) / 0.362144
TBIL = (TBIL - 127.8267) / 123.9332
Na = (Na - 138.7888) / 3.736281
HDL = (HDL - 0.7207303) / 0.4138742
URA = (URA - 279.6014) / 118.7426

feature_values = [INR, TBIL, Na, HDL, URA]
features = np.array([feature_values])

# Center the predict button
st.markdown("""
    <style>
    div.stButton > button:first-child {
        display: block;
        margin: 0 auto;
    }
    </style>""", unsafe_allow_html=True)

# Predict button
if st.button("Predict"):
    # Predict risk score
    risk_score = model.predict(features)[0]

    # Get the cumulative hazard function for each individual (hazard function)
    hazard_functions = model.predict_cumulative_hazard_function(features)

    # Calculate the death probabilities at 7, 14, and 28 days
    death_probabilities = []
    for hazard in hazard_functions:
        # Get the cumulative hazard at 7, 14, and 28 days
        hazard_7 = hazard(7)  # 7 days
        hazard_14 = hazard(14)  # 14 days
        hazard_28 = hazard(28)  # 28 days
        
        # Calculate the death probability
        prob_7 = 1 - np.exp(-hazard_7)
        prob_14 = 1 - np.exp(-hazard_14)
        prob_28 = 1 - np.exp(-hazard_28)
        
        # Append the death probabilities for this sample
        death_probabilities.append([prob_7, prob_14, prob_28])
    
    # Convert to numpy array for easy handling and rounding
    death_probabilities = np.array(death_probabilities)

    st.markdown("<h3 style='font-weight: bold;'>Prediction Results</h3>", unsafe_allow_html=True)
    # Display Risk Score
    st.markdown(f"<h3 style='text-align: center;'>Risk Score: {risk_score:.4f}</h3>", unsafe_allow_html=True)

    # Display HEV-ACLF onset risk based on threshold
    if risk_score >= 0.4458827:
        st.markdown(f"<h3 style='text-align: center; color: red;'>7 day HEV-ACLF probability: {death_probabilities[0][0]*100:.2f}% (High Risk)</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='text-align: center; color: green;'>7 day HEV-ACLF probability: {death_probabilities[0][0]*100:.2f}% (Low Risk)</h3>", unsafe_allow_html=True)

    if risk_score >= 0.4458827:
        st.markdown(f"<h3 style='text-align: center; color: red;'>14 day HEV-ACLF probability: {death_probabilities[0][1]*100:.2f}% (High Risk)</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='text-align: center; color: green;'>14 day HEV-ACLF probability: {death_probabilities[0][1]*100:.2f}% (Low Risk)</h3>", unsafe_allow_html=True)

    if risk_score >= 0.3980038:
        st.markdown(f"<h3 style='text-align: center; color: red;'>28 day HEV-ACLF probability: {death_probabilities[0][2]*100:.2f}% (High Risk)</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='text-align: center; color: green;'>28 day HEV-ACLF probability: {death_probabilities[0][2]*100:.2f}% (Low Risk)</h3>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='font-weight: bold;'>Prediction Interpretations</h3>", unsafe_allow_html=True)
    st.caption('The explanations for this prediction are shown below. Please note the prediction results should be interpreted by medical professionals only.')

    # Compute SHAP values
    explainer = joblib.load('shap_explainer.pkl')
    shap_values = explainer(features)
    
    # Create a figure for the SHAP force plot
    features_df = pd.DataFrame([feature_values], columns=feature_names)
    shap.plots.force(shap_values.base_values,
                     shap_values.values[0],
                     pd.DataFrame([features_df.iloc[0].values], columns=features_df.columns), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")

st.caption('Version: 20241223 [This is currently a demo version for review]')
st.caption('Contact: wangjienjmu@126.com')

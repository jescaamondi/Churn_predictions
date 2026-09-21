import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model
@st.cache_resource
def load_model_and_preprocessor():
    model = joblib.load('best_churn_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, preprocessor, feature_names

try:
    model, preprocessor, feature_names = load_model_and_preprocessor()
    st.sidebar.success("✅ Model loaded!")
except FileNotFoundError:
    st.error("❌ Model files not found. Train the model first.")
    st.stop()

# Sidebar form
st.sidebar.header("👤 Customer Info")
st.sidebar.markdown("---")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.sidebar.selectbox("Payment Method", 
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
phone_service = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, 0.5)
total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0, 10.0)

# Main
st.title("📊 Customer Churn Predictor")
st.markdown("Predict customer churn risk in real-time.")

# Create input
input_data = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
    'Partner': [1 if partner == "Yes" else 0],
    'Dependents': [1 if dependents == "Yes" else 0],
    'tenure': [tenure],
    'PhoneService': [phone_service],
    'MultipleLines': [multiple_lines],
    'InternetService': [internet_service],
    'OnlineSecurity': [online_security],
    'OnlineBackup': [online_backup],
    'DeviceProtection': [device_protection],
    'TechSupport': [tech_support],
    'StreamingTV': [streaming_tv],
    'StreamingMovies': [streaming_movies],
    'Contract': [contract],
    'PaperlessBilling': [1 if paperless_billing == "Yes" else 0],
    'PaymentMethod': [payment_method],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges]
})

# Feature engineering
input_data['tenure_to_monthly_ratio'] = input_data['tenure'] / (input_data['MonthlyCharges'] + 1)
input_data['total_to_monthly_ratio'] = input_data['TotalCharges'] / (input_data['MonthlyCharges'] + 1)
input_data['avg_monthly_spend'] = input_data['TotalCharges'] / (input_data['tenure'] + 1)

service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for col in service_cols:
    input_data[f'{col}_bin'] = input_data[col].map({'Yes': 1, 'No': 0, 'No phone service': 0, 'No internet service': 0})
input_data['service_count'] = input_data[[f'{col}_bin' for col in service_cols]].sum(axis=1)
input_data['streaming_count'] = input_data['StreamingTV_bin'] + input_data['StreamingMovies_bin']
input_data['protection_count'] = input_data['OnlineSecurity_bin'] + input_data['DeviceProtection_bin'] + input_data['TechSupport_bin']
input_data.drop(columns=[f'{col}_bin' for col in service_cols], axis=1, inplace=True)

input_data['contract_risk_score'] = input_data['Contract'].map({'Month-to-month': 3, 'One year': 2, 'Two year': 1})
input_data['payment_risk_score'] = (input_data['PaymentMethod'] == 'Electronic check').astype(int)
input_data['internet_risk_score'] = input_data['InternetService'].map({'No': 0, 'DSL': 1, 'Fiber optic': 2})

# Ensure column order
input_data = input_data[feature_names]

# Predict
if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
    with st.spinner("Analyzing..."):
        input_processed = preprocessor.transform(input_data)
        churn_proba = model.predict_proba(input_processed)[0, 1]
        churn_pred = model.predict(input_processed)[0]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if churn_proba < 0.3:
                st.markdown("### 🟢 LOW RISK")
                st.info("This customer is unlikely to churn.")
            elif churn_proba < 0.6:
                st.markdown("### 🟡 MEDIUM RISK")
                st.warning("This customer has moderate churn risk.")
            else:
                st.markdown("### 🔴 HIGH RISK")
                st.error("This customer is at high risk of churning!")
            
            st.markdown(f"**Churn Probability: {churn_proba*100:.1f}%**")
        
        with col2:
            st.markdown("#### Risk Gauge")
            st.progress(min(churn_proba, 1.0))
        
        st.markdown("---")
        st.metric("Prediction", "Churn" if churn_pred == 1 else "No Churn")
        st.metric("Probability", f"{churn_proba*100:.2f}%")

st.markdown("---")
st.caption("Built with XGBoost | Portfolio Project")

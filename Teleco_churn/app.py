import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model and Scaler
# -----------------------------
model = joblib.load("Teleco_churn\\logistic_regression.pkl")
scaler = joblib.load("Teleco_churn\\standard_scaler.pkl")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn.")

st.divider()

# -----------------------------
# User Inputs
# -----------------------------
col1,col2,col3,col4=st.columns(4)
with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    phone = st.selectbox("Phone Service", ["No", "Yes"])
with col2:
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    multiple = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )
    contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
    )
with col3:
    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )
    device = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )
    paperless = st.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
    )
with col4:
    tech = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

    tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )

    movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )
    payment = st.selectbox(
    "Payment Method",
    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
    )

tenure = st.slider("Tenure (Months)", 0, 72, 12)

monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Churn"):

    data = {
        'SeniorCitizen': 1 if senior == "Yes" else 0,
        'tenure': tenure,
        'MonthlyCharges': monthly,
        'TotalCharges': total,

        'gender_Male': 1 if gender == "Male" else 0,
        'Partner_Yes': 1 if partner == "Yes" else 0,
        'Dependents_Yes': 1 if dependents == "Yes" else 0,
        'PhoneService_Yes': 1 if phone == "Yes" else 0,

        'MultipleLines_No phone service': 1 if multiple == "No phone service" else 0,
        'MultipleLines_Yes': 1 if multiple == "Yes" else 0,

        'InternetService_Fiber optic': 1 if internet == "Fiber optic" else 0,
        'InternetService_No': 1 if internet == "No" else 0,

        'OnlineSecurity_No internet service': 1 if online_security == "No internet service" else 0,
        'OnlineSecurity_Yes': 1 if online_security == "Yes" else 0,

        'OnlineBackup_No internet service': 1 if online_backup == "No internet service" else 0,
        'OnlineBackup_Yes': 1 if online_backup == "Yes" else 0,

        'DeviceProtection_No internet service': 1 if device == "No internet service" else 0,
        'DeviceProtection_Yes': 1 if device == "Yes" else 0,

        'TechSupport_No internet service': 1 if tech == "No internet service" else 0,
        'TechSupport_Yes': 1 if tech == "Yes" else 0,

        'StreamingTV_No internet service': 1 if tv == "No internet service" else 0,
        'StreamingTV_Yes': 1 if tv == "Yes" else 0,

        'StreamingMovies_No internet service': 1 if movies == "No internet service" else 0,
        'StreamingMovies_Yes': 1 if movies == "Yes" else 0,

        'Contract_One year': 1 if contract == "One year" else 0,
        'Contract_Two year': 1 if contract == "Two year" else 0,

        'PaperlessBilling_Yes': 1 if paperless == "Yes" else 0,

        'PaymentMethod_Credit card (automatic)': 1 if payment == "Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check': 1 if payment == "Electronic check" else 0,
        'PaymentMethod_Mailed check': 1 if payment == "Mailed check" else 0,
    }

    feature_order = [
        'SeniorCitizen',
        'tenure',
        'MonthlyCharges',
        'TotalCharges',
        'gender_Male',
        'Partner_Yes',
        'Dependents_Yes',
        'PhoneService_Yes',
        'MultipleLines_No phone service',
        'MultipleLines_Yes',
        'InternetService_Fiber optic',
        'InternetService_No',
        'OnlineSecurity_No internet service',
        'OnlineSecurity_Yes',
        'OnlineBackup_No internet service',
        'OnlineBackup_Yes',
        'DeviceProtection_No internet service',
        'DeviceProtection_Yes',
        'TechSupport_No internet service',
        'TechSupport_Yes',
        'StreamingTV_No internet service',
        'StreamingTV_Yes',
        'StreamingMovies_No internet service',
        'StreamingMovies_Yes',
        'Contract_One year',
        'Contract_Two year',
        'PaperlessBilling_Yes',
        'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check',
        'PaymentMethod_Mailed check'
    ]

    df = pd.DataFrame([data])[feature_order]

    scaled = scaler.transform(df)

    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    st.divider()

    if prediction == 1:
        st.error("⚠️ Customer is likely to Churn")
    else:
        st.success("✅ Customer is likely to Stay")

    st.write(f"**Churn Probability:** {probability:.2%}")

    st.progress(float(probability))
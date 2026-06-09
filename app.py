import streamlit as st
import pandas as pd
import pickle

# Page Configuration
st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide"
)

# Load Model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Title
st.title("💳 Credit Card Defaulter Prediction System")
st.markdown("---")

st.write(
    """
    Enter customer details below to predict whether
    the customer is likely to default on their credit card payment.
    """
)

# Sidebar Inputs
st.sidebar.header("Customer Information")

income = st.sidebar.number_input(
    "Annual Income",
    min_value=0.0,
    value=50000.0
)

age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

loan_amount = st.sidebar.number_input(
    "Loan Amount",
    min_value=0.0,
    value=10000.0
)

credit_score = st.sidebar.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

months_employed = st.sidebar.number_input(
    "Months Employed",
    min_value=0,
    value=24
)

num_credit_lines = st.sidebar.number_input(
    "Number of Credit Lines",
    min_value=0,
    value=3
)

interest_rate = st.sidebar.number_input(
    "Interest Rate",
    min_value=0.0,
    value=10.0
)

# Create Input DataFrame
input_data = pd.DataFrame({
    "income": [income],
    "age": [age],
    "loan_amount": [loan_amount],
    "credit_score": [credit_score],
    "months_employed": [months_employed],
    "num_credit_lines": [num_credit_lines],
    "interest_rate": [interest_rate]
})

# Show Input Data
st.subheader("Entered Information")
st.dataframe(input_data, use_container_width=True)

# Prediction
if st.button("🔍 Predict Default Risk"):

    prediction = model.predict(input_data)[0]

    try:
        probability = model.predict_proba(input_data)[0][1]
    except:
        probability = None

    st.markdown("---")

    if prediction == 1:
        st.error("⚠️ High Risk: Customer is likely to DEFAULT.")
    else:
        st.success("✅ Low Risk: Customer is NOT likely to DEFAULT.")

    if probability is not None:
        st.subheader("Prediction Probability")
        st.progress(float(probability))
        st.write(f"Default Probability: **{probability:.2%}**")

# Footer
st.markdown("---")
st.caption("Developed using Streamlit & Machine Learning")
import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Load Model
# -------------------------------
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Default Prediction")
st.markdown("Predict whether a customer will default on their credit card payment.")

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Customer Information")

customer_id = st.sidebar.number_input(
    "Customer ID",
    min_value=1,
    value=1
)

limit_bal = st.sidebar.number_input(
    "Credit Limit Balance",
    min_value=0,
    value=50000
)

age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

sex = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

education = st.sidebar.selectbox(
    "Education",
    [
        "Graduate School",
        "University",
        "High School",
        "Others"
    ]
)

marriage = st.sidebar.selectbox(
    "Marriage Status",
    [
        "Married",
        "Single",
        "Others"
    ]
)

# -------------------------------
# Repayment Status
# -------------------------------
st.sidebar.subheader("Repayment Status")

pay_0 = st.sidebar.number_input("PAY_0", value=0)
pay_2 = st.sidebar.number_input("PAY_2", value=0)
pay_3 = st.sidebar.number_input("PAY_3", value=0)
pay_4 = st.sidebar.number_input("PAY_4", value=0)
pay_5 = st.sidebar.number_input("PAY_5", value=0)
pay_6 = st.sidebar.number_input("PAY_6", value=0)

# -------------------------------
# Bill Amounts
# -------------------------------
st.sidebar.subheader("Bill Amounts")

bill_amt1 = st.sidebar.number_input("BILL_AMT1", value=0)
bill_amt2 = st.sidebar.number_input("BILL_AMT2", value=0)
bill_amt3 = st.sidebar.number_input("BILL_AMT3", value=0)
bill_amt4 = st.sidebar.number_input("BILL_AMT4", value=0)
bill_amt5 = st.sidebar.number_input("BILL_AMT5", value=0)
bill_amt6 = st.sidebar.number_input("BILL_AMT6", value=0)

# -------------------------------
# Payment Amounts
# -------------------------------
st.sidebar.subheader("Payment Amounts")

pay_amt1 = st.sidebar.number_input("PAY_AMT1", value=0)
pay_amt2 = st.sidebar.number_input("PAY_AMT2", value=0)
pay_amt3 = st.sidebar.number_input("PAY_AMT3", value=0)
pay_amt4 = st.sidebar.number_input("PAY_AMT4", value=0)
pay_amt5 = st.sidebar.number_input("PAY_AMT5", value=0)
pay_amt6 = st.sidebar.number_input("PAY_AMT6", value=0)

# -------------------------------
# Input DataFrame
# -------------------------------
input_data = pd.DataFrame({
    "id": [customer_id],
    "limit_bal": [limit_bal],
    "sex": [sex],
    "education": [education],
    "marriage": [marriage],
    "age": [age],
    "pay_0": [pay_0],
    "pay_2": [pay_2],
    "pay_3": [pay_3],
    "pay_4": [pay_4],
    "pay_5": [pay_5],
    "pay_6": [pay_6],
    "bill_amt1": [bill_amt1],
    "bill_amt2": [bill_amt2],
    "bill_amt3": [bill_amt3],
    "bill_amt4": [bill_amt4],
    "bill_amt5": [bill_amt5],
    "bill_amt6": [bill_amt6],
    "pay_amt1": [pay_amt1],
    "pay_amt2": [pay_amt2],
    "pay_amt3": [pay_amt3],
    "pay_amt4": [pay_amt4],
    "pay_amt5": [pay_amt5],
    "pay_amt6": [pay_amt6]
})

# -------------------------------
# Display Input
# -------------------------------
st.subheader("Input Data")
st.dataframe(input_data)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Default Risk"):

    try:
        prediction = model.predict(input_data)[0]

        try:
            probability = model.predict_proba(input_data)[0][1]
        except:
            probability = None

        if prediction == 1:
            st.error("⚠️ Customer is likely to DEFAULT")
        else:
            st.success("✅ Customer is NOT likely to DEFAULT")

        if probability is not None:
            st.write(f"Default Probability: {probability:.2%}")
            st.progress(float(probability))

    except Exception as e:
        st.error(f"Prediction Error: {e}")

        st.write("Columns:")
        st.write(input_data.columns.tolist())

        st.write("Data Types:")
        st.write(input_data.dtypes)

        st.write("Input Values:")
        st.dataframe(input_data)

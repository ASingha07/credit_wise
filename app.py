# app.py

import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CreditWise - Loan Approval Predictor", page_icon="🏦", layout="centered")


# -----------------------------------------------------------
# Load the trained model and all the preprocessing tools
# -----------------------------------------------------------
def load_file(filename):
    with open("model/" + filename, "rb") as f:
        return pickle.load(f)


try:
    best_model = load_file("best_model.pkl")
    best_model_name = load_file("best_model_name.pkl")
    best_threshold = load_file("best_threshold.pkl")
    number_filler = load_file("number_filler.pkl")
    text_filler = load_file("text_filler.pkl")
    education_encoder = load_file("education_encoder.pkl")
    target_encoder = load_file("target_encoder.pkl")
    onehot_encoder = load_file("onehot_encoder.pkl")
    scaler = load_file("scaler.pkl")
    feature_columns = load_file("feature_columns.pkl")
    number_columns = load_file("number_columns.pkl")
    text_columns = load_file("text_columns.pkl")
    comparison_table = load_file("comparison_table.pkl")
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


# -----------------------------------------------------------
# Title and simple explanation of what this app does
# -----------------------------------------------------------
st.title("🏦 CreditWise: Loan Approval Predictor")

st.write(
    """
    **What is this?**
    This is a simple tool that predicts whether a bank loan application
    would likely be **Approved** or **Rejected**. It was trained on
    past loan records using machine learning, so it recognizes patterns
    from real historical decisions.

    **What should I do?**
    1. Fill in the applicant's details in the form below.
    2. Click the **"Check Loan Approval"** button at the bottom.
    3. See the prediction and how confident the model is.

    """
)

if not model_loaded:
    st.error(
        "No trained model was found yet. Please run 'python train_model.py' "
        "first from the project folder, then reload this app."
    )
    st.stop()

st.divider()


# -----------------------------------------------------------
# The input form (using full, easy to understand field names)
# -----------------------------------------------------------
st.header("Step 1: Enter Applicant Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Applicant's Age (in years)", min_value=20, max_value=100, value=30)

    gender = st.selectbox("Applicant's Gender", ["Male", "Female"])

    marital_status = st.selectbox("Marital Status", ["Married", "Single"])

    dependents = st.number_input("Number of Dependents (people relying on the applicant)", min_value=0, max_value=10, value=0)

    education_level = st.selectbox("Highest Education Level Completed", ["Graduate", "Not Graduate"])

    employment_status = st.selectbox(
        "Current Employment Status",
        ["Salaried", "Self-employed", "Contract", "Unemployed"],
    )

    employer_category = st.selectbox(
        "Type of Employer",
        ["Private", "Government", "MNC", "Business", "Unemployed"],
    )

    property_area = st.selectbox("Area Where the Applicant's Property is Located", ["Urban", "Semiurban", "Rural"])

    loan_purpose = st.selectbox(
        "Purpose of the Loan",
        ["Home", "Education", "Personal", "Business", "Car"],
    )

with col2:
    applicant_income = st.number_input("Applicant's Monthly Income", min_value=15000.0, max_value=300000.0, value=50000.0, step=1000.0)

    coapplicant_income = st.number_input("Co-Applicant's Monthly Income (enter 0 if none)", min_value=0.0, max_value=150000.0, value=0.0, step=1000.0)

    loan_amount = st.number_input("Total Loan Amount Being Requested", min_value=50000.0, max_value=5000000.0, value=500000.0, step=10000.0)

    loan_term = st.selectbox("Loan Repayment Term (in years)", list(range(1, 11)), index=4)

    collateral_value = st.number_input("Value of Collateral Offered (enter 0 if none)", min_value=0.0, max_value=2000000.0, value=0.0, step=10000.0)

    credit_score = st.number_input("Applicant's Credit Score", min_value=300, max_value=900, value=650)

    debt_to_income_percent = st.number_input("Debt-to-Income Ratio (percentage of income already going to debt)", min_value=10.0, max_value=60.0, value=30.0)
    # the model was trained on this as a fraction (0.10 to 0.60), not a
    # percentage (10 to 60), so we convert it here before using it
    debt_to_income_ratio = debt_to_income_percent / 100

    savings = st.number_input("Applicant's Total Savings", min_value=0.0, max_value=1000000.0, value=100000.0, step=5000.0)

    existing_loans = st.number_input("Number of Loans the Applicant Already Has", min_value=0, max_value=5, value=0)


st.header("Step 2: Get the Prediction")
check_button = st.button("Check Loan Approval", use_container_width=True, type="primary")


# -----------------------------------------------------------
# When the button is clicked: prepare the data the same way
# train_model.py did, then ask the model for a prediction
# -----------------------------------------------------------
if check_button:

    # the model was trained using Loan_Term in months, but the form asks
    # for years (easier for a person to understand), so convert it here
    loan_term_months = loan_term * 12

    # put the form answers into one row of a table, just like the training data
    applicant_row = pd.DataFrame([{
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Employment_Status": employment_status,
        "Age": age,
        "Marital_Status": marital_status,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": debt_to_income_ratio,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term_months,
        "Loan_Purpose": loan_purpose,
        "Property_Area": property_area,
        "Education_Level": education_level,
        "Gender": gender,
        "Employer_Category": employer_category,
    }])

    # fill missing values (won't really do anything here since the form
    # always has values, but we use the same tools for consistency)
    applicant_row[number_columns] = number_filler.transform(applicant_row[number_columns])
    applicant_row[text_columns] = text_filler.transform(applicant_row[text_columns])

    # encode Education_Level the same way as training
    applicant_row["Education_Level"] = education_encoder.transform(applicant_row["Education_Level"])

    # one-hot encode the other text columns the same way as training
    onehot_columns = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]
    onehot_result = pd.DataFrame(
        onehot_encoder.transform(applicant_row[onehot_columns]),
        columns=onehot_encoder.get_feature_names_out(onehot_columns),
    )
    applicant_row = pd.concat([applicant_row.drop(columns=onehot_columns), onehot_result], axis=1)

    # make sure the columns are in the exact same order the model expects
    applicant_row = applicant_row.reindex(columns=feature_columns, fill_value=0)

    # scale the numbers the same way as training
    applicant_row_scaled = scaler.transform(applicant_row)

    # ask the model for a prediction, using the threshold that
    # train_model.py picked (not just the default 50%)
    prediction_probabilities = best_model.predict_proba(applicant_row_scaled)[0]
    approval_chance = prediction_probabilities[1] * 100
    rejection_chance = prediction_probabilities[0] * 100
    prediction = 1 if prediction_probabilities[1] >= best_threshold else 0

    st.divider()
    st.header("Result")

    if prediction == 1:
        st.success("This Loan application is likely to be **APPROVED** ✅")
    else:
        st.error("This Loan application is likely to be **REJECTED** ❌")

    result_col1, result_col2 = st.columns(2)
    with result_col1:
        st.metric("Chance of Approval", f"{approval_chance:.1f}%")
    with result_col2:
        st.metric("Chance of Rejection", f"{rejection_chance:.1f}%")


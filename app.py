import pandas as pd
import joblib
import streamlit as st

# Load model and features
model = joblib.load('churn_model_pipeline.pkl')
features = joblib.load('model_features.pkl')

st.set_page_config(page_title='Customer Churn Predictor', layout='centered')

st.title('Customer Churn Prediction App')
st.write('Predict whether a customer is likely to churn')

# ---- User Inputs
monthly_charges = st.number_input('Monthly Charges', min_value=0.0)
tenure = st.number_input('Tenure (months)', min_value=0)

contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
tech_support = st.selectbox('Tech Support', ['Yes', 'No'])

# ---- Input Data
input_dict = {
    'MonthlyCharges': monthly_charges,
    'tenure': tenure,
    'Contract_One_year': 1 if contract == 'One year' else 0,
    'Contract_two_year': 1 if contract == 'Two year' else 0,
    'InternetService_Fiber optic': 1 if internet_service == 'Fiber optic' else 0,
    'TechSupport_Yes': 1 if tech_support == 'Yes' else 0
}

# Create DataFrame
input_df = pd.DataFrame([input_dict])

# Add missing columns
for col in features:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns
input_df = input_df[features]

# ---- Prediction
if st.button('Predict Churn'):
    prob = model.predict_proba(input_df)[0][1]
    prediction = 'High Risk' if prob > 0.5 else 'Low Risk'

    st.subheader('Prediction Result')
    st.write(f'**Churn Probability:** {prob:.2f}')
    st.write(f'**Churn Risk:** {prediction}')

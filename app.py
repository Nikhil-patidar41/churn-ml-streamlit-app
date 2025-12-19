import pandas as pd
import joblib
import streamlit as st



#model load
model=joblib.load('churn_model_pipeline.pkl')
features=joblib.load('model_features.pkl')

st.set_page_config(page_title='Customer Churn Predictor',layout='centered')

st.title('Customer Churn Prediction App')
st.write('predict whether a customer is likely to churn')


#--user input
monthly_charges =st.number_input ('Monthly Charges',min_value=0.0)
tenure=st.number_input('tenure (months)',min_value=0)

contract =st.selectbox('Contract Type',['Month-to-month','One year','Two year'])
internet_service=st.selectbox('Internet Services ',['DSL','Fiber optic','No'])
tec_support=st.selectbox('Tec Support',['Yes','No'])

#input data
input_dict={
    'MonthlyCharges':monthly_charges,
    'tenure':tenure,
    'Contract_One_year':1 if contract =='One year' else 0,
    'Contract_two_year':1 if contract =='Two year',else 0,
    'InternetService_Fiber optic':1 if internet_service == 'Fiber optic' else 0,
    'TechSupport_Yes':1 if tech_support == 'Yes' else 0

}


# Create input dataframe
input_df = pd.DataFrame([input_dict])

# add missing columns
for col in features:
    if col not in input_df.columns:
        input_df[col]=0

input_df=input_df[features]

#prediction
if st.button('Predict Churn'):
    prob=model.predict_proba(input_df)[0][1]
    prediction ='High Risk' if prob > 0.5  else 'Low Risk'

    st.subheader('Prediction Result')
    st.write(f'**Churn Probability :**{prob:.2f}')
    st.write(f"**Churn Risk:** {prediction}")



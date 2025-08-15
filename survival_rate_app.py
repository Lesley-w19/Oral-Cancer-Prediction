import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor


# Load data
oral_cancer_df = pd.read_csv('oral_cancer_prediction_dataset.csv') 

# Preprocessing: Label Encoding
label_encoder = LabelEncoder()
for col in ['treatment_type', 'early_diagnosis', 'diagnosis']:
    oral_cancer_df[col] = label_encoder.fit_transform(oral_cancer_df[col])

# Define features and target
features = ['tumor_size', 'cancer_stage', 'treatment_type', 'cost_of_treatment', 'economic_burden', 'early_diagnosis', 'diagnosis']
target = 'survival_rate'


# Train the Gradient Boosting Regressor model
gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbr_model.fit(oral_cancer_df[features], oral_cancer_df[target])


# # Streamlit App
st.title('Oral Cancer Survival Rate Prediction')

# Input form
tumor_size = st.number_input('Tumor Size', min_value=0.0, max_value=5.0, value=0.0)
cancer_stage = st.slider('Cancer Stage', min_value=0, max_value=3, value=0)
treatment_type = st.selectbox('Treatment Type', options=['No Treatment', 'Surgery'])
cost_of_treatment = st.number_input('Cost of Treatment', min_value=0.0, value=0.0)
economic_burden = st.number_input('Economic Burden', min_value=0, value=0)
early_diagnosis = st.selectbox('Early Diagnosis', options=['No', 'Yes'])
diagnosis = st.selectbox('Diagnosis', options=['No', 'Yes'])

# Preprocess input
input_data = pd.DataFrame({
    'tumor_size': [tumor_size],
    'cancer_stage': [cancer_stage],
    'treatment_type': [treatment_type],
    'cost_of_treatment': [cost_of_treatment],
    'economic_burden': [economic_burden],
    'early_diagnosis': [early_diagnosis],
    'diagnosis': [diagnosis]
})

# Encode categorical features
for col in ['treatment_type','early_diagnosis', 'diagnosis']:
    input_data[col] = label_encoder.fit_transform(input_data[col])

# Make prediction
if st.button('Predict Survival Rate'):
    prediction = gbr_model.predict(input_data[features])[0]
    st.success(f'Predicted Survival Rate: {prediction:.2f}%')

# =========================================================
# Customer Churn Prediction Web App
# Built using Streamlit + TensorFlow ANN Model
# =========================================================

# Import Required Libraries
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder


# =========================================================
# Load Trained ANN Model
# =========================================================
model = tf.keras.models.load_model("model.h5")

# =========================================================
# Load Encoders and Scaler
# =========================================================
with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("onehot_encoding_geo.pkl", "rb") as file:
    onehot_encoding_geo = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# =========================================================
# Streamlit App Configuration
# =========================================================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction")
st.markdown(
    "Predict whether a bank customer is likely to churn using an Artificial Neural Network (ANN)."
)

# =========================================================
# User Input Section
# =========================================================
st.subheader("Enter Customer Details")

geography = st.selectbox(
    "Geography",
    onehot_encoding_geo.categories_[0]
)

gender = st.selectbox(
    "Gender",
    label_encoder_gender.classes_
)

age = st.slider("Age", 18, 92)
balance = st.number_input("Balance", min_value=0.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0)
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)

has_cr_card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)

is_active_member = st.selectbox(
    "Is Active Member",
    [0, 1]
)

# =========================================================
# Prepare Input Data
# =========================================================
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})

# One-Hot Encode Geography Feature
geo_encoded = onehot_encoding_geo.transform([[geography]]).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoding_geo.get_feature_names_out()
)

# Combine Encoded Geography Data
input_data = pd.concat(
    [input_data.reset_index(drop=True), geo_encoded_df],
    axis=1
)

# Scale Input Features
input_data_scaled = scaler.transform(input_data)

# =========================================================
# Make Prediction
# =========================================================
prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]

# =========================================================
# Display Prediction Result
# =========================================================
st.subheader("Prediction Result")

st.write(f"Churn Probability: **{prediction_probability:.2f}**")

if prediction_probability > 0.5:
    st.error("⚠️ Customer is likely to churn.")
else:
    st.success("✅ Customer is not likely to churn.")

# =========================================================
# Footer
# =========================================================
st.markdown("---")
st.caption("Research & Development in AI/ML Portfolio.")

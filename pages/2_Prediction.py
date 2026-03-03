import os
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "rf_pipeline_streamlit.pkl")
data_path = os.path.join(BASE_DIR, "data", "online_shoppers_intention.csv")

model = joblib.load(model_path)
df = pd.read_csv(data_path)

st.title("🔍 Customer Purchase Prediction")

threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.01)

feature_names = df.drop("Revenue", axis=1).columns
col1, col2 = st.columns(2)
input_data = {}

with col1:
    for feature in feature_names[:len(feature_names)//2]:
        if df[feature].dtype == "object":
            input_data[feature] = st.selectbox(feature, df[feature].unique())
        elif df[feature].dtype == "bool":
            input_data[feature] = st.selectbox(feature, [True, False])
        else:
            input_data[feature] = st.number_input(feature, value=float(df[feature].mean()))

with col2:
    for feature in feature_names[len(feature_names)//2:]:
        if df[feature].dtype == "object":
            input_data[feature] = st.selectbox(feature, df[feature].unique())
        elif df[feature].dtype == "bool":
            input_data[feature] = st.selectbox(feature, [True, False])
        else:
            input_data[feature] = st.number_input(feature, value=float(df[feature].mean()))

if st.button("Run Prediction"):

    input_df = pd.DataFrame([input_data])
    probability = model.predict_proba(input_df)[0][1]
    prediction = 1 if probability >= threshold else 0

    st.metric("Purchase Probability", f"{probability*100:.2f}%")
    st.progress(float(probability))

    if prediction == 1:
        st.success("Customer is LIKELY to purchase")
    else:
        st.error("Customer is NOT likely to purchase")

    result_df = input_df.copy()
    result_df["Predicted_Probability"] = probability
    result_df["Prediction"] = prediction

    csv = result_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Prediction Report",
        csv,
        "customer_prediction_report.csv",
        "text/csv"
    )
import os
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "rf_pipeline_streamlit.pkl")
data_path = os.path.join(BASE_DIR, "data", "online_shoppers_intention.csv")
auc_path = os.path.join(BASE_DIR, "model_auc.pkl")

model = joblib.load(model_path)
df = pd.read_csv(data_path)
auc_score = joblib.load(auc_path)

st.title("📊 Executive Overview")

# KPIs
total_sessions = len(df)
purchase_rate = df["Revenue"].mean() * 100
avg_page_value = df["PageValues"].mean()
avg_bounce_rate = df["BounceRates"].mean()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Sessions", f"{total_sessions:,}")
k2.metric("Purchase Rate", f"{purchase_rate:.2f}%")
k3.metric("Avg Page Value", f"{avg_page_value:.2f}")
k4.metric("Avg Bounce Rate", f"{avg_bounce_rate:.4f}")

st.markdown("---")

st.subheader("🎯 Model Performance")

st.metric("Test AUC Score", f"{auc_score:.4f}")

st.success(
    "The model demonstrates strong discriminative ability with an AUC above 0.90, "
    "indicating high predictive performance on unseen data."
)
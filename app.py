import streamlit as st
import os
import joblib
import pandas as pd

st.set_page_config(
    page_title="Customer Behavior Prediction",
    page_icon="📊",
    layout="wide"
)

# ----------------------------
# Load Model + Dataset
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "rf_pipeline_streamlit.pkl")
data_path = os.path.join(BASE_DIR, "data", "online_shoppers_intention.csv")

model = joblib.load(model_path)
df = pd.read_csv(data_path)

# ----------------------------
# Header
# ----------------------------
st.markdown(
"""
<h1 style='text-align:center;color:#1F618D;'>
Customer Behavior Prediction System
</h1>
""",
unsafe_allow_html=True
)

st.markdown("---")

# ----------------------------
# System Status Panel
# ----------------------------
st.subheader("🚀 System Status")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Model Status", "Active")
col2.metric("Model Type", "Random Forest")
col3.metric("Dataset Size", len(df))
col4.metric("Prediction Mode", "Real-time")

st.markdown("---")

# ----------------------------
# Description
# ----------------------------
st.markdown(
"""
### Welcome

This dashboard demonstrates a **machine learning system for predicting
customer purchase behavior** based on website interaction patterns.

The system includes:

• Real-time customer prediction  
• Batch dataset prediction  
• Model interpretability (feature importance)  
• Model performance metrics  
• Downloadable prediction reports

Use the navigation menu on the left to explore:

📊 Overview  
🔍 Prediction Engine  
📈 Model Insights  
📂 Batch Prediction
"""
)

st.markdown("---")

st.markdown(
"<center><b>Developed by Mohammed Grema Alkali & Bashir Umar Zanna</b></center>",
unsafe_allow_html=True
)
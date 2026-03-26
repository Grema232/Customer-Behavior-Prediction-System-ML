import os
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(layout="wide")

# ----------------------------
# Load Resources
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "rf_pipeline_streamlit.pkl")
data_path = os.path.join(BASE_DIR, "data", "online_shoppers_intention.csv")
auc_path = os.path.join(BASE_DIR, "model_auc.pkl")

model = joblib.load(model_path)
df = pd.read_csv(data_path)
auc_score = joblib.load(auc_path)

# ----------------------------
# Title
# ----------------------------
st.title("📊 Executive Overview")

# ----------------------------
# Business Objective
# ----------------------------
st.markdown("""
### 💼 Business Objective

This system predicts whether an online customer session will result in a purchase.

It enables e-commerce platforms to:
- Identify high-value customers in real time  
- Optimize targeted marketing strategies  
- Reduce bounce-related losses  
- Improve overall conversion rates  

""")

st.markdown("---")

# ----------------------------
# Key Performance Indicators
# ----------------------------
st.subheader("📈 Key Dataset Insights")

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

# ----------------------------
# Model Performance
# ----------------------------
st.subheader("🎯 Model Performance")

col1, col2 = st.columns([1, 2])

with col1:
    st.metric("Test AUC Score", f"{auc_score:.4f}")

with col2:
    st.info(
        "AUC (Area Under the ROC Curve) measures the model’s ability to distinguish "
        "between purchasing and non-purchasing customers across all thresholds."
    )

st.success(
    "The model demonstrates strong predictive performance with an AUC above 0.90, "
    "indicating excellent capability in identifying high-intent customers."
)

st.markdown("---")

# ----------------------------
# Model Explanation
# ----------------------------
st.subheader("🧠 Model Overview")

st.markdown("""
- **Model Used:** Random Forest Classifier  
- **Training Strategy:** Stratified train-test split  
- **Feature Processing:** One-hot encoding for categorical variables  
- **Class Handling:** Balanced class weights to address class imbalance  

""")

# ----------------------------
# Why This Matters
# ----------------------------
st.subheader("🚀 Business Impact")

st.markdown("""
This system can be used to:

- Prioritize high-probability customers for targeted promotions  
- Reduce marketing spend on low-conversion traffic  
- Support real-time decision-making in e-commerce platforms  
- Improve ROI through data-driven customer engagement strategies  

""")
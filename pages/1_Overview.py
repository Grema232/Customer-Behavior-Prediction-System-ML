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
# Executive KPI Dashboard
# ----------------------------
st.subheader("🚀 Executive Dashboard")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Best Model", "Random Forest")
col2.metric("Accuracy", "0.896")
col3.metric("AUC Score", f"{auc_score:.3f}")
col4.metric("Dataset Size", f"{len(df):,}")

st.markdown("---")

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
# Key Dataset Insights
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
# Top Predictive Features
# ----------------------------
st.subheader("🔍 Top Predictive Factors")

col1, col2 = st.columns(2)

with col1:
    st.success("""
    High Purchase Indicators
    
    • High Page Values  
    • Low Bounce Rates  
    • Long Product Duration  
    • High Product Interaction  
    """)

with col2:
    st.warning("""
    Low Purchase Indicators
    
    • High Exit Rates  
    • High Bounce Rates  
    • Short Session Duration  
    • Low Page Value  
    """)

st.markdown("---")

# ----------------------------
# Business Decision Panel
# ----------------------------
st.subheader("📌 Business Decision Panel")

c1, c2, c3 = st.columns(3)

with c1:
    st.info("""
    🎯 Target Customers
    
    Focus marketing on:
    • High page value users  
    • Returning visitors  
    • Long sessions  
    """)

with c2:
    st.warning("""
    ⚠️ Risk Customers
    
    Users with:
    • High bounce rate  
    • High exit rate  
    • Short visits  
    """)

with c3:
    st.success("""
    💰 Revenue Opportunity
    
    Prioritize:
    • Product related visitors  
    • Engaged users  
    • High intent sessions  
    """)

st.markdown("---")

# ----------------------------
# Insight Interpretation
# ----------------------------
st.subheader("📊 Insight Interpretation")

if purchase_rate < 15:
    st.warning("Low conversion rate → focus on improving user engagement and reducing bounce.")
elif purchase_rate < 30:
    st.info("Moderate conversion → targeted marketing strategies can improve performance.")
else:
    st.success("Strong conversion → prioritize high-value customer targeting.")

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
        "AUC measures the model’s ability to distinguish "
        "between purchasing and non-purchasing customers."
    )

st.success(
    "The model demonstrates strong predictive performance with AUC above 0.90."
)

st.markdown("---")

# ----------------------------
# Model Overview
# ----------------------------
st.subheader("🧠 Model Overview")

st.markdown("""
- Model Used: Random Forest Classifier  
- Training Strategy: Stratified train-test split  
- Feature Processing: One-hot encoding  
- Class Handling: Balanced class weights  
""")

st.markdown("---")

# ----------------------------
# Business Recommendation
# ----------------------------
st.subheader("💡 Business Recommendation")

st.success("""
Target customers with:

• High Page Values  
• Low Bounce Rates  
• Longer Product Interaction Time  

These users show strongest purchase intent.
""")

st.markdown("---")

# ----------------------------
# Business Impact
# ----------------------------
st.subheader("🚀 Business Impact")

st.markdown("""
This system enables:

- Prioritization of high-probability customers  
- Reduction of marketing inefficiencies  
- Real-time decision-making support  
- Improved return on investment (ROI)  
""")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")

st.caption(
"Customer Behavior Prediction System | "
"Random Forest Model | "
"Streamlit ML Dashboard | "
"Developed by Mohammed Grema Alkali & Bashir Umar Zanna"
)
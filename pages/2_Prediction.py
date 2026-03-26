import os
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(layout="wide")

# ----------------------------
# Load Model and Dataset
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "rf_pipeline_streamlit.pkl")
data_path = os.path.join(BASE_DIR, "data", "online_shoppers_intention.csv")

model = joblib.load(model_path)
df = pd.read_csv(data_path)

st.title("🔍 Customer Purchase Prediction Engine")

# ----------------------------
# HOW TO USE (VERY IMPORTANT)
# ----------------------------
st.info("""
**How to use this tool:**
1. Enter customer session data below  
2. Adjust decision threshold (optional)  
3. Click **Run Prediction**  
4. Interpret probability and decision results  
""")

# ----------------------------
# Business Context
# ----------------------------
st.markdown("""
### 💼 Business Context

This tool predicts whether a customer session will result in a purchase.  
It supports real-time decision-making in e-commerce platforms.

""")

# ----------------------------
# Sidebar Controls
# ----------------------------
threshold = st.sidebar.slider(
    "Decision Threshold",
    0.0,
    1.0,
    0.5,
    0.01
)

st.sidebar.markdown(f"""
**Current Threshold:** {threshold:.2f}

- Lower → more customers classified as buyers  
- Higher → stricter prediction  
""")

# ----------------------------
# Input Section
# ----------------------------
st.markdown("### 🧾 Customer Input Data")

feature_names = df.drop("Revenue", axis=1).columns

col1, col2 = st.columns(2)

input_data = {}

def clean_label(name):
    return name.replace("_", " ")

with col1:
    for feature in feature_names[:len(feature_names)//2]:
        label = clean_label(feature)

        if df[feature].dtype == "object":
            input_data[feature] = st.selectbox(label, df[feature].unique())

        elif df[feature].dtype == "bool":
            input_data[feature] = st.selectbox(label, [True, False])

        else:
            input_data[feature] = st.number_input(
                label,
                value=float(df[feature].mean())
            )

with col2:
    for feature in feature_names[len(feature_names)//2:]:
        label = clean_label(feature)

        if df[feature].dtype == "object":
            input_data[feature] = st.selectbox(label, df[feature].unique())

        elif df[feature].dtype == "bool":
            input_data[feature] = st.selectbox(label, [True, False])

        else:
            input_data[feature] = st.number_input(
                label,
                value=float(df[feature].mean())
            )

# ----------------------------
# Run Prediction
# ----------------------------
if st.button("🚀 Run Prediction"):

    input_df = pd.DataFrame([input_data])

    probabilities = model.predict_proba(input_df)[0]

    prob_not_purchase = probabilities[0]
    prob_purchase = probabilities[1]

    prediction = 1 if prob_purchase >= threshold else 0

    st.markdown("---")
    st.subheader("📊 Prediction Results")

    # ----------------------------
    # Probability Metrics
    # ----------------------------
    m1, m2 = st.columns(2)

    m1.metric("Purchase Probability", f"{prob_purchase*100:.2f}%")
    m2.metric("No Purchase Probability", f"{prob_not_purchase*100:.2f}%")

    # ----------------------------
    # Comparison Chart
    # ----------------------------
    st.markdown("### 📈 Probability Comparison")

    comparison_df = pd.DataFrame({
        "Class": ["No Purchase", "Purchase"],
        "Probability": [prob_not_purchase, prob_purchase]
    })

    st.bar_chart(comparison_df.set_index("Class"))

    st.markdown("---")

    # ----------------------------
    # Decision Output
    # ----------------------------
    if prediction == 1:
        st.success("🟢 Customer is LIKELY to purchase")
        prediction_label = "Likely Purchase"
    else:
        st.error("🔴 Customer is NOT likely to purchase")
        prediction_label = "No Purchase"

    st.caption(f"Decision based on threshold = {threshold:.2f}")

    # ----------------------------
    # Confidence Interpretation
    # ----------------------------
    st.markdown("### 🎯 Model Confidence Interpretation")

    if prob_purchase >= 0.80:
        level = "High Confidence"
        msg = "Strong behavioral signals indicate high likelihood of purchase."
    elif prob_purchase >= 0.60:
        level = "Moderate Confidence"
        msg = "Customer shows meaningful purchase intent."
    elif prob_purchase >= 0.40:
        level = "Uncertain"
        msg = "Mixed behavioral signals detected."
    else:
        level = "Low Confidence"
        msg = "Weak purchase intent indicators."

    st.info(f"**{level}:** {msg}")

    # ----------------------------
    # Gauge
    # ----------------------------
    st.markdown("### 📊 Purchase Probability Gauge")

    st.progress(float(prob_purchase))
    st.write(f"Confidence Score: **{prob_purchase*100:.2f}%**")

    # ----------------------------
    # Download Report
    # ----------------------------
    result_df = input_df.copy()

    result_df["Probability_Purchase"] = prob_purchase
    result_df["Probability_No_Purchase"] = prob_not_purchase
    result_df["Prediction_Label"] = prediction_label
    result_df["Threshold"] = threshold

    csv = result_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Prediction Report",
        csv,
        "customer_prediction_report.csv",
        "text/csv"
    )
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

st.title("🔍 Customer Purchase Prediction Engine")

# Sidebar Controls
threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.01)

st.markdown("### 🧾 Customer Input Data")

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
            input_data[feature] = st.number_input(
                feature,
                value=float(df[feature].mean())
            )

with col2:
    for feature in feature_names[len(feature_names)//2:]:
        if df[feature].dtype == "object":
            input_data[feature] = st.selectbox(feature, df[feature].unique())
        elif df[feature].dtype == "bool":
            input_data[feature] = st.selectbox(feature, [True, False])
        else:
            input_data[feature] = st.number_input(
                feature,
                value=float(df[feature].mean())
            )

if st.button("Run Prediction"):

    input_df = pd.DataFrame([input_data])

    probabilities = model.predict_proba(input_df)[0]
    prob_not_purchase = probabilities[0]
    prob_purchase = probabilities[1]

    prediction = 1 if prob_purchase >= threshold else 0

    st.markdown("---")
    st.subheader("📊 Prediction Results")

    # Dual Probability Metrics
    m1, m2 = st.columns(2)
    m1.metric("Probability of Purchase", f"{prob_purchase*100:.2f}%")
    m2.metric("Probability of No Purchase", f"{prob_not_purchase*100:.2f}%")

    st.markdown("### 📈 Probability Comparison")

    comparison_df = pd.DataFrame({
        "Class": ["No Purchase", "Purchase"],
        "Probability": [prob_not_purchase, prob_purchase]
    })

    st.bar_chart(comparison_df.set_index("Class"))

    st.markdown("---")

    # Decision Indicator
    if prediction == 1:
        st.success("🟢 Decision: Customer is LIKELY to purchase")
    else:
        st.error("🔴 Decision: Customer is NOT likely to purchase")

    # Confidence Interpretation
    st.markdown("### 🎯 Model Confidence Interpretation")

    if prob_purchase >= 0.80:
        st.info("High confidence prediction. Strong behavioral purchase signals detected.")
    elif prob_purchase >= 0.60:
        st.info("Moderate confidence prediction. Customer shows meaningful purchase intent.")
    elif prob_purchase >= 0.40:
        st.info("Uncertain prediction. Behavioral signals are mixed.")
    else:
        st.info("Low confidence prediction. Weak purchase intent indicators.")

    # Downloadable Report
    result_df = input_df.copy()
    result_df["Probability_Purchase"] = prob_purchase
    result_df["Probability_No_Purchase"] = prob_not_purchase
    result_df["Final_Prediction"] = prediction

    csv = result_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Prediction Report",
        csv,
        "customer_prediction_report.csv",
        "text/csv"
    )
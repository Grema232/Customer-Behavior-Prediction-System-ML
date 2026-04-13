import os
import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

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
# Sidebar Threshold
# ----------------------------
threshold = st.sidebar.slider(
    "Decision Threshold",
    0.0, 1.0, 0.5, 0.01
)

# ----------------------------
# Input Section
# ----------------------------
st.markdown("### 🧾 Customer Input Data")

feature_names = df.drop("Revenue", axis=1).columns
input_data = {}

col1, col2 = st.columns(2)

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

    c1, c2 = st.columns(2)
    c1.metric("Purchase Probability", f"{prob_purchase*100:.2f}%")
    c2.metric("No Purchase Probability", f"{prob_not_purchase*100:.2f}%")

    # ----------------------------
    # Probability Chart
    # ----------------------------
    st.markdown("### 📈 Probability Comparison")

    chart_df = pd.DataFrame({
        "Class": ["No Purchase", "Purchase"],
        "Probability": [prob_not_purchase, prob_purchase]
    })

    st.bar_chart(chart_df.set_index("Class"))

    st.markdown("---")

    # ----------------------------
    # Decision Label
    # ----------------------------
    if prediction == 1:
        st.success("🟢 Customer is LIKELY to purchase")
        prediction_label = "Likely Purchase"
    else:
        st.error("🔴 Customer is NOT likely to purchase")
        prediction_label = "No Purchase"

    st.caption(f"Decision threshold: {threshold:.2f}")

    # ----------------------------
    # Confidence Gauge
    # ----------------------------
    st.markdown("### 🎯 Prediction Confidence")

    confidence = prob_purchase
    st.progress(float(confidence))
    st.write(f"Confidence Score: **{confidence*100:.2f}%**")

    if confidence > 0.80:
        st.success("High confidence prediction")
    elif confidence > 0.60:
        st.info("Moderate confidence prediction")
    elif confidence > 0.40:
        st.warning("Uncertain prediction")
    else:
        st.error("Low confidence prediction")

    st.markdown("---")

    # ----------------------------
    # SHAP Explainability
    # ----------------------------
    st.subheader("🧠 Model Explainability (SHAP)")

    try:
        explainer = shap.TreeExplainer(model.named_steps["classifier"])
        transformed = model.named_steps["preprocessor"].transform(input_df)

        shap_values = explainer.shap_values(transformed)

        st.markdown("### 🔎 Feature Impact")

        fig = plt.figure()
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[1][0],
                base_values=explainer.expected_value[1],
                data=transformed[0]
            ),
            show=False
        )

        st.pyplot(fig)

    except:
        st.info("SHAP explanation available for tree-based models")

    st.markdown("---")

    # ----------------------------
    # Download Report
    # ----------------------------
    result_df = input_df.copy()
    result_df["Prediction"] = prediction_label
    result_df["Probability_Purchase"] = prob_purchase
    result_df["Probability_No_Purchase"] = prob_not_purchase
    result_df["Threshold"] = threshold

    csv = result_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Prediction Report",
        csv,
        "customer_prediction_report.csv",
        "text/csv"
    )
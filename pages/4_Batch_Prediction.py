import os
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(layout="wide")

# ----------------------------
# Load Model
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "rf_pipeline_streamlit.pkl")

model = joblib.load(model_path)

st.title("📂 Batch Customer Prediction")

# ----------------------------
# HOW TO USE
# ----------------------------
st.info("""
**How to use:**
1. Upload a dataset (CSV or Excel)
2. Ensure it matches the required structure
3. System validates data automatically
4. Predictions and insights will be generated
""")

# ----------------------------
# Business Context
# ----------------------------
st.markdown("""
### 💼 Business Context

This module enables bulk prediction of customer purchase behavior.

Use cases:
- Campaign targeting  
- Customer segmentation  
- Conversion optimization  
""")

# ----------------------------
# Upload Dataset
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload dataset",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    # ----------------------------
    # Read Dataset
    # ----------------------------
    try:
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()

    st.subheader("📊 Uploaded Dataset Preview")
    st.dataframe(data.head())

    # ----------------------------
    # Expected Features
    # ----------------------------
    expected_features = [
        "Administrative",
        "Administrative_Duration",
        "Informational",
        "Informational_Duration",
        "ProductRelated",
        "ProductRelated_Duration",
        "BounceRates",
        "ExitRates",
        "PageValues",
        "SpecialDay",
        "Month",
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType",
        "VisitorType",
        "Weekend"
    ]

    # ----------------------------
    # VALIDATION SYSTEM
    # ----------------------------
    uploaded_columns = list(data.columns)

    missing_cols = [col for col in expected_features if col not in uploaded_columns]
    extra_cols = [col for col in uploaded_columns if col not in expected_features]

    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
        st.stop()

    if extra_cols:
        st.warning(f"⚠ Extra columns detected and removed: {extra_cols}")
        data = data[expected_features]

    # Ensure correct order
    data = data[expected_features]

    st.success("✅ Dataset validated successfully. Ready for prediction.")

    # ----------------------------
    # Run Predictions
    # ----------------------------
    try:
        predictions = model.predict(data)
        probabilities = model.predict_proba(data)[:, 1]
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        st.stop()

    # ----------------------------
    # Results Table
    # ----------------------------
    results = data.copy()
    results["Purchase_Probability"] = probabilities
    results["Prediction"] = predictions

    results["Prediction"] = results["Prediction"].map({
        1: "Likely Purchase",
        0: "No Purchase"
    })

    # Sort
    results = results.sort_values(
        by="Purchase_Probability",
        ascending=False
    )

    # ----------------------------
    # SUMMARY METRICS
    # ----------------------------
    st.subheader("📊 Prediction Summary")

    total_customers = len(results)
    likely_purchase = (results["Prediction"] == "Likely Purchase").sum()
    not_purchase = total_customers - likely_purchase
    avg_probability = results["Purchase_Probability"].mean()

    # NEW METRIC (important)
    high_intent = (results["Purchase_Probability"] >= 0.7).sum()

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Customers", total_customers)
    col2.metric("Likely Buyers", likely_purchase)
    col3.metric("Non-Buyers", not_purchase)
    col4.metric("Avg Probability", f"{avg_probability:.2%}")
    col5.metric("High Intent Customers", high_intent)

    # ----------------------------
    # Business Interpretation
    # ----------------------------
    st.markdown("### 💡 Business Interpretation")

    if likely_purchase / total_customers < 0.15:
        st.warning("Low conversion potential → Improve engagement and UX.")
    elif likely_purchase / total_customers < 0.30:
        st.info("Moderate conversion → Apply targeted campaigns.")
    else:
        st.success("High conversion potential → Focus on high-value customers.")

    # ----------------------------
    # Chart
    # ----------------------------
    chart_data = pd.DataFrame({
        "Category": ["Likely Purchase", "No Purchase"],
        "Customers": [likely_purchase, not_purchase]
    })

    st.bar_chart(chart_data.set_index("Category"))

    # ----------------------------
    # Results Table
    # ----------------------------
    st.subheader("📈 Prediction Results (Sorted by Probability)")
    st.dataframe(results)

    # ----------------------------
    # Download
    # ----------------------------
    csv = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Predictions",
        csv,
        "batch_predictions.csv",
        "text/csv"
    )
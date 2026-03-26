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

st.markdown("""
Upload a dataset to generate purchase predictions for multiple customers.

Supported formats:
- CSV
- Excel (.xlsx)
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
    # VALIDATION SYSTEM (PRO)
    # ----------------------------
    uploaded_columns = list(data.columns)

    missing_cols = [col for col in expected_features if col not in uploaded_columns]
    extra_cols = [col for col in uploaded_columns if col not in expected_features]

    # ❌ Missing columns → STOP
    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
        st.stop()

    # ⚠ Extra columns → warn + drop
    if extra_cols:
        st.warning(f"⚠ Extra columns detected and removed: {extra_cols}")
        data = data[expected_features]

    # Ensure correct column order
    data = data[expected_features]

    st.success("✅ Dataset structure validated successfully.")

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
    # Create Results Table
    # ----------------------------
    results = data.copy()
    results["Purchase_Probability"] = probabilities
    results["Prediction"] = predictions

    # Convert predictions to readable labels
    results["Prediction"] = results["Prediction"].map({
        1: "Likely Purchase",
        0: "No Purchase"
    })

    # ----------------------------
    # Sort Customers by Probability
    # ----------------------------
    results = results.sort_values(
        by="Purchase_Probability",
        ascending=False
    )

    # ----------------------------
    # Prediction Summary Dashboard
    # ----------------------------
    st.subheader("📊 Prediction Summary")

    total_customers = len(results)
    likely_purchase = (results["Prediction"] == "Likely Purchase").sum()
    not_purchase = total_customers - likely_purchase
    avg_probability = results["Purchase_Probability"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", total_customers)
    col2.metric("Likely to Purchase", likely_purchase)
    col3.metric("Not Likely to Purchase", not_purchase)
    col4.metric("Avg Purchase Probability", f"{avg_probability:.2%}")

    # ----------------------------
    # Chart Visualization
    # ----------------------------
    chart_data = pd.DataFrame({
        "Category": ["Likely Purchase", "Not Purchase"],
        "Customers": [likely_purchase, not_purchase]
    })

    st.bar_chart(chart_data.set_index("Category"))

    # ----------------------------
    # Prediction Results Table
    # ----------------------------
    st.subheader("📈 Prediction Results (Sorted by Purchase Probability)")
    st.dataframe(results)

    # ----------------------------
    # Download Predictions
    # ----------------------------
    csv = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Predictions",
        csv,
        "batch_predictions.csv",
        "text/csv"
    )
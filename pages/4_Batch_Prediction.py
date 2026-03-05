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
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

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
    # Validate Dataset Structure
    # ----------------------------
    missing = [col for col in expected_features if col not in data.columns]

    if missing:
        st.error(f"Dataset missing required columns: {missing}")
    else:

        st.success("Dataset structure validated successfully.")

        # ----------------------------
        # Run Predictions
        # ----------------------------
        predictions = model.predict(data)
        probabilities = model.predict_proba(data)[:, 1]

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
        # Prediction Summary Dashboard
        # ----------------------------
        st.subheader("📊 Prediction Summary")

        total_customers = len(results)
        likely_purchase = (results["Prediction"] == "Likely Purchase").sum()
        not_purchase = total_customers - likely_purchase

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Customers", total_customers)
        col2.metric("Likely to Purchase", likely_purchase)
        col3.metric("Not Likely to Purchase", not_purchase)

        # Chart
        chart_data = pd.DataFrame({
            "Category": ["Likely Purchase", "Not Purchase"],
            "Customers": [likely_purchase, not_purchase]
        })

        st.bar_chart(chart_data.set_index("Category"))

        # ----------------------------
        # Prediction Results Table
        # ----------------------------
        st.subheader("📈 Prediction Results")
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
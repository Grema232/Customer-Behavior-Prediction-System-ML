import os
import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

st.title("SHAP Feature Importance")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "rf_pipeline_streamlit.pkl")
data_path = os.path.join(BASE_DIR, "data", "online_shoppers_intention.csv")

model = joblib.load(model_path)
df = pd.read_csv(data_path)

X = df.drop("Revenue", axis=1)

st.info("Global feature importance using SHAP values")

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

X_transformed = preprocessor.transform(X)

explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(X_transformed)

fig = plt.figure()

shap.summary_plot(
    shap_values,
    X_transformed,
    show=False
)

st.pyplot(fig)
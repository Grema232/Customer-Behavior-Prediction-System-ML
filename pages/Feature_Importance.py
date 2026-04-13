import streamlit as st
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt

st.title("Feature Importance")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "rf_pipeline_streamlit.pkl")
data_path = os.path.join(BASE_DIR, "data", "online_shoppers_intention.csv")

model = joblib.load(model_path)
df = pd.read_csv(data_path)

X = df.drop("Revenue", axis=1)

# get feature names after preprocessing
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

feature_names = preprocessor.get_feature_names_out()

importance = classifier.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

st.subheader("Top Important Features")
st.dataframe(importance_df.head(10))

fig, ax = plt.subplots(figsize=(8,6))
ax.barh(importance_df["Feature"].head(10), importance_df["Importance"].head(10))
ax.invert_yaxis()

st.pyplot(fig)
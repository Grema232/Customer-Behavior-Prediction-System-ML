import streamlit as st
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt

st.title("Prediction Probability Distribution")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "rf_pipeline_streamlit.pkl")
data_path = os.path.join(BASE_DIR, "data", "online_shoppers_intention.csv")

model = joblib.load(model_path)
df = pd.read_csv(data_path)

X = df.drop("Revenue", axis=1)

probs = model.predict_proba(X)[:,1]

fig, ax = plt.subplots()
ax.hist(probs, bins=30)

st.pyplot(fig)

st.info("Distribution shows likelihood of customer purchase probability")
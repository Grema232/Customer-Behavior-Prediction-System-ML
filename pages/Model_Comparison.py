import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

st.title("📊 Model Comparison")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(BASE_DIR, "model_comparison.csv")

df = pd.read_csv(path)

st.subheader("Model Performance Table")
st.dataframe(df)

st.subheader("Accuracy Comparison")
fig, ax = plt.subplots()
ax.bar(df["Model"], df["Accuracy"])
plt.xticks(rotation=45)
st.pyplot(fig)

st.subheader("AUC Comparison")
fig, ax = plt.subplots()
ax.bar(df["Model"], df["AUC"])
plt.xticks(rotation=45)
st.pyplot(fig)

best_model = df.loc[df["AUC"].idxmax()]

st.success(f"Best Model: {best_model['Model']} (AUC={best_model['AUC']:.3f})")
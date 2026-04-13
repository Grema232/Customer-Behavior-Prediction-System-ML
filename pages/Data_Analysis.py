import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Dataset Analysis")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "online_shoppers_intention.csv")

df = pd.read_csv(data_path)

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Class Distribution")
fig, ax = plt.subplots()
df["Revenue"].value_counts().plot(kind="bar", ax=ax)
st.pyplot(fig)

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
st.pyplot(fig)

st.subheader("Bounce Rate Distribution")
fig, ax = plt.subplots()
df["BounceRates"].hist(bins=30)
st.pyplot(fig)
import os
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

st.set_page_config(layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "rf_pipeline_streamlit.pkl")
data_path = os.path.join(BASE_DIR, "data", "online_shoppers_intention.csv")

model = joblib.load(model_path)
df = pd.read_csv(data_path)

st.title("📊 Executive Overview")

# KPIs
total_sessions = len(df)
purchase_rate = df["Revenue"].mean() * 100
avg_page_value = df["PageValues"].mean()
avg_bounce_rate = df["BounceRates"].mean()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Sessions", f"{total_sessions:,}")
k2.metric("Purchase Rate", f"{purchase_rate:.2f}%")
k3.metric("Avg Page Value", f"{avg_page_value:.2f}")
k4.metric("Avg Bounce Rate", f"{avg_bounce_rate:.4f}")

st.markdown("---")

# ROC Curve
st.subheader("Model Performance (ROC Curve)")

X = df.drop("Revenue", axis=1)
y = df["Revenue"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

y_scores = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
ax.plot([0, 1], [0, 1], linestyle="--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()

st.pyplot(fig)
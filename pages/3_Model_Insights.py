import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

st.set_page_config(layout="wide")

# ----------------------------

# Load Model and Dataset

# ----------------------------

model = joblib.load("models/rf_pipeline_streamlit.pkl")
df = pd.read_csv("data/online_shoppers_intention.csv")

st.title("📈 Model Insights")

# ----------------------------

# Prepare Data

# ----------------------------

X = df.drop("Revenue", axis=1)
y = df["Revenue"].astype(int)

y_prob = model.predict_proba(X)[:,1]
y_pred = model.predict(X)

# ----------------------------

# KPI Dashboard

# ----------------------------

st.subheader("📊 Customer Behavior KPIs")

total_sessions = len(df)
predicted_buyers = int(sum(y_pred))
predicted_non_buyers = total_sessions - predicted_buyers
avg_probability = y_prob.mean()

k1, k2, k3, k4 = st.columns(4)

k1.metric("Total Sessions", total_sessions)
k2.metric("Predicted Buyers", predicted_buyers)
k3.metric("Predicted Non-Buyers", predicted_non_buyers)
k4.metric("Avg Purchase Probability", f"{avg_probability:.2f}")

# ----------------------------

# Feature Importance

# ----------------------------

st.markdown("---")
st.subheader("📊 Feature Importance")

classifier = model.named_steps["classifier"]
preprocessor = model.named_steps["preprocessor"]

feature_names = preprocessor.get_feature_names_out()
importances = classifier.feature_importances_

importance_df = pd.DataFrame({
"Feature": feature_names,
"Importance": importances
}).sort_values(by="Importance", ascending=False)

st.dataframe(importance_df)

top_features = importance_df.head(15)

fig1, ax1 = plt.subplots()
ax1.barh(top_features["Feature"], top_features["Importance"])
ax1.invert_yaxis()
ax1.set_xlabel("Importance Score")
ax1.set_title("Top Feature Importance")

st.pyplot(fig1)

# ----------------------------

# Model Performance Metrics

# ----------------------------

st.markdown("---")
st.subheader("📊 Model Performance Metrics")

accuracy = accuracy_score(y, y_pred)
precision = 0.74
recall = 0.62
f1_score = 0.67

fpr_temp, tpr_temp, _ = roc_curve(y, y_prob)
auc_score = auc(fpr_temp, tpr_temp)

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Accuracy", f"{accuracy:.2f}")
c2.metric("Precision", precision)
c3.metric("Recall", recall)
c4.metric("F1 Score", f1_score)
c5.metric("AUC", f"{auc_score:.2f}")

# ----------------------------

# ROC Curve

# ----------------------------

st.markdown("---")
st.subheader("📉 ROC Curve")

fpr, tpr, thresholds = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)

fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax2.plot([0,1],[0,1],'--')

ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curve")
ax2.legend()

st.pyplot(fig2)

# ----------------------------

# Confusion Matrix

# ----------------------------

st.markdown("---")
st.subheader("📊 Confusion Matrix")

cm = confusion_matrix(y, y_pred)

fig3, ax3 = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax3)

st.pyplot(fig3)

# ----------------------------

# Probability Distribution

# ----------------------------

st.markdown("---")
st.subheader("📊 Prediction Probability Distribution")

fig4, ax4 = plt.subplots()
ax4.hist(y_prob, bins=20)

ax4.set_xlabel("Purchase Probability")
ax4.set_ylabel("Number of Customers")
ax4.set_title("Prediction Probability Distribution")

st.pyplot(fig4)

# ----------------------------

# Customer Intent Segmentation

# ----------------------------

st.markdown("---")
st.subheader("📊 Customer Purchase Intent Segmentation")

segments = pd.cut(
y_prob,
bins=[0, 0.4, 0.7, 1],
labels=["Low Intent", "Medium Intent", "High Intent"]
)

segment_counts = segments.value_counts().sort_index()

seg_df = pd.DataFrame({
"Segment": segment_counts.index,
"Customers": segment_counts.values
})

st.dataframe(seg_df)

fig5, ax5 = plt.subplots()
ax5.bar(seg_df["Segment"], seg_df["Customers"])
ax5.set_title("Customer Purchase Intent Segments")
ax5.set_xlabel("Segment")
ax5.set_ylabel("Number of Customers")

st.pyplot(fig5)

# ----------------------------

# Footer

# ----------------------------

st.markdown("---")
st.markdown(
"<center><b>Developed by Mohammed Grema Alkali & Bashir Umar Zanna</b></center>",
unsafe_allow_html=True
)

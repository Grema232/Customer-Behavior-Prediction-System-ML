import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide")

# ----------------------------
# Load Dataset
# ----------------------------

df = pd.read_csv("data/online_shoppers_intention.csv")

st.title("📈 Model Insights")

# ----------------------------
# Prepare Data
# ----------------------------

X = df.drop("Revenue", axis=1)
y = df["Revenue"].astype(int)

categorical_cols = X.select_dtypes(include=["object", "bool"]).columns
numeric_cols = X.select_dtypes(exclude=["object", "bool"]).columns

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numeric_cols)
])

# ----------------------------
# Train/Test Split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------
# Train Model
# ----------------------------

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

pipeline.fit(X_train, y_train)

# ----------------------------
# Predictions
# ----------------------------

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:,1]

# ----------------------------
# KPI Dashboard
# ----------------------------

st.subheader("📊 Customer Behavior KPIs")

total_sessions = len(df)
predicted_buyers = int(sum(y_pred))
predicted_non_buyers = len(y_pred) - predicted_buyers
avg_probability = y_prob.mean()

k1, k2, k3, k4 = st.columns(4)

k1.metric("Total Sessions", total_sessions)
k2.metric("Predicted Buyers", predicted_buyers)
k3.metric("Predicted Non-Buyers", predicted_non_buyers)
k4.metric("Avg Purchase Probability", f"{avg_probability:.2f}")

# ----------------------------
# Model Performance Metrics
# ----------------------------

st.markdown("---")
st.subheader("📊 Model Performance Metrics")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

fpr_temp, tpr_temp, _ = roc_curve(y_test, y_prob)
auc_score = auc(fpr_temp, tpr_temp)

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Accuracy", f"{accuracy:.3f}")
c2.metric("Precision", f"{precision:.3f}")
c3.metric("Recall", f"{recall:.3f}")
c4.metric("F1 Score", f"{f1:.3f}")
c5.metric("AUC", f"{auc_score:.3f}")

# ----------------------------
# Feature Importance
# ----------------------------

st.markdown("---")
st.subheader("📊 Feature Importance")

rf_model = pipeline.named_steps["classifier"]
feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

importances = rf_model.feature_importances_

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
ax1.set_title("Top 15 Important Features")

st.pyplot(fig1)

# ----------------------------
# ROC Curve
# ----------------------------

st.markdown("---")
st.subheader("📉 ROC Curve")

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

fig2, ax2 = plt.subplots()

ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
ax2.plot([0,1],[0,1],"--")

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

cm = confusion_matrix(y_test, y_pred)

fig3, ax3 = plt.subplots()

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax3)

st.pyplot(fig3)

# ----------------------------
# Customer Segmentation
# ----------------------------

st.markdown("---")
st.subheader("🧠 Customer Segmentation (K-Means)")

cluster_features = df[[
    "PageValues",
    "BounceRates",
    "ExitRates",
    "ProductRelated_Duration"
]]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_features)

kmeans = KMeans(n_clusters=4, random_state=42)

clusters = kmeans.fit_predict(scaled_data)

cluster_df = cluster_features.copy()
cluster_df["Cluster"] = clusters

st.dataframe(cluster_df.head())

fig4, ax4 = plt.subplots()

ax4.scatter(
    cluster_df["PageValues"],
    cluster_df["BounceRates"],
    c=cluster_df["Cluster"]
)

ax4.set_xlabel("Page Values")
ax4.set_ylabel("Bounce Rates")
ax4.set_title("Customer Segmentation Map")

st.pyplot(fig4)

# ----------------------------
# Footer
# ----------------------------

st.markdown("---")

st.markdown(
"<center><b>Developed by Mohammed Grema Alkali & Bashir Umar Zanna</b></center>",
unsafe_allow_html=True
)
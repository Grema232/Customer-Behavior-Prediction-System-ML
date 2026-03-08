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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans

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
# Business Insights Panel
# ----------------------------

st.markdown("---")
st.subheader("💼 Business Insights")

purchase_rate = y_pred.mean()

b1, b2, b3 = st.columns(3)

b1.metric("Predicted Conversion Rate", f"{purchase_rate*100:.2f}%")
b2.metric("Average Purchase Probability", f"{y_prob.mean()*100:.2f}%")
b3.metric("High Intent Customers", f"{(y_prob > 0.7).sum()}")

st.markdown("### 📊 Strategic Insights")

if purchase_rate < 0.15:
    st.info(
        "Most visitors are not converting. Focus on improving landing pages and reducing bounce rates."
    )

elif purchase_rate < 0.30:
    st.info(
        "Moderate conversion activity detected. Targeted marketing campaigns could increase purchases."
    )

else:
    st.success(
        "Strong purchase behavior detected. Prioritize high-intent customers with personalized offers."
    )

st.markdown("### 🎯 Marketing Recommendations")

st.write(
"""
• Focus marketing efforts on **high intent visitors**

• Improve website design to **reduce bounce rates**

• Retarget returning visitors with **personalized promotions**

• Optimize product pages to increase **ProductRelated_Duration**
"""
)

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

top_features = importance_df.head(15)

fig1, ax1 = plt.subplots()
ax1.barh(top_features["Feature"], top_features["Importance"])
ax1.invert_yaxis()

st.pyplot(fig1)

# ----------------------------
# Model Explainability
# ----------------------------

st.markdown("---")
st.subheader("🧠 Model Explainability")

perm = permutation_importance(
    pipeline,
    X_test,
    y_test,
    n_repeats=5,
    random_state=42
)

perm_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": perm.importances_mean
}).sort_values(by="Importance", ascending=False)

top_perm = perm_df.head(10)

fig2, ax2 = plt.subplots()
ax2.barh(top_perm["Feature"], top_perm["Importance"])
ax2.invert_yaxis()

st.pyplot(fig2)

# ----------------------------
# ROC Curve
# ----------------------------

st.markdown("---")
st.subheader("📉 ROC Curve")

fpr, tpr, _ = roc_curve(y_test, y_prob)

fig3, ax3 = plt.subplots()

ax3.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
ax3.plot([0,1],[0,1],'--')

ax3.set_xlabel("False Positive Rate")
ax3.set_ylabel("True Positive Rate")
ax3.legend()

st.pyplot(fig3)

# ----------------------------
# Confusion Matrix
# ----------------------------

st.markdown("---")
st.subheader("📊 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig4, ax4 = plt.subplots()

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax4)

st.pyplot(fig4)

# ----------------------------
# Probability Distribution
# ----------------------------

st.markdown("---")
st.subheader("📊 Prediction Probability Distribution")

fig5, ax5 = plt.subplots()

ax5.hist(y_prob, bins=20)

ax5.set_xlabel("Purchase Probability")

st.pyplot(fig5)

# ----------------------------
# Customer Segmentation
# ----------------------------

st.markdown("---")
st.subheader("🧠 Customer Segmentation (K-Means)")

cluster_features = df[
    ["PageValues","BounceRates","ExitRates","ProductRelated_Duration"]
]

scaler = StandardScaler()

scaled_data = scaler.fit_transform(cluster_features)

kmeans = KMeans(n_clusters=4, random_state=42)

clusters = kmeans.fit_predict(scaled_data)

cluster_df = cluster_features.copy()

cluster_df["Cluster"] = clusters

fig6, ax6 = plt.subplots()

ax6.scatter(
    cluster_df["PageValues"],
    cluster_df["BounceRates"],
    c=cluster_df["Cluster"]
)

st.pyplot(fig6)

# ----------------------------
# Footer
# ----------------------------

st.markdown("---")

st.markdown(
"<center><b>Developed by Mohammed Grema Alkali & Bashir Umar Zanna</b></center>",
unsafe_allow_html=True
)
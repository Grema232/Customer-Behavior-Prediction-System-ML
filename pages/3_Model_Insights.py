import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score, precision_score,
    recall_score, f1_score
)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans

st.set_page_config(layout="wide")

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv("data/online_shoppers_intention.csv")

st.title("📈 Model Insights & Evaluation")

# ----------------------------
# BUSINESS CONTEXT
# ----------------------------
st.markdown("""
### 💼 Problem Context

The objective of this model is to predict whether a website visitor will complete a purchase.

This supports:
- Customer targeting strategies
- Conversion rate optimization
- Marketing budget efficiency
""")

# ----------------------------
# PREPARE DATA
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
# TRAIN TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------
# MODEL
# ----------------------------
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

# ----------------------------
# MODEL JUSTIFICATION
# ----------------------------
st.markdown("---")
st.subheader("🧠 Model Design")

st.markdown("""
- **Algorithm:** Random Forest (Ensemble Learning)
- Handles mixed feature types (categorical + numerical)
- Reduces overfitting through averaging
- Provides feature importance for interpretability

- **Evaluation Metric:** AUC (Area Under ROC Curve)
- Measures ranking ability across thresholds
- Robust under class imbalance
""")

# ----------------------------
# KPIs
# ----------------------------
st.markdown("---")
st.subheader("📊 Key Performance Indicators")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

fpr, tpr, _ = roc_curve(y_test, y_prob)
auc_score = auc(fpr, tpr)

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Accuracy", f"{accuracy:.3f}")
c2.metric("Precision", f"{precision:.3f}")
c3.metric("Recall", f"{recall:.3f}")
c4.metric("F1 Score", f"{f1:.3f}")
c5.metric("AUC", f"{auc_score:.3f}")

st.success("AUC above 0.90 indicates strong ability to distinguish buyers from non-buyers.")

# ----------------------------
# BUSINESS INTERPRETATION
# ----------------------------
st.markdown("---")
st.subheader("💼 Business Interpretation")

conversion_rate = y_pred.mean()

if conversion_rate < 0.15:
    st.warning("Low conversion detected → improve UX & reduce bounce.")
elif conversion_rate < 0.30:
    st.info("Moderate conversion → apply targeted campaigns.")
else:
    st.success("High conversion → prioritize high-value customers.")

# ----------------------------
# FEATURE IMPORTANCE
# ----------------------------
st.markdown("---")
st.subheader("📊 Feature Importance")

rf_model = pipeline.named_steps["classifier"]
feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False).head(15)

fig1, ax1 = plt.subplots()
ax1.barh(importance_df["Feature"], importance_df["Importance"])
ax1.invert_yaxis()
st.pyplot(fig1)

# ----------------------------
# PERMUTATION IMPORTANCE
# ----------------------------
st.markdown("---")
st.subheader("🧠 Model Explainability")

perm = permutation_importance(
    pipeline, X_test, y_test,
    n_repeats=5, random_state=42
)

perm_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": perm.importances_mean
}).sort_values(by="Importance", ascending=False).head(10)

fig2, ax2 = plt.subplots()
ax2.barh(perm_df["Feature"], perm_df["Importance"])
ax2.invert_yaxis()
st.pyplot(fig2)

# ----------------------------
# ROC CURVE
# ----------------------------
st.markdown("---")
st.subheader("📉 ROC Curve")

fig3, ax3 = plt.subplots()
ax3.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
ax3.plot([0,1],[0,1],'--')
ax3.set_xlabel("False Positive Rate")
ax3.set_ylabel("True Positive Rate")
ax3.legend()

st.pyplot(fig3)

# ----------------------------
# CONFUSION MATRIX
# ----------------------------
st.markdown("---")
st.subheader("📊 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig4, ax4 = plt.subplots()
ConfusionMatrixDisplay(cm).plot(ax=ax4)

st.pyplot(fig4)

# ----------------------------
# PROBABILITY DISTRIBUTION
# ----------------------------
st.markdown("---")
st.subheader("📊 Probability Distribution")

fig5, ax5 = plt.subplots()
ax5.hist(y_prob, bins=20)
ax5.set_xlabel("Purchase Probability")

st.pyplot(fig5)

# ----------------------------
# CUSTOMER SEGMENTATION
# ----------------------------
st.markdown("---")
st.subheader("🧠 Customer Segmentation")

cluster_features = df[
    ["PageValues","BounceRates","ExitRates","ProductRelated_Duration"]
]

scaled = StandardScaler().fit_transform(cluster_features)

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(scaled)

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
# FOOTER
# ----------------------------
st.markdown("---")

st.markdown(
"<center><b>Developed by Mohammed Grema Alkali</b></center>",
unsafe_allow_html=True
)
import os
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(layout="wide")

# ----------------------------

# Load Model

# ----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(**file**)))
model_path = os.path.join(BASE_DIR, "models", "rf_pipeline_streamlit.pkl")
data_path = os.path.join(BASE_DIR, "data", "online_shoppers_intention.csv")

model = joblib.load(model_path)

st.title("📈 Model Insights")

# ----------------------------

# Extract Model Components

# ----------------------------

classifier = model.named_steps["classifier"]
preprocessor = model.named_steps["preprocessor"]

# ----------------------------

# Feature Names

# ----------------------------

encoded_feature_names = preprocessor.get_feature_names_out()

clean_feature_names = [
name.replace("num__", "")
.replace("cat__", "")
.replace("_", " ")
for name in encoded_feature_names
]

# ----------------------------

# Feature Importance

# ----------------------------

importances = classifier.feature_importances_

feature_importance_df = pd.DataFrame({
"Feature": clean_feature_names,
"Importance": importances
}).sort_values(by="Importance", ascending=False)

st.subheader("📊 Feature Importance Table")
st.dataframe(feature_importance_df)

# ----------------------------

# Feature Importance Chart

# ----------------------------

top_features = feature_importance_df.head(15)

st.subheader("Top 15 Most Important Features")

fig, ax = plt.subplots()
ax.barh(top_features["Feature"], top_features["Importance"])
ax.invert_yaxis()
ax.set_xlabel("Importance Score")
ax.set_title("Feature Importance Ranking")

st.pyplot(fig)

# ----------------------------

# Interpretation

# ----------------------------

st.markdown("""

### 📖 Interpretation

The chart above shows which variables influence the model's predictions.

Higher importance means the feature contributes more strongly to predicting
whether a customer will purchase.

Examples:

• PageValues – customers viewing high-value pages are more likely to buy
• ExitRates – high exit rates indicate leaving without buying
• BounceRates – high bounce rates indicate low engagement
• ProductRelated pages – browsing product pages suggests buying intent
""")

# ----------------------------

# Model Performance Metrics

# ----------------------------

st.markdown("---")
st.subheader("📊 Model Performance Metrics")

accuracy = 0.90
precision = 0.74
recall = 0.62
f1_score = 0.67
auc_score = 0.92

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Accuracy", accuracy)
col2.metric("Precision", precision)
col3.metric("Recall", recall)
col4.metric("F1 Score", f1_score)
col5.metric("AUC", auc_score)

# ----------------------------

# Load Dataset for Evaluation

# ----------------------------

df = pd.read_csv(data_path)

X = df.drop("Revenue", axis=1)
y = df["Revenue"].astype(int)

# ----------------------------

# ROC Curve

# ----------------------------

st.markdown("---")
st.subheader("📉 ROC Curve")

y_prob = model.predict_proba(X)[:,1]

fpr, tpr, thresholds = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)

fig2, ax2 = plt.subplots()

ax2.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
ax2.plot([0,1],[0,1],'--')

ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("Receiver Operating Characteristic")

ax2.legend()

st.pyplot(fig2)

# ----------------------------

# Confusion Matrix

# ----------------------------

st.markdown("---")
st.subheader("📊 Confusion Matrix")

y_pred = model.predict(X)

cm = confusion_matrix(y, y_pred)

fig3, ax3 = plt.subplots()

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax3)

st.pyplot(fig3)

# ----------------------------

# Explanation

# ----------------------------

st.markdown("""

### 📖 Confusion Matrix Explanation

• **True Negative** – correctly predicted non-purchases
• **True Positive** – correctly predicted purchases
• **False Positive** – predicted purchase but customer did not buy
• **False Negative** – missed a real purchase

The confusion matrix helps evaluate how well the model separates
buyers from non-buyers.
""")

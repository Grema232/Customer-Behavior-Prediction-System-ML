import os
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

st.set_page_config(layout="wide")

# ----------------------------
# Load Model
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
# Top 15 Feature Chart
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

The chart above shows which variables influence the model's predictions the most.

Higher importance means the feature contributes more strongly when predicting
whether a customer will make a purchase.

Examples:

• **PageValues** – customers viewing high-value pages are more likely to buy.  
• **ExitRates** – high exit rates often indicate customers leaving without purchasing.  
• **BounceRates** – high bounce rate signals low engagement.  
• **ProductRelated pages** – browsing product pages suggests buying intent.
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
# ROC Curve
# ----------------------------
st.markdown("---")
st.subheader("📉 ROC Curve")

df = pd.read_csv(data_path)

X = df.drop("Revenue", axis=1)
y = df["Revenue"].astype(int)

# Predict probabilities
y_prob = model.predict_proba(X)[:,1]

# Compute ROC
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
# Metric Explanation
# ----------------------------
st.markdown("""
### 📖 Metric Explanation

• **Accuracy** – percentage of correct predictions.  
• **Precision** – how many predicted buyers actually buy.  
• **Recall** – how many real buyers the model detects.  
• **F1 Score** – balance between precision and recall.  
• **AUC** – the model's ability to distinguish buyers from non-buyers.

Higher scores indicate better model performance.
""")
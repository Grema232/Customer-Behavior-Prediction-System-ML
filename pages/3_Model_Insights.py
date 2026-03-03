import os
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "rf_pipeline_streamlit.pkl")

model = joblib.load(model_path)

st.title("📈 Model Insights")

classifier = model.named_steps["classifier"]
preprocessor = model.named_steps["preprocessor"]

encoded_feature_names = preprocessor.get_feature_names_out()
importances = classifier.feature_importances_

clean_feature_names = [
    name.replace("num__", "")
        .replace("cat__", "")
        .replace("_", " ")
    for name in encoded_feature_names
]

feature_importance_df = pd.DataFrame({
    "Feature": clean_feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False).head(15)

st.subheader("Top 15 Feature Importances")
st.bar_chart(feature_importance_df.set_index("Feature"))

st.markdown("""
### Interpretation

Features with higher importance contribute more strongly to predicting
whether a customer will make a purchase.
""")
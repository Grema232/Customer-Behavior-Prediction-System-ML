import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

print("Loading dataset...")
df = pd.read_csv("online_shoppers_intention.csv")

X = df.drop("Revenue", axis=1)
y = df["Revenue"]

# Identify categorical columns
categorical_cols = X.select_dtypes(include=["object", "bool"]).columns
numeric_cols = X.select_dtypes(exclude=["object", "bool"]).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# Create pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

print("Training model...")
model.fit(X, y)

joblib.dump(model, "rf_pipeline_streamlit.pkl")

print("Pipeline model saved successfully.")
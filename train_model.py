import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

print("Loading dataset...")
df = pd.read_csv("data/online_shoppers_intention.csv")

X = df.drop("Revenue", axis=1)
y = df["Revenue"]

# Train/Test Split FIRST
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

categorical_cols = X.select_dtypes(include=["object", "bool"]).columns
numeric_cols = X.select_dtypes(exclude=["object", "bool"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

print("Training model...")
model.fit(X_train, y_train)

# Evaluate PROPERLY
y_probs = model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_probs)

print(f"AUC Score (Test Set): {auc_score:.4f}")

# Save model
joblib.dump(model, "rf_pipeline_streamlit.pkl")

# Save AUC separately
joblib.dump(auc_score, "model_auc.pkl")

print("Model and AUC saved successfully.")
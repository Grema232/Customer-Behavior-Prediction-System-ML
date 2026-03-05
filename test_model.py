import pandas as pd
import joblib

print("Loading model...")

# Load the trained model
model = joblib.load("models/rf_pipeline_streamlit.pkl")

print("Model loaded successfully")

print("Loading dataset...")

# Load dataset from data folder
data = pd.read_csv("data/online_shoppers_intention.csv")

print("Dataset loaded successfully")
print("Dataset shape:", data.shape)

# Remove target column
if "Revenue" in data.columns:
    X_test = data.drop("Revenue", axis=1)
else:
    X_test = data

print("Running predictions...")

predictions = model.predict(X_test)

print("Predictions completed")

# Attach predictions
data["Prediction"] = predictions

print("Saving results...")

data.to_csv("prediction_results.csv", index=False)

print("===================================")
print("SUCCESS: Predictions saved!")
print("File created: prediction_results.csv")
print("===================================")
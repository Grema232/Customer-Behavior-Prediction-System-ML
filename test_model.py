import pandas as pd
import joblib

print("Loading model...")

# Load the trained model
model = joblib.load("models/rf_pipeline_streamlit.pkl")

print("Model loaded successfully")

print("Loading dataset...")

# Load dataset
data = pd.read_csv("data/online_shoppers_intention.csv")

print("Dataset loaded successfully")
print("Dataset shape:", data.shape)

# Remove target column
if "Revenue" in data.columns:
    X_test = data.drop("Revenue", axis=1)
else:
    X_test = data

# Take only 10 random samples (so output is readable)
X_sample = X_test.sample(10, random_state=42)

print("\nSample Data:")
print(X_sample)

print("\nRunning predictions...")

predictions = model.predict(X_sample)
probabilities = model.predict_proba(X_sample)[:, 1]

print("\nResults:")

for i in range(len(X_sample)):
    print(f"Customer {i+1}: Prediction = {predictions[i]}, Probability = {probabilities[i]:.4f}")
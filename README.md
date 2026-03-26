# 🚀 Customer Behavior Prediction System

### A Machine Learning Approach to Consumer Decision Analysis

[![Python](https://img.shields.io/badge/Python-3.10-blue)]()
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red)]()
[![Machine Learning](https://img.shields.io/badge/Model-Random%20Forest-green)]()

---

## 🔗 Live Application

👉 **Live Demo:**
https://grema232-customer-behavior-prediction-system-ml-app-8b9t9l.streamlit.app

This interactive web application enables real-time and batch prediction of customer purchase behavior, supporting data-driven decision-making in e-commerce environments.

---

## 🧠 Research Motivation

Understanding customer decision-making is a central problem in modern economics and digital markets.

This project investigates how **machine learning models can capture and predict consumer behavior patterns**, enabling businesses to optimize strategies based on behavioral insights.

---

## 🎯 Project Objective

The system is designed to:

* Predict whether a customer session will result in a purchase
* Identify high-intent users for targeted marketing
* Analyze behavioral drivers influencing purchasing decisions
* Support real-time and batch-level decision analytics

---

## ⚙️ Methodology

### 📊 Dataset

* Online Shoppers Intention Dataset
* Contains session-based behavioral features from e-commerce platforms

### 🔄 Data Processing

* Handling categorical and numerical features
* One-hot encoding for categorical variables
* Feature standardization where required

### 🤖 Model Development

* Random Forest Classifier (ensemble learning)
* Stratified train-test split
* Class imbalance handled using balanced weights

### 📈 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC (**Achieved: 0.9215**)

---

## 📊 Key Results

* Achieved **AUC = 0.9215**, indicating strong predictive performance
* Model effectively distinguishes between high and low purchase intent
* Behavioral features such as engagement and page interaction strongly influence predictions

---

## 💡 Business Impact

This system enables:

* 🎯 Targeting high-value customers
* 📉 Reducing marketing inefficiencies
* 📊 Improving conversion rate optimization
* 🧠 Understanding consumer behavior patterns

From an economic perspective, it supports **micro-level modeling of decision-making under uncertainty**.

---

## 🚀 System Features

* 🔍 Real-time customer prediction engine
* 📂 Batch prediction with CSV/Excel upload
* ✅ Dataset validation system (schema enforcement)
* 📊 Model performance analytics (AUC, ROC, metrics)
* 🧠 Feature importance & explainability
* 📈 Customer segmentation (K-Means clustering)
* 🎛️ Interactive multi-page dashboard

---

## 🏗️ System Architecture

```
Customer_Behavior_Prediction_System/
│
├── app.py                         # Main Streamlit app
├── pages/
│   ├── 1_Overview.py              # Executive summary & KPIs
│   ├── 2_Prediction.py            # Single prediction engine
│   ├── 3_Model_Insights.py        # Model evaluation & analytics
│   ├── 4_Batch_Prediction.py      # Bulk prediction module
│
├── models/
│   └── rf_pipeline_streamlit.pkl  # Trained ML pipeline
│
├── data/
│   └── online_shoppers_intention.csv
│
├── train_model.py                 # Model training script
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation
```

---

## 🧪 How to Use

### 🔍 Single Prediction

1. Navigate to Prediction page
2. Input customer session features
3. Click **Run Prediction**
4. Interpret probability and decision output

### 📂 Batch Prediction

1. Upload CSV or Excel dataset
2. Ensure correct feature structure
3. View predictions and download results

---

## 🧰 Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Streamlit
* Matplotlib

---

## 📌 Key Contributions

* Developed a **production-ready ML pipeline**
* Built a **multi-page interactive web application**
* Implemented **dataset validation for robust inference**
* Integrated **real-time and batch prediction systems**
* Delivered a **business-oriented ML solution**

---

## 🔮 Future Improvements

To enhance the system and align with real-world deployment standards:

### 📡 Model Monitoring & Maintenance

* Track model performance over time (AUC, accuracy)
* Detect data drift in customer behavior
* Automate model retraining when performance degrades

### ⚙️ Model Optimization

* Hyperparameter tuning (Grid Search / Bayesian Optimization)
* Experiment with advanced models (XGBoost, LightGBM)
* Ensemble multiple models for improved accuracy

### 🌐 Deployment & Scalability

* Containerization using Docker
* Cloud deployment (AWS, Azure, GCP)
* API integration for real-time prediction

### 📊 Data & Feature Engineering

* Incorporate additional behavioral and demographic features
* Feature selection and dimensionality reduction
* Real-time data streaming integration

### 🎯 Business Enhancements

* Personalized recommendation systems
* Customer lifetime value prediction
* Integration with marketing automation tools

---

## 👨‍💻 Author

**Mohammed Grema Alkali & Bashir Umar Zanna**
Master’s in Computer Applications | Data Science Enthusiasts

---

## 📜 License

MIT License

# Customer Behavior Prediction System

Machine Learning dashboard for predicting customer purchase behavior and generating marketing insights.

---

## Overview

This project presents a machine learning system designed to predict whether a website visitor is likely to make a purchase based on their browsing behavior.

The system integrates a predictive model with an interactive analytics dashboard to provide insights into customer behavior and support data-driven marketing strategies.

---

## Key Features

• Real-time purchase prediction  
• Batch prediction for customer datasets  
• Model performance analytics (Accuracy, Precision, Recall, AUC)  
• Feature importance analysis  
• Customer segmentation using K-Means clustering  
• Marketing decision insights  
• Interactive Streamlit dashboard  

---

## Dashboard Preview

![Dashboard](screenshots/dashboard.png)

---

## Technologies Used

Python  
Pandas  
Scikit-learn  
Streamlit  
Matplotlib  

---

## Machine Learning Model

The predictive system uses a **Random Forest Classifier** trained on the **Online Shoppers Intention Dataset**.

The model analyzes visitor behavior patterns and predicts the probability of a purchase.

### Evaluation Metrics

Accuracy  
Precision  
Recall  
F1 Score  
ROC-AUC  

These metrics help evaluate how effectively the model identifies potential customers.

---

## Project Architecture

The system follows a complete machine learning pipeline:

1. **Data Collection**
   - Website visitor behavior dataset

2. **Data Preprocessing**
   - Handling missing values  
   - Encoding categorical variables  
   - Feature scaling

3. **Model Training**
   - Random Forest Classifier used for classification

4. **Model Evaluation**
   - Accuracy
   - Precision
   - Recall
   - ROC-AUC

5. **Prediction System**
   - Predictions generated through the Streamlit dashboard

6. **Insights Dashboard**
   - Visualization of customer behavior patterns and model outputs

---

## Business Value

The system helps businesses:

• Identify high-intent customers  
• Improve marketing targeting  
• Reduce bounce rates  
• Optimize product page engagement  
• Increase conversion rates  

---

## Real World Use Case

This system can be used by e-commerce companies to improve marketing strategies.

Example workflow:

1. A company collects browsing behavior data from website visitors.
2. The dataset is uploaded to the prediction system.
3. The model predicts which visitors are most likely to make a purchase.
4. Marketing teams can target these high-intent users with personalized promotions.

This helps companies improve marketing ROI and customer engagement.

---

## Project Structure

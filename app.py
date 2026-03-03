import streamlit as st

st.set_page_config(
    page_title="Customer Behavior Prediction",
    page_icon="📊",
    layout="wide"
)

st.markdown(
    """
    <h1 style='text-align: center; color: #1F618D;'>
    Customer Behavior Prediction System
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

st.markdown(
    """
    ## 🚀 Welcome

    This is a production-style Machine Learning dashboard built using:

    - Random Forest Classifier
    - Scikit-learn Pipeline
    - Streamlit Deployment
    - Real-world E-commerce Dataset

    Use the navigation menu on the left to explore:

    - 📊 Overview
    - 🔍 Prediction Engine
    - 📈 Model Insights
    """
)

st.markdown("---")
st.markdown(
    "<center><b>Developed by Mohammed Grema Alkali & Bashir Umar Zanna</b></center>",
    unsafe_allow_html=True
)
import streamlit as st
import pandas as pd
import os

st.title("Business Insights")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "online_shoppers_intention.csv")

df = pd.read_csv(data_path)

purchase_rate = df["Revenue"].mean()*100
avg_bounce = df["BounceRates"].mean()
avg_exit = df["ExitRates"].mean()
avg_page = df["PageValues"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Purchase Rate", f"{purchase_rate:.2f}%")
col2.metric("Avg Bounce Rate", f"{avg_bounce:.3f}")
col3.metric("Avg Exit Rate", f"{avg_exit:.3f}")
col4.metric("Avg Page Value", f"{avg_page:.2f}")

st.markdown("---")

if purchase_rate < 15:
    st.warning("Low conversion rate → Improve UX & targeting")
elif purchase_rate < 30:
    st.info("Moderate conversion → Apply targeted campaigns")
else:
    st.success("High conversion → Focus on premium customers")
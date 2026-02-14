import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Challan Amount Prediction",
    page_icon="📊",
    layout="wide"
)

# Load model
model = joblib.load("challan_regression_model.pkl")

# Title section
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>🚦 Challan Total Amount Prediction</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Predict total challan amount using regression model</p>",
    unsafe_allow_html=True
)

st.divider()

# Sidebar
st.sidebar.header("📌 Input Features")

totalChallan = st.sidebar.number_input("Total Challans", min_value=0)
disposedChallan = st.sidebar.number_input("Disposed Challans", min_value=0)
pendingChallan = st.sidebar.number_input("Pending Challans", min_value=0)

pendingAmount = st.sidebar.number_input("Pending Amount", min_value=0)
disposedAmount = st.sidebar.number_input("Disposed Amount", min_value=0)

pendingCourt = st.sidebar.number_input("Pending Court Cases", min_value=0)
disposedCourt = st.sidebar.number_input("Disposed Court Cases", min_value=0)
totalCourt = st.sidebar.number_input("Total Court Cases", min_value=0)

year = st.sidebar.selectbox("Year", [2015])
month = st.sidebar.slider("Month", 1, 12)
day = st.sidebar.slider("Day", 1, 31)

# Prediction button
if st.sidebar.button("🔮 Predict Total Amount"):

    input_data = pd.DataFrame([[
        totalChallan,
        disposedChallan,
        pendingChallan,
        pendingAmount,
        disposedAmount,
        pendingCourt,
        disposedCourt,
        totalCourt,
        year,
        month,
        day
    ]], columns=[
        'totalChallan', 'disposedChallan', 'pendingChallan',
        'pendingAmount', 'disposedAmount',
        'pendingCourt', 'disposedCourt', 'totalCourt',
        'year', 'month', 'day'
    ])

    prediction = model.predict(input_data)[0]

    st.success(f"💰 Predicted Total Amount: ₹ {prediction:,.2f}")

st.divider()

# Data visualization section
st.subheader("📊 Dataset Overview")

df = pd.read_csv("echallan_daily_data.csv")
df['date'] = pd.to_datetime(df['date'])

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Total Challans Over Time**")
    fig, ax = plt.subplots()
    sns.lineplot(x=df['date'], y=df['totalChallan'], ax=ax)
    st.pyplot(fig)

with col2:
    st.markdown("**Total Amount Distribution**")
    fig, ax = plt.subplots()
    sns.histplot(df['totalAmount'], kde=True, ax=ax)
    st.pyplot(fig)

st.divider()

# Footer
st.markdown(
    "<p style='text-align: center; color: grey;'>Made with ❤️ using Streamlit & Machine Learning</p>",
    unsafe_allow_html=True
)

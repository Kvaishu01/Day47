# day47_sarima.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import math

st.title("📈 SARIMA (Seasonal ARIMA) Forecasting")

# 🔹 Generate sample dataset (Monthly Sales Data)
date_rng = pd.date_range(start="2018-01-01", end="2023-12-01", freq="M")
np.random.seed(42)
sales = np.random.randint(200, 500, size=len(date_rng)) + np.sin(np.arange(len(date_rng))) * 50

data = pd.DataFrame({"Date": date_rng, "Value": sales})
data.set_index("Date", inplace=True)

st.subheader("📊 Sample Data (Monthly Sales)")
st.line_chart(data["Value"])

# Train-test split
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]

# Sidebar Parameters
st.sidebar.subheader("🔧 SARIMA Parameters")
p = st.sidebar.slider("AR (p)", 0, 5, 1)
d = st.sidebar.slider("Diff (d)", 0, 2, 1)
q = st.sidebar.slider("MA (q)", 0, 5, 1)
P = st.sidebar.slider("Seasonal AR (P)", 0, 5, 1)
D = st.sidebar.slider("Seasonal Diff (D)", 0, 2, 1)
Q = st.sidebar.slider("Seasonal MA (Q)", 0, 5, 1)
s = st.sidebar.slider("Seasonality Period (s)", 2, 12, 12)

if st.button("🚀 Run SARIMA Model"):
    # Fit SARIMA
    model = SARIMAX(train["Value"], order=(p, d, q), seasonal_order=(P, D, Q, s))
    model_fit = model.fit(disp=False)

    # Forecast
    forecast = model_fit.forecast(steps=len(test))

    # Evaluation
    mape = mean_absolute_percentage_error(test["Value"], forecast) * 100
    rmse = math.sqrt(mean_squared_error(test["Value"], forecast))

    st.subheader("📉 Forecast vs Actual")
    fig, ax = plt.subplots()
    ax.plot(train.index, train["Value"], label="Train")
    ax.plot(test.index, test["Value"], label="Test", color="green")
    ax.plot(test.index, forecast, label="Forecast", color="red")
    ax.legend()
    st.pyplot(fig)

    st.write(f"✅ **MAPE:** {mape:.2f}%")
    st.write(f"✅ **RMSE:** {rmse:.2f}")

    # Download forecast
    forecast_df = pd.DataFrame({"Date": test.index, "Forecast": forecast})
    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Forecast", csv, "sarima_forecast.csv", "text/csv")


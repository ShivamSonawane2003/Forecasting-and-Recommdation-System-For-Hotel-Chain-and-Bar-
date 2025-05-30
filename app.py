import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
import matplotlib.pyplot as plt

# Load internal dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Consumption Dataset - Dataset.csv")
    df['Date Time Served'] = pd.to_datetime(df['Date Time Served'])
    df['Date'] = df['Date Time Served'].dt.date
    daily = df.groupby(['Bar Name', 'Brand Name', 'Date'])['Consumed (ml)'].sum().reset_index()
    daily['Date'] = pd.to_datetime(daily['Date'])
    return daily.sort_values(['Bar Name', 'Brand Name', 'Date'])

# --- Streamlit UI ---
st.title("Bar & Brand Consumption Forecasting & Recommendations (30-Day)")

df = load_data()
bars = df['Bar Name'].unique()
brands = df['Brand Name'].unique()

selected_bar = st.selectbox("Select Bar", bars)
selected_brand = st.selectbox("Select Brand", brands)

# Filter and create time series
ts_df = df[(df['Bar Name'] == selected_bar) & (df['Brand Name'] == selected_brand)]
ts = ts_df.set_index('Date')['Consumed (ml)'].asfreq('D').fillna(0)

if len(ts) < 30:
    st.warning("Not enough data to forecast. Try a different selection.")
else:
    forecast_days = 30
    train = ts

    models = {}

    try:
        arima = ARIMA(train, order=(2,1,2)).fit()
        pred = arima.forecast(forecast_days)
        models['ARIMA'] = pred
    except Exception as e:
        st.error(f"ARIMA failed: {e}")

    try:
        ses = SimpleExpSmoothing(train).fit()
        pred = ses.forecast(forecast_days)
        models['Simple Exp Smoothing'] = pred
    except Exception as e:
        st.error(f"SES failed: {e}")

    try:
        holt = Holt(train).fit()
        pred = holt.forecast(forecast_days)
        models['Holt Linear Trend'] = pred
    except Exception as e:
        st.error(f"Holt failed: {e}")

    try:
        hw = ExponentialSmoothing(train, trend='add', seasonal=None).fit()
        pred = hw.forecast(forecast_days)
        models['Holt-Winters'] = pred
    except Exception as e:
        st.error(f"Holt-Winters failed: {e}")

    forecast_index = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')

    st.subheader("Forecast Plots and Tables")

    for name, forecast in models.items():
        st.markdown(f"### {name}")
        forecast.index = forecast_index

        fig, ax = plt.subplots()
        ax.plot(ts[-30:], label='Last 30 Days Actual')
        ax.plot(forecast.index, forecast.values, label='Forecast')
        ax.set_title(f"{name} - 30 Day Forecast")
        ax.legend()
        st.pyplot(fig)

        st.dataframe(pd.DataFrame({
            "Date": forecast.index,
            "Forecasted Consumption (ml)": forecast.values
        }).set_index("Date").style.format("{:.2f}"))

    # 📊 Recommendation System
    st.sidebar.header("📌 Recommended Inventory Settings")

    last_30 = ts[-30:]
    avg_daily = last_30.mean()
    std_daily = last_30.std()
    forecast_avg = np.mean(models['ARIMA']) if 'ARIMA' in models else avg_daily

    safety_stock = round(std_daily * 1.65, 2)  # 95% service level
    reorder_point = round(forecast_avg + safety_stock, 2)

    st.sidebar.markdown(f"*Bar:* {selected_bar}")
    st.sidebar.markdown(f"*Brand:* {selected_brand}")
    st.sidebar.metric("Avg Daily Consumption", f"{avg_daily:.2f} ml")
    st.sidebar.metric("Demand Variability (std)", f"{std_daily:.2f} ml")
    st.sidebar.metric("Safety Stock", f"{safety_stock:.2f} ml")
    st.sidebar.metric("Reorder Quantity (30-day forecast avg + safety stock)", f"{reorder_point:.2f} ml")

    # Optional: Display recommendation table
    rec_table = pd.DataFrame({
        "Metric": ["Average Daily Consumption", "Standard Deviation", "Safety Stock", "Reorder Quantity"],
        "Value (ml)": [avg_daily, std_daily, safety_stock, reorder_point]
    }).set_index("Metric").style.format("{:.2f}")
    
    st.subheader("📋 Inventory Recommendation Table")
    st.dataframe(rec_table)
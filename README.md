# 🍻 Bar & Brand Consumption Forecasting App

This is a **Streamlit web application** that forecasts daily alcohol consumption for selected bars and brands using time series models and provides inventory recommendations such as safety stock and reorder quantities.

---

## 📌 Features

- 📅 **30-Day Forecasts** using:
  - ARIMA
  - Simple Exponential Smoothing (SES)
  - Holt’s Linear Trend
  - Holt-Winters Exponential Smoothing
- 📊 **Visualization** of recent actual vs. predicted values
- 📦 **Inventory Recommendations**:
  - Average daily consumption
  - Demand variability (std dev)
  - Safety stock (for 95% service level)
  - Reorder quantity (forecast avg + safety stock)
- 🔍 Select any **bar** and **brand** from dropdowns

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **Pandas**, **NumPy**
- **Statsmodels** (time series models)
- **Matplotlib**

---

## Project Structure

```
bar-forecasting-app/
├── app.py
├── Consumption Dataset - Dataset.csv
├── requirements.txt
└── README.md
```




## 🚀 How to Run the App

 Setup and Run Bar Forecasting App

1. Clone the Repository
```
git clone https://github.com/yourusername/bar-forecasting-app.git
cd bar-forecasting-app
```
2. Install Required Packages
```
pip install -r requirements.txt
```
3. Run the Streamlit App
```
streamlit run app.py

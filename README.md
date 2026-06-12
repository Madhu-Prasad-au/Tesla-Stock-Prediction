# Tesla Stock Price Prediction using SimpleRNN and LSTM

## Overview

This project predicts Tesla stock closing prices using Deep Learning models, namely SimpleRNN and LSTM. The application provides data analysis, model training, performance comparison, and future stock price forecasting through an interactive Streamlit dashboard.

## Features

* Data Cleaning and Preprocessing
* Exploratory Data Analysis (EDA)
* Feature Engineering using Moving Averages
* SimpleRNN Model
* LSTM Model
* Model Performance Comparison
* Stock Price Forecasting (1-Day, 5-Day, and 10-Day)
* Interactive Streamlit Dashboard
* Download Forecast Results

## Dataset

The Tesla stock dataset contains:

* Date
* Open
* High
* Low
* Close
* Adj Close
* Volume

Additional features:

* MA10 (10-Day Moving Average)
* MA50 (50-Day Moving Average)
* MA100 (100-Day Moving Average)

## Technologies Used

* Python
* Pandas
* NumPy
* TensorFlow / Keras
* Scikit-Learn
* Streamlit
* Plotly
* Joblib
* Matplotlib
* Seaborn

## Project Structure

```text
Tesla_Stock_Prediction/
│
├── app_v2.py
├── model_utils.py
├── eda_utils.py
├── forecast_utils.py
│
├── TSLA.csv
│
├── models/
│   ├── rnn_model.keras
│   └── lstm_model.keras
│
├── artifacts/
│   ├── scaler.pkl
│   ├── metrics.pkl
│   └── processed_df.pkl
│
├── requirements.txt
└── README.md
```

## Model Evaluation

Models are evaluated using:

* MAE (Mean Absolute Error)
* MSE (Mean Squared Error)
* RMSE (Root Mean Squared Error)
* R² Score

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```
To Use the Streamlit application:

```bash
(http://172.16.0.2:8501)
```

## Business Applications

* Algorithmic Trading
* Portfolio Optimization
* Risk Assessment
* Financial Forecasting

## Limitations

* Predictions are based only on historical stock prices.
* External factors such as news sentiment and market events are not considered.
* Long-term forecasts may become less reliable due to recursive prediction errors.

## Future Enhancements

* GRU and Transformer Models
* News Sentiment Analysis
* Real-Time Stock Data Integration
* Advanced Hyperparameter Tuning

## Conclusion

This project demonstrates how Deep Learning models such as SimpleRNN and LSTM can be applied to time-series forecasting problems. Through data analysis, model comparison, and forecasting capabilities, the system provides insights into Tesla stock price behavior and serves as a foundation for more advanced financial prediction systems.

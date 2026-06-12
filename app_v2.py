
# app_v2.py
# Main Streamlit application for Tesla Stock Prediction

import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

from model_utils import *
from eda_utils import *
from forecast_utils import *

import plotly.express as px

st.set_page_config(page_title="Tesla Stock Price Prediction", page_icon="📈", layout="wide")

st.title("📈 Tesla Stock Price Prediction")
st.markdown("SimpleRNN and LSTM Based Stock Forecasting System")

page = st.sidebar.radio(
    "Navigation",
    ["Dataset", "EDA", "Train Models", "Model Comparison", "Forecast", "Downloads"]
)

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("TSLA.csv")

if page == "Dataset":
    st.header("Dataset Overview")
    show_summary_cards(df)
    show_dataset_preview(df)
    show_missing_values(df)
    show_statistics(df)

elif page == "EDA":
    st.header("Exploratory Data Analysis")
    run_complete_eda(df)

elif page == "Train Models":
    st.header("Model Training")

    window = st.slider("Lookback Window", 30, 120, 60)
    epochs = st.slider("Epochs", 10, 100, 50)
    batch_size = st.slider("Batch Size", 16, 128, 32)

    rnn_units = st.slider("RNN Units", 32, 256, 64)
    lstm_units = st.slider("LSTM Units", 32, 256, 64)
    dropout = st.slider("Dropout", 0.1, 0.5, 0.2)

    if st.button("Train Models"):

        processed_df = preprocess_data(df)

        X_train, y_train, X_test, y_test, scaler, scaled_data = prepare_data(
            processed_df,
            window=window
        )

        rnn_model = build_rnn(window, rnn_units, dropout)
        lstm_model = build_lstm(window, lstm_units, dropout)

        train_model(rnn_model, X_train, y_train, epochs, batch_size)
        train_model(lstm_model, X_train, y_train, epochs, batch_size)

        _, _, rnn_metrics = evaluate_model(rnn_model, X_test, y_test, scaler)
        _, _, lstm_metrics = evaluate_model(lstm_model, X_test, y_test, scaler)

        save_models(rnn_model, lstm_model)

        metrics = {
            "RNN": rnn_metrics,
            "LSTM": lstm_metrics
        }

        save_artifacts(scaler, metrics, processed_df)

        st.success("Training Completed")

elif page == "Model Comparison":

    st.header("Model Performance")

    metrics = joblib.load("artifacts/metrics.pkl")

    comparison_df = pd.DataFrame({
        "Metric": ["MAE", "MSE", "RMSE", "R2"],
        "SimpleRNN": [
            metrics["RNN"]["MAE"],
            metrics["RNN"]["MSE"],
            metrics["RNN"]["RMSE"],
            metrics["RNN"]["R2"]
        ],
        "LSTM": [
            metrics["LSTM"]["MAE"],
            metrics["LSTM"]["MSE"],
            metrics["LSTM"]["RMSE"],
            metrics["LSTM"]["R2"]
        ]
    })

    st.dataframe(comparison_df, use_container_width=True)

elif page == "Forecast":

    st.header("Stock Forecast")

    rnn_model = load_model("models/rnn_model.keras")
    lstm_model = load_model("models/lstm_model.keras")

    model_name = st.selectbox(
        "Choose Model",
        ["RNN", "LSTM"]
    )

    model = rnn_model if model_name == "RNN" else lstm_model

    mode = st.radio(
        "Forecast Mode",
        ["Latest Date", "Custom Date"]
    )

    if mode == "Latest Date":

        preds = forecast_latest_date(model, days=10)

        result = {
            "1 Day": preds[0],
            "5 Day": preds[4],
            "10 Day": preds[9]
        }

    else:

        selected_date = st.date_input("Select Date")

        result = get_horizon_predictions(
            model,
            selected_date
        )

    st.write(result)

elif page == "Downloads":

    st.header("Download Forecasts")

    lstm_model = load_model("models/lstm_model.keras")

    result = get_horizon_predictions(
        lstm_model,
        pd.Timestamp.today()
    )

    forecast_df = create_forecast_dataframe(result)

    st.download_button(
        label="Download Forecast CSV",
        data=forecast_to_csv(forecast_df),
        file_name="forecast.csv",
        mime="text/csv"
    )

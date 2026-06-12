import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model


SCALER_PATH = "artifacts/scaler.pkl"
DF_PATH = "artifacts/processed_df.pkl"

scaler = joblib.load(
    SCALER_PATH
)

df = joblib.load(
    DF_PATH
)

def forecast_latest_date(
        model,
        days=10,
        window=60):

    scaled = scaler.transform(
        df[['Close']]
    )

    current = scaled[
        -window:
    ].copy()

    predictions = []

    for _ in range(days):

        pred = model.predict(
            current.reshape(
                1,
                window,
                1
            ),
            verbose=0
        )

        predictions.append(
            pred[0][0]
        )

        current = np.vstack(
            [current[1:], pred]
        )

    predictions = scaler.inverse_transform(
        np.array(predictions)
        .reshape(-1,1)
    )

    return predictions.flatten()


def get_index_for_date(
        selected_date):

    selected_date = pd.to_datetime(
        selected_date
    )

    if selected_date in df.index:

        return df.index.get_loc(
            selected_date
        )

    earlier_dates = df.index[
        df.index <= selected_date
    ]

    if len(
        earlier_dates
    ) == 0:

        raise ValueError(
            "Selected date is before dataset start."
        )

    return len(
        earlier_dates
    ) - 1


def forecast_from_date(
        model,
        selected_date,
        days=10,
        window=60):

    idx = get_index_for_date(
        selected_date
    )

    if idx < window:

        raise ValueError(

            f"Need at least "
            f"{window} days "
            f"before selected date."
        )

    scaled = scaler.transform(
        df[['Close']]
    )

    current = scaled[
        idx-window+1:
        idx+1
    ].copy()

    predictions = []

    for _ in range(days):

        pred = model.predict(
            current.reshape(
                1,
                window,
                1
            ),
            verbose=0
        )

        predictions.append(
            pred[0][0]
        )

        current = np.vstack(
            [current[1:], pred]
        )

    predictions = scaler.inverse_transform(
        np.array(predictions)
        .reshape(-1,1)
    )

    return predictions.flatten()


def get_horizon_predictions(
        model,
        selected_date,
        window=60):

    preds = forecast_from_date(
        model,
        selected_date,
        days=10,
        window=window
    )

    return {

        "1 Day":
        float(preds[0]),

        "5 Day":
        float(preds[4]),

        "10 Day":
        float(preds[9])
    }


def create_forecast_dataframe(
        predictions):

    df_pred = pd.DataFrame({

        "Forecast Horizon":
        predictions.keys(),

        "Predicted Price":
        predictions.values()
    })

    return df_pred


def prediction_interval(
        prediction,
        pct=5):

    lower = prediction * (
        1 - pct/100
    )

    upper = prediction * (
        1 + pct/100
    )

    return (

        round(lower,2),

        round(upper,2)
    )


def compare_models(
        rnn_model,
        lstm_model,
        selected_date):

    rnn_preds = get_horizon_predictions(
        rnn_model,
        selected_date
    )

    lstm_preds = get_horizon_predictions(
        lstm_model,
        selected_date
    )

    comparison = pd.DataFrame({

        "Horizon":[

            "1 Day",

            "5 Day",

            "10 Day"
        ],

        "RNN":[

            rnn_preds["1 Day"],

            rnn_preds["5 Day"],

            rnn_preds["10 Day"]
        ],

        "LSTM":[

            lstm_preds["1 Day"],

            lstm_preds["5 Day"],

            lstm_preds["10 Day"]
        ]
    })

    return comparison

def forecast_to_csv(
        forecast_df):

    return forecast_df.to_csv(
        index=False
    ).encode(
        "utf-8"
    )



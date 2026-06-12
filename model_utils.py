import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LSTM,
    SimpleRNN
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)


# =====================================================
# DATA PREPROCESSING
# =====================================================

def preprocess_data(df):

    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])

    df.sort_values(
        by="Date",
        inplace=True
    )

    df.set_index(
        "Date",
        inplace=True
    )

    # Moving Averages

    df["MA10"] = (
        df["Close"]
        .rolling(10)
        .mean()
    )

    df["MA50"] = (
        df["Close"]
        .rolling(50)
        .mean()
    )

    df["MA100"] = (
        df["Close"]
        .rolling(100)
        .mean()
    )

    df.dropna(inplace=True)

    return df

def create_sequences(
        data,
        window=60):

    X = []
    y = []

    for i in range(
            window,
            len(data)):

        X.append(
            data[i-window:i]
        )

        y.append(
            data[i]
        )

    return (
        np.array(X),
        np.array(y)
    )


def prepare_data(
        df,
        target_col="Close",
        window=60,
        test_size=0.2):

    data = df[[target_col]]

    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(
        data
    )

    split_idx = int(
        len(scaled_data)
        * (1-test_size)
    )

    train = scaled_data[:split_idx]

    test = scaled_data[split_idx:]

    X_train, y_train = create_sequences(
        train,
        window
    )

    X_test, y_test = create_sequences(
        test,
        window
    )

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        scaler,
        scaled_data
    )


def build_rnn(
        window,
        units=50,
        dropout=0.2):

    model = Sequential()

    model.add(
        SimpleRNN(
            units,
            input_shape=(window,1)
        )
    )

    model.add(
        Dropout(dropout)
    )

    model.add(
        Dense(1)
    )

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model


def build_lstm(
        window,
        units=50,
        dropout=0.2):

    model = Sequential()

    model.add(
        LSTM(
            units,
            return_sequences=True,
            input_shape=(window,1)
        )
    )

    model.add(
        Dropout(dropout)
    )

    model.add(
        LSTM(units)
    )

    model.add(
        Dropout(dropout)
    )

    model.add(
        Dense(1)
    )

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model


def train_model(
        model,
        X_train,
        y_train,
        epochs=50,
        batch_size=32):

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    return history


def evaluate_model(
        model,
        X_test,
        y_test,
        scaler):

    predictions = model.predict(
        X_test
    )

    predictions = scaler.inverse_transform(
        predictions
    )

    actual = scaler.inverse_transform(
        y_test.reshape(-1,1)
    )

    mae = mean_absolute_error(
        actual,
        predictions
    )

    mse = mean_squared_error(
        actual,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        actual,
        predictions
    )

    metrics = {

        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2

    }

    return (
        actual,
        predictions,
        metrics
    )


def save_models(
        rnn_model,
        lstm_model):

    rnn_model.save(
        "models/rnn_model.keras"
    )

    lstm_model.save(
        "models/lstm_model.keras"
    )


def load_saved_models():

    rnn_model = load_model(
        "models/rnn_model.keras"
    )

    lstm_model = load_model(
        "models/lstm_model.keras"
    )

    return (
        rnn_model,
        lstm_model
    )


def save_artifacts(
        scaler,
        metrics,
        processed_df):

    joblib.dump(
        scaler,
        "artifacts/scaler.pkl"
    )

    joblib.dump(
        metrics,
        "artifacts/metrics.pkl"
    )

    joblib.dump(
        processed_df,
        "artifacts/processed_df.pkl"
    )

    
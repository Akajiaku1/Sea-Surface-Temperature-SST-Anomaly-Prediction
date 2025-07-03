# 🌊 Sea Surface Temperature (SST) Anomaly Prediction using LSTM

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ------------------ Load Dataset ------------------
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['date'])
    df.set_index('date', inplace=True)
    return df

# ------------------ Preprocessing ------------------
def preprocess_data(df, look_back=12):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    X, y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:i+look_back])
        y.append(scaled_data[i+look_back])
    return np.array(X), np.array(y), scaler

# ------------------ Model ------------------
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# ------------------ Plot ------------------
def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="Actual SST")
    plt.plot(y_pred, label="Predicted SST")
    plt.title("Sea Surface Temperature Anomaly Prediction")
    plt.xlabel("Time Step")
    plt.ylabel("Normalized SST")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------ Main ------------------
def main():
    # Replace with your path to the dataset
    file_path = "sst_data.csv"
    if not os.path.exists(file_path):
        print("ERROR: File sst_data.csv not found.")
        return

    df = load_data(file_path)
    print("Dataset loaded successfully.")

    X, y, scaler = preprocess_data(df, look_back=12)
    print("Data preprocessing completed.")

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_lstm_model(X_train.shape[1:])
    model.summary()
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=1)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test MSE: {mse:.6f}")

    plot_predictions(y_test, y_pred)

    os.makedirs("models", exist_ok=True)
    model.save("models/lstm_sst_model.h5")
    print("Model saved to models/lstm_sst_model.h5")

# ------------------ Entry Point ------------------
if __name__ == "__main__":
    main()


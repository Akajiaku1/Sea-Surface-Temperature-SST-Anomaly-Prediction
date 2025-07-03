import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from src.utils import load_data

X_train, y_train, X_val, y_val = load_data()

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.TimeDistributed(layers.Flatten()),
    layers.LSTM(64, return_sequences=False),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val))
model.save('models/cnn_lstm_model.h5')

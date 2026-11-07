"""
lstm_autoencoder.py  (optional sequence model)
----------------------------------------------
The design documents specify an LSTM autoencoder as the deep temporal model.
This module provides that implementation using TensorFlow/Keras. It is kept
optional so the main pipeline runs with no heavy dependencies; the portable
dense Autoencoder in models.py is used by default and produced the sample
figures. Install TensorFlow (see requirements.txt) to use this variant.

Usage sketch:
    from src.lstm_autoencoder import LSTMAEModel
    m = LSTMAEModel(window=12, n_features=7)
    m.fit(X_train_normal_windows)          # shape (samples, window, features)
    scores = m.score(X_test_windows)       # reconstruction error per window
"""

import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    _TF = True
except Exception:
    _TF = False


def make_windows(X, station_ids, window=12):
    """Turn per-station rows into overlapping sequence windows."""
    seqs = []
    for st in np.unique(station_ids):
        rows = X[station_ids == st]
        for i in range(len(rows) - window + 1):
            seqs.append(rows[i:i + window])
    return np.asarray(seqs)


class LSTMAEModel:
    name = "LSTM Autoencoder"

    def __init__(self, window=12, n_features=7, latent=8, seed=42):
        if not _TF:
            raise ImportError("TensorFlow is required for the LSTM autoencoder. "
                              "Install it with: pip install tensorflow")
        tf.random.set_seed(seed)
        inp = layers.Input(shape=(window, n_features))
        enc = layers.LSTM(latent, activation="tanh")(inp)
        dec = layers.RepeatVector(window)(enc)
        dec = layers.LSTM(latent, activation="tanh", return_sequences=True)(dec)
        out = layers.TimeDistributed(layers.Dense(n_features))(dec)
        self.model = Model(inp, out)
        self.model.compile(optimizer="adam", loss="mse")

    def fit(self, X_windows, epochs=15, batch_size=128):
        self.model.fit(X_windows, X_windows, epochs=epochs,
                       batch_size=batch_size, verbose=0)

    def score(self, X_windows):
        recon = self.model.predict(X_windows, verbose=0)
        return np.mean((X_windows - recon) ** 2, axis=(1, 2))

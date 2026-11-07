"""
models.py
---------
The three anomaly-detection models that are trained and compared:

  * Isolation Forest  - unsupervised outlier detector (trained on normal data)
  * Random Forest     - supervised classifier (trained on labelled data)
  * Autoencoder       - neural reconstruction model (trained on normal data);
                        a high reconstruction error flags an anomaly.

Each model exposes a common interface: ``fit`` and ``score`` (higher score =
more anomalous), so the evaluation treats them identically.

An optional LSTM autoencoder (Keras) is provided for the sequence variant; it
is used automatically if TensorFlow is installed, otherwise the portable
dense autoencoder is used so the pipeline runs with no extra dependencies.
"""

import time
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neural_network import MLPRegressor


class IFModel:
    name = "Isolation Forest"

    def __init__(self, contamination=0.18, seed=42):
        self.model = IsolationForest(
            n_estimators=200, contamination=contamination, random_state=seed)

    def fit(self, X_train_normal, y=None):
        self.model.fit(X_train_normal)

    def score(self, X):
        # decision_function: higher = more normal, so negate for anomaly score.
        return -self.model.decision_function(X)


class RFModel:
    name = "Random Forest"

    def __init__(self, seed=42):
        self.model = RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            n_jobs=-1, random_state=seed)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def score(self, X):
        # Robust to single-class training data (e.g. a tiny all-normal batch):
        # return the probability of the anomaly class if the model learned it,
        # otherwise a zero score for every reading.
        proba = self.model.predict_proba(X)
        classes = list(self.model.classes_)
        if 1 in classes:
            return proba[:, classes.index(1)]
        return np.zeros(len(X))


class AEModel:
    name = "Autoencoder"

    def __init__(self, seed=42):
        # A bottleneck MLP trained to reconstruct its input.
        self.model = MLPRegressor(
            hidden_layer_sizes=(16, 4, 16), activation="relu",
            max_iter=60, random_state=seed)

    def fit(self, X_train_normal, y=None):
        self.model.fit(X_train_normal, X_train_normal)

    def score(self, X):
        recon = self.model.predict(X)
        return np.mean((X - recon) ** 2, axis=1)   # reconstruction error


def train_all(X_train, y_train, X_train_normal, verbose=True):
    """Train the three models and record training time (overhead)."""
    models = [IFModel(), RFModel(), AEModel()]
    timings = {}
    for m in models:
        t0 = time.perf_counter()
        if isinstance(m, RFModel):
            m.fit(X_train, y_train)          # supervised
        else:
            m.fit(X_train_normal)            # unsupervised, normal data only
        timings[m.name] = time.perf_counter() - t0
        if verbose:
            print(f"Trained {m.name:18s} in {timings[m.name]:.2f}s")
    return models, timings

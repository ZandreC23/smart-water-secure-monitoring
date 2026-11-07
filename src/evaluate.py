"""
evaluate.py
-----------
Evaluates each trained model on the held-out test set using metrics that are
meaningful under class imbalance: precision, recall, F1, and the
precision-recall AUC (average precision). Also computes the confusion matrix,
the per-attack detection rate, and inference latency.
"""

import time
import numpy as np
import pandas as pd
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, confusion_matrix,
                             precision_recall_curve)


def choose_threshold(scores_train_normal, scores_test, y_test):
    """
    Pick a decision threshold from the normal training scores
    (95th percentile), a deployment-realistic rule that needs no test labels.
    """
    return np.quantile(scores_train_normal, 0.95)


def evaluate_model(model, X_train_normal, X_test, y_test, attack_types):
    """Return a dict of metrics plus arrays needed for plotting."""
    # Latency: time to score the full test set, expressed per 1,000 samples.
    t0 = time.perf_counter()
    scores = model.score(X_test)
    latency_ms = 1000.0 * (time.perf_counter() - t0) / len(X_test) * 1000

    train_scores = model.score(X_train_normal)
    thr = choose_threshold(train_scores, scores, y_test)
    y_pred = (scores >= thr).astype(int)

    metrics = {
        "model": model.name,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_test, scores),
        "latency_ms_per_1k": latency_ms,
    }
    cm = confusion_matrix(y_test, y_pred)

    # Per-attack detection rate: recall within each injected attack type.
    per_attack = {}
    for atk in ["quality", "fdi", "spoof", "replay", "drift"]:
        mask = (attack_types == atk)
        if mask.sum() > 0:
            per_attack[atk] = float(np.mean(y_pred[mask] == 1))
    prec, rec, _ = precision_recall_curve(y_test, scores)

    return {"metrics": metrics, "cm": cm, "per_attack": per_attack,
            "pr_curve": (rec, prec), "scores": scores, "y_pred": y_pred}


def summarise(results):
    """Build a tidy metrics table across all models."""
    rows = [r["metrics"] for r in results]
    df = pd.DataFrame(rows)
    df = df[["model", "precision", "recall", "f1", "pr_auc",
             "latency_ms_per_1k"]]
    return df.round(3)

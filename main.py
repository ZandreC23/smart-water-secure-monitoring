"""
main.py
-------
End-to-end pipeline for the Lightweight Secure Communication Framework for
Smart Water Monitoring Systems (AI and Data Science capstone).

Run:  python main.py --data path/to/Ponds_data.csv

Steps: load & clean -> inject attacks -> chronological split -> scale ->
train three models -> evaluate -> save metrics and figures.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_and_clean, SENSORS
from src.attack_injection import inject_attacks
from src.models import train_all
from src.evaluate import evaluate_model, summarise
from src import visualize as viz


def chronological_split(df, train_frac=0.7):
    """Split each station's timeline: earliest train_frac = train, rest = test."""
    train_parts, test_parts = [], []
    for _, g in df.groupby("station"):
        cut = int(len(g) * train_frac)
        train_parts.append(g.iloc[:cut])
        test_parts.append(g.iloc[cut:])
    train = pd.concat(train_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True)
    return train, test


def main(args):
    os.makedirs(args.results, exist_ok=True)

    # 1-2. Load, clean, and inject attacks.
    df = load_and_clean(args.data)
    df = inject_attacks(df, rate=args.attack_rate, seed=args.seed)

    # 3. Chronological split (no shuffling: honest for time series).
    train, test = chronological_split(df, train_frac=0.7)
    print(f"\nTrain records: {len(train):,}   Test records: {len(test):,}")

    # 4. Scale using training statistics only (no leakage).
    scaler = StandardScaler().fit(train[SENSORS])
    Xtr = scaler.transform(train[SENSORS])
    Xte = scaler.transform(test[SENSORS])
    ytr = train["y"].to_numpy()
    yte = test["y"].to_numpy()
    atk_te = test["attack_type"].to_numpy()

    # Normal-only training data for the unsupervised models.
    Xtr_normal = Xtr[ytr == 0]

    # 5. Train the three models.
    print("\n── Training models ────────────────────────────────────────")
    models, timings = train_all(Xtr, ytr, Xtr_normal)

    # 6. Evaluate.
    print("\n── Evaluating on held-out test set ────────────────────────")
    results = [evaluate_model(m, Xtr_normal, Xte, yte, atk_te) for m in models]
    summary = summarise(results)
    summary["train_time_s"] = [round(timings[m.name], 2) for m in models]
    print("\nPerformance summary:")
    print(summary.to_string(index=False))
    summary.to_csv(f"{args.results}/metrics_summary.csv", index=False)

    # 7. Visualisations (mockups).
    print("\n── Generating figures ─────────────────────────────────────")
    viz.data_overview(df, SENSORS, args.results)
    viz.confusion_matrices(results, args.results)
    viz.pr_curves(results, args.results)
    viz.metric_bars(summary, args.results)
    viz.per_attack(results, args.results)
    viz.latency_bar(summary, args.results)
    best = max(results, key=lambda r: r["metrics"]["f1"])
    viz.detection_timeline(test, best, args.results)
    print(f"Saved metrics + 7 figures to '{args.results}/'")
    print(f"\nBest model by F1: {best['metrics']['model']} "
          f"(F1={best['metrics']['f1']:.3f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="Ponds_data.csv", help="path to dataset CSV")
    p.add_argument("--results", default="results", help="output folder")
    p.add_argument("--attack_rate", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())

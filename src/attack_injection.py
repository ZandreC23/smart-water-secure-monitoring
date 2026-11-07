"""
attack_injection.py
-------------------
Injects realistic communication attacks onto normal sensor readings to
emulate security incidents and to create ground-truth labels for the
security scenario. Four attack types are supported:

  * FDI    - False Data Injection: values pushed outside plausible bounds
  * SPOOF  - Spoofing: a sudden step change to a fixed off-value
  * REPLAY - Replay: a previous reading is repeated (stale data)
  * DRIFT  - Drift: a slow cumulative deviation over consecutive samples

Each injected row is tagged with its attack type and marked anomalous, so
detection performance can be reported per attack.
"""

import numpy as np
from .data_loader import SENSORS

ATTACKS = ["fdi", "spoof", "replay", "drift"]


def inject_attacks(df, rate=0.15, seed=42):
    """
    Inject attacks onto a fraction of the *normal* rows of ``df``.

    Returns a copy with two new columns:
      * y          - final target: 1 if quality anomaly OR injected attack
      * attack_type- one of {normal, quality, fdi, spoof, replay, drift}
    """
    rng = np.random.default_rng(seed)
    df = df.copy().reset_index(drop=True)

    # Baseline target and tag from the existing quality label.
    df["y"] = df["label"].astype(int)
    df["attack_type"] = np.where(df["label"] == 1, "quality", "normal")

    # Candidate rows = currently-normal readings only.
    normal_idx = df.index[df["label"] == 0].to_numpy()
    n_inject = int(len(normal_idx) * rate)
    chosen = rng.choice(normal_idx, size=n_inject, replace=False)

    # Plausible per-sensor bounds (used by the FDI/spoof attacks).
    lo = df[SENSORS].quantile(0.01)
    hi = df[SENSORS].quantile(0.99)

    for i in chosen:
        attack = rng.choice(ATTACKS)
        row = df.loc[i, SENSORS].astype(float).copy()

        if attack == "fdi":
            # Push 2-4 sensors far outside their plausible range.
            for col in rng.choice(SENSORS, size=rng.integers(2, 5), replace=False):
                span = hi[col] - lo[col]
                row[col] = hi[col] + rng.uniform(0.5, 2.0) * span

        elif attack == "spoof":
            # Replace all sensors with a fixed, implausible off-value.
            for col in SENSORS:
                row[col] = hi[col] * rng.uniform(1.2, 1.6)

        elif attack == "replay":
            # Repeat the reading from ~10 samples earlier (stale data).
            j = max(i - 10, 0)
            row = df.loc[j, SENSORS].astype(float).copy()

        elif attack == "drift":
            # Apply a cumulative multiplicative drift.
            factor = 1.0 + rng.uniform(0.25, 0.6)
            row = row * factor

        df.loc[i, SENSORS] = row.values
        df.loc[i, "y"] = 1
        df.loc[i, "attack_type"] = attack

    print("── Attack injection ───────────────────────────────────────")
    print(f"Rows injected with attacks    : {n_inject:,}")
    for a in ATTACKS:
        print(f"  {a:8s}                    : {int((df['attack_type']==a).sum()):,}")
    print(f"Final anomaly rate (combined) : {100*df['y'].mean():.1f}%")
    print("───────────────────────────────────────────────────────────")
    return df

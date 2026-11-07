"""
data_loader.py
--------------
Loads the raw IoT water-quality dataset and performs the cleaning steps
described in the Data Processing Workflow: fixing mixed data types,
handling missing values, treating physically impossible sensor readings,
removing duplicates, and dropping unlabeled records.
"""

import pandas as pd
import numpy as np

# The seven sensor features used by every model.
SENSORS = ["NITRATE(PPM)", "PH", "AMMONIA(mg/l)", "TEMP",
           "DO", "TURBIDITY", "MANGANESE(mg/l)"]

# Physically impossible values that indicate a sensor dropout rather than a
# true reading (a dissolved-oxygen or temperature of exactly zero).
IMPOSSIBLE_ZERO = ["DO", "TEMP"]


def load_and_clean(path, verbose=True):
    """Load the CSV and return a cleaned, chronologically ordered DataFrame."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    n_raw = len(df)

    # 1. Fix mixed data types: coerce sensor columns to numeric.
    #    Non-numeric text becomes NaN and is handled as missing below.
    junk = 0
    for col in SENSORS:
        coerced = pd.to_numeric(df[col], errors="coerce")
        junk += int((coerced.isna() & df[col].notna()).sum())
        df[col] = coerced

    # 2. Treat physically impossible zeros as missing (sensor dropouts).
    zero_fixed = 0
    for col in IMPOSSIBLE_ZERO:
        mask = df[col] == 0
        zero_fixed += int(mask.sum())
        df.loc[mask, col] = np.nan

    # 3. Remove duplicate rows.
    dupes = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    # 4. Drop rows with no label (cannot be used for supervised evaluation).
    no_label = int(df["label"].isna().sum())
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)

    # 5. Build a timestamp and order chronologically within each station.
    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        errors="coerce", dayfirst=True)
    df = df.sort_values(["station", "timestamp"]).reset_index(drop=True)

    # 6. Impute remaining missing sensor values by per-station interpolation
    #    (short gaps) with a forward/backward fill fallback.
    missing_before = int(df[SENSORS].isna().sum().sum())
    df[SENSORS] = (df.groupby("station")[SENSORS]
                     .transform(lambda g: g.interpolate(limit_direction="both")))
    df[SENSORS] = df[SENSORS].ffill().bfill()

    if verbose:
        print("── Data loading and cleaning ──────────────────────────────")
        print(f"Raw records loaded            : {n_raw:,}")
        print(f"Non-numeric cells coerced     : {junk}")
        print(f"Impossible zeros -> missing   : {zero_fixed}")
        print(f"Duplicate rows removed        : {dupes}")
        print(f"Unlabeled rows dropped        : {no_label}")
        print(f"Missing sensor cells imputed  : {missing_before}")
        print(f"Clean records remaining       : {len(df):,}")
        print(f"Stations                      : {df['station'].nunique()}")
        print(f"Anomaly rate (quality label)  : {100*df['label'].mean():.1f}%")
        print("───────────────────────────────────────────────────────────")

    return df

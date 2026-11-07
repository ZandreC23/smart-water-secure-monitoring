"""
pipeline.py
-----------
The integrated runtime that fuses the two defence layers into a single
service (Design Phase objective O5 / requirement REQ-08). A reading is passed
through the secure channel first (encryption, authentication, replay
protection); only if it is delivered intact does it reach the anomaly-detection
model, which scores it and decides normal vs anomalous. This is the end-to-end
path the dashboard drives.

process() returns a single verdict dict for one reading, recording which stage
made the decision, so the two layers act as one pipeline rather than separate
stages.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .secure_channel import SecureChannel
from .data_loader import load_and_clean, SENSORS
from .attack_injection import inject_attacks
from .models import RFModel


class SecureMonitoringPipeline:
    def __init__(self, scaler, model, threshold, channel=None):
        self.scaler = scaler
        self.model = model
        self.threshold = threshold
        self.channel = channel or SecureChannel()

    def process(self, reading: dict, packet: bytes = None) -> dict:
        """
        Run one reading through the full pipeline.

        Returns a verdict dict:
          delivered   - did it pass the secure channel?
          stage       - 'secure_channel' or 'detector'
          verdict     - 'rejected', 'anomaly', or 'normal'
          reason      - human-readable explanation
          score       - anomaly score (None if rejected at the channel)
        """
        # Layer 1: secure channel (encrypt on send, verify on receive).
        if packet is None:
            packet = self.channel.send(reading)
        try:
            received = self.channel.receive(packet)
        except ValueError as e:
            return {"delivered": False, "stage": "secure_channel",
                    "verdict": "rejected", "reason": str(e), "score": None}

        # Layer 2: anomaly detection on the delivered reading.
        x = np.array([[float(received[s]) for s in SENSORS]])
        xs = self.scaler.transform(x)
        score = float(self.model.score(xs)[0])
        is_anom = score >= self.threshold
        return {"delivered": True, "stage": "detector",
                "verdict": "anomaly" if is_anom else "normal",
                "reason": "anomalous reading flagged" if is_anom else "reading accepted",
                "score": score}


def build_pipeline(data_path, attack_rate=0.15, seed=42):
    """
    Build a ready-to-use pipeline from the dataset: clean, inject attacks,
    scale, train a Random Forest, choose a threshold, and wire everything
    into a SecureMonitoringPipeline. Returns (pipeline, test_df, info).
    """
    df = inject_attacks(load_and_clean(data_path, verbose=False),
                        rate=attack_rate, seed=seed)
    # chronological split per station
    tr, te = [], []
    for _, g in df.groupby("station"):
        cut = int(len(g) * 0.7)
        tr.append(g.iloc[:cut]); te.append(g.iloc[cut:])
    train = pd.concat(tr).reset_index(drop=True)
    test = pd.concat(te).reset_index(drop=True)

    scaler = StandardScaler().fit(train[SENSORS].to_numpy())
    Xtr = scaler.transform(train[SENSORS].to_numpy()); ytr = train["y"].to_numpy()
    model = RFModel(); model.fit(Xtr, ytr)
    threshold = 0.5  # Random Forest probability threshold

    pipe = SecureMonitoringPipeline(scaler, model, threshold)
    info = {"train_n": len(train), "test_n": len(test),
            "anomaly_rate": float(df["y"].mean())}
    return pipe, test, info

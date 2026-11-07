"""
test_capstone.py
----------------
Automated test suite for the Lightweight Secure Communication Framework.
Each test is traced to a requirement ID from the Design Phase (REQ-01..08)
and is referenced in the Test Plan of the Implementation & Testing report.

Run (no external dependency needed):
    python -m unittest -v tests.test_capstone
"""

import time
import unittest
import numpy as np
import pandas as pd

from src.secure_channel import SecureChannel
from src.data_loader import SENSORS
from src.attack_injection import inject_attacks, ATTACKS
from src.models import IFModel, RFModel, AEModel


def _sample_df(n=1500, seed=0):
    """A small, realistic normal dataset so tests run fast."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "NITRATE(PPM)": rng.normal(38, 10, n).clip(0),
        "PH": rng.normal(6.5, 0.5, n).clip(4, 9),
        "AMMONIA(mg/l)": rng.normal(0.1, 0.05, n).clip(0),
        "TEMP": rng.normal(27, 3, n).clip(5, 45),
        "DO": rng.normal(10, 2, n).clip(0),
        "TURBIDITY": rng.normal(31, 8, n).clip(10),
        "MANGANESE(mg/l)": rng.normal(1.0, 0.3, n).clip(0),
        "label": 0,
    })


class SecureChannelTests(unittest.TestCase):
    """REQ-02  Confidentiality, integrity, and replay protection."""

    def test_TC01_secure_roundtrip(self):
        """TC-01 / REQ-02: a reading survives encrypt -> decrypt unchanged."""
        ch = SecureChannel()
        reading = {"station": "s1", "PH": 6.8, "DO": 9.1}
        self.assertEqual(ch.receive(ch.send(reading)), reading)

    def test_TC02_tamper_rejected(self):
        """TC-02 / REQ-02: a tampered packet is rejected."""
        ch = SecureChannel()
        packet = bytearray(ch.send({"PH": 7.0}))
        packet[10] ^= 0x01
        with self.assertRaises(ValueError):
            ch.receive(bytes(packet))

    def test_TC03_replay_rejected(self):
        """TC-03 / REQ-02: resending a valid packet (replay) is rejected."""
        ch = SecureChannel()
        packet = ch.send({"PH": 7.0})
        ch.receive(packet)
        with self.assertRaises(ValueError):
            ch.receive(packet)

    def test_TC04_stale_packet_rejected(self):
        """TC-04 / REQ-02: a packet older than the freshness window is rejected."""
        ch = SecureChannel(freshness_window=0.0001)
        packet = ch.send({"PH": 7.0})
        time.sleep(0.01)
        with self.assertRaises(ValueError):
            ch.receive(packet)


class AttackInjectionTests(unittest.TestCase):
    """REQ-04 / REQ-05  Attack generation and data handling."""

    def test_TC05_attack_injection_tags(self):
        """TC-05 / REQ-04: injection tags every attack type and sets labels."""
        out = inject_attacks(_sample_df(), rate=0.2, seed=1)
        self.assertTrue(set(out.loc[out.y == 1, "attack_type"]).issuperset(set(ATTACKS)))
        self.assertTrue((out.loc[out.attack_type.isin(ATTACKS), "y"] == 1).all())

    def test_TC06_injection_leaves_normals(self):
        """TC-06 / REQ-05: non-injected rows remain labelled normal."""
        out = inject_attacks(_sample_df(), rate=0.2, seed=1)
        self.assertTrue((out.loc[out.attack_type == "normal", "y"] == 0).all())


class ModelTests(unittest.TestCase):
    """REQ-03  Anomaly-detection models."""

    def test_TC07_models_score_range(self):
        """TC-07 / REQ-03: every model returns one finite score per reading."""
        out = inject_attacks(_sample_df(), rate=0.2, seed=2)
        X = out[SENSORS].to_numpy(); y = out["y"].to_numpy(); Xn = X[y == 0]
        for Model in (IFModel, RFModel, AEModel):
            m = Model()
            m.fit(X, y) if Model is RFModel else m.fit(Xn)
            s = m.score(X)
            self.assertEqual(len(s), len(X))
            self.assertTrue(np.all(np.isfinite(s)))

    def test_TC08_random_forest_detects(self):
        """TC-08 / REQ-03: Random Forest separates attacks from normal."""
        out = inject_attacks(_sample_df(), rate=0.25, seed=3)
        X = out[SENSORS].to_numpy(); y = out["y"].to_numpy()
        m = RFModel(); m.fit(X, y); s = m.score(X)
        self.assertGreater(s[y == 1].mean(), s[y == 0].mean())


class EdgeCaseTests(unittest.TestCase):
    """Edge cases (avoids happy-path-only testing)."""

    def test_TC09_extreme_values_no_crash(self):
        """TC-09 / edge: extreme out-of-range values do not crash scoring."""
        out = inject_attacks(_sample_df(), rate=0.1, seed=4)
        X = out[SENSORS].to_numpy().copy(); y = out["y"].to_numpy()
        X[0] = X[0] * 1e6
        m = IFModel(); m.fit(X[y == 0]); s = m.score(X)
        self.assertTrue(np.all(np.isfinite(s)))

    def test_TC10_single_row_handled(self):
        """TC-10 / edge: a single-row batch is scored without error."""
        df = pd.DataFrame({c: [1.0] for c in SENSORS}); df["label"] = 0
        out = inject_attacks(df, rate=0.0, seed=5)
        m = RFModel(); m.fit(out[SENSORS].to_numpy(), out["y"].to_numpy())
        self.assertEqual(len(m.score(out[SENSORS].to_numpy())), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class IntegrationTests(unittest.TestCase):
    """REQ-08 / O5  End-to-end fusion of the secure channel and the detector."""

    @classmethod
    def setUpClass(cls):
        # Build the integrated pipeline once from a small fixture so the
        # end-to-end path (channel -> detector) can be exercised directly.
        from sklearn.preprocessing import StandardScaler
        from src.pipeline import SecureMonitoringPipeline
        df = inject_attacks(_sample_df(2000, seed=7), rate=0.25, seed=7)
        X = df[SENSORS].to_numpy(); y = df["y"].to_numpy()
        scaler = StandardScaler().fit(X)
        m = RFModel(); m.fit(scaler.transform(X), y)
        cls.pipe = SecureMonitoringPipeline(scaler, m, threshold=0.5)
        cls.normal = {s: float(df[df.y == 0][s].iloc[0]) for s in SENSORS}

    def test_TC11_endtoend_normal_delivered(self):
        """TC-11 / REQ-08: a clean reading passes the channel and is scored."""
        v = self.pipe.process(self.normal)
        self.assertTrue(v["delivered"])
        self.assertEqual(v["stage"], "detector")
        self.assertIn(v["verdict"], ("normal", "anomaly"))

    def test_TC12_endtoend_tampered_blocked(self):
        """TC-12 / REQ-08: a tampered packet is blocked at the channel and
        never reaches the detector (defence in depth)."""
        pkt = bytearray(self.pipe.channel.send(self.normal)); pkt[10] ^= 0x01
        v = self.pipe.process(self.normal, packet=bytes(pkt))
        self.assertFalse(v["delivered"])
        self.assertEqual(v["stage"], "secure_channel")
        self.assertEqual(v["verdict"], "rejected")

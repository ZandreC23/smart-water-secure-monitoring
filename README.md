# Lightweight Secure Communication Framework for Smart Water Monitoring Systems

**Capstone project — Artificial Intelligence and Data Science stream**
Zandre Coetzee (222533161), Cape Peninsula University of Technology

A prototype that simulates IoT water sensors sending readings to a municipal
control server over a secure channel, and uses AI anomaly detection to flag
abnormal or falsified data. It cleans a real IoT water quality dataset, injects
realistic communication attacks, and trains and compares three anomaly
detection models.

## What it does

1. **Loads and cleans** a real IoT water quality dataset (74,758 usable records,
   3 stations, 7 sensors): fixes mixed data types, imputes missing values,
   treats impossible sensor readings, and removes duplicates.
2. **Injects attacks** onto normal readings — false data injection, spoofing,
   replay, and drift — each tagged for per attack evaluation.
3. **Secures the channel** (see `src/secure_channel.py`): each reading is
   encrypted and authenticated, and tampering is detected on arrival.
4. **Trains three models**: Isolation Forest (unsupervised), Random Forest
   (supervised), and an Autoencoder (neural reconstruction). An optional LSTM
   autoencoder is provided in `src/lstm_autoencoder.py`.
5. **Evaluates** with imbalance aware metrics (precision, recall, F1, PR-AUC),
   confusion matrices, per attack detection rate, and inference latency.
6. **Produces figures** in `results/` for the report.

## How to run

```bash
pip install -r requirements.txt
python main.py --data path/to/Ponds_data.csv
```

Outputs are written to `results/` (a metrics table and seven figures).
To see the secure channel demonstration on its own:

```bash
python -m src.secure_channel
```

## Sample results (real data, seed 42)

| Model            | Precision | Recall | F1   | PR-AUC | Latency (ms/1k) |
|------------------|-----------|--------|------|--------|-----------------|
| Isolation Forest | 0.76      | 0.59   | 0.66 | 0.79   | ~18.0           |
| Random Forest    | 0.74      | 0.92   | 0.82 | 0.83   | ~10.0           |
| Autoencoder      | 0.77      | 0.59   | 0.67 | 0.78   | ~0.6            |

Random Forest gives the best detection (highest F1 and recall); the
Autoencoder is the most lightweight at inference. Exact numbers vary slightly
by machine and library version.


## Live dashboard

A Streamlit dashboard drives the integrated runtime (`src/pipeline.py`): each reading flows through the secure channel and, if delivered, on to the anomaly detector. It streams readings live, lets you launch tamper/replay/FDI/spoof attacks and watch them handled, and shows the metrics, per-attack detection, and replay before/after.

```
pip install streamlit
streamlit run dashboard.py
```

## Testing

An automated test suite (Python `unittest`, no extra install) covers the secure channel (including replay protection), attack injection, the models, and edge cases. Each test is traced to a requirement ID.

```
python -m unittest -v tests.test_capstone
```

All ten test cases pass. See the Implementation & Testing report for the test plan, results, and defect log.

## Project structure

```
capstone_prototype/
├── main.py                    end-to-end pipeline
├── requirements.txt
├── README.md
├── results/                   generated metrics + figures
└── src/
    ├── data_loader.py         load and clean the dataset
    ├── attack_injection.py    inject FDI / spoof / replay / drift
    ├── secure_channel.py      encryption + authentication demo
    ├── models.py              Isolation Forest, Random Forest, Autoencoder
    ├── lstm_autoencoder.py    optional LSTM autoencoder (needs TensorFlow)
    ├── evaluate.py            metrics, thresholds, per attack, latency
    └── visualize.py           result figures
```

## Notes

- The default pipeline uses only scikit-learn, so it runs with no deep learning
  framework installed. The sample figures were produced this way. Install
  TensorFlow and use `src/lstm_autoencoder.py` for the LSTM sequence variant
  described in the design documents.
- The split is chronological (earliest 70% train, latest 30% test) to avoid
  look ahead leakage, which is the honest choice for time series data.
- `cryptography`'s Fernet (AES-128 + HMAC) stands in for a lightweight cipher;
  on a constrained device a lighter algorithm such as PRESENT or SPECK would be
  substituted behind the same interface.
```

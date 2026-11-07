"""
dashboard.py
------------
Security operations dashboard for the Smart Water Monitoring framework.
Every sensor reading flows through two defence layers in one runtime
(src/pipeline.py):

  Layer 1 - Channel security : encryption, authentication, replay protection.
                               Tampered or replayed packets are BLOCKED here.
  Layer 2 - Anomaly detection: an AI model inspects delivered readings and
                               FLAGS abnormal or falsified values.

Real-time processing; the sensor data and attacks are simulated for
demonstration (readings come from a recorded dataset, attacks are injected).

Run:
    pip install streamlit
    streamlit run dashboard.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.pipeline import build_pipeline
from src.secure_channel import SecureChannel
from src.data_loader import SENSORS

st.set_page_config(page_title="Smart Water Security Operations",
                   page_icon="💧", layout="wide")

GREEN="#1e7d45"; RED="#b23b3b"; AMBER="#c77d2e"; NAVY="#0F4761"; GREY="#5A6570"
BLUE="#5aa9e6"; DARK="#0e1117"
DATA_DEFAULT = "Ponds data.csv"


@st.cache_resource(show_spinner="Starting monitoring service…")
def load_everything(path):
    pipe, test, info = build_pipeline(path)
    X = test[SENSORS].to_numpy()
    scores = pipe.model.score(pipe.scaler.transform(X))
    pred = (scores >= pipe.threshold).astype(int)
    per_attack = {}
    for atk in ["quality", "fdi", "spoof", "replay", "drift"]:
        m = (test["attack_type"] == atk).to_numpy()
        if m.sum():
            per_attack[atk] = float((pred[m] == 1).mean())
    ch = SecureChannel()
    pk = [ch.send({"i": i}) for i in range(200)]
    for p in pk: ch.receive(p)
    rej = 0
    for p in pk:
        try: ch.receive(p)
        except ValueError: rej += 1
    replay_channel = rej / len(pk)
    return pipe, test, info, per_attack, replay_channel


def badge(text, color):
    return (f"<span style='background:{color};color:#fff;padding:3px 12px;"
            f"border-radius:12px;font-weight:700;font-size:12px;"
            f"letter-spacing:.5px'>{text}</span>")


def feed_html(rows, kind):
    if not rows:
        return "<div style='color:#888;padding:10px'>No events yet.</div>"
    html = "<div style='font-family:Arial'>"
    for r in rows:
        if kind == "channel":
            col = GREEN if r["status"] == "ALLOWED" else RED
            b = badge(r["status"], col)
        else:
            col = GREEN if r["verdict"] == "NORMAL" else AMBER
            b = badge(r["verdict"], col)
        extra = (f"&nbsp;&nbsp;<span style='color:#888'>score {r['score']:.2f}</span>"
                 if kind == "detector" and r.get("score") is not None else "")
        html += (f"<div style='padding:7px 0;border-bottom:1px solid #2a2a2a'>"
                 f"{b}&nbsp;&nbsp;<b>{r['source']}</b>{extra}<br>"
                 f"<span style='color:#aaa;font-size:13px'>{r['reason']}</span></div>")
    return html + "</div>"


def signal_chart(history, sensor):
    """Live line chart of one sensor with anomalies and blocked attacks marked."""
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(11, 3.3))
    fig.patch.set_facecolor(DARK); ax.set_facecolor(DARK)
    delivered = h[h["status"] != "BLOCKED"]
    ax.plot(delivered["t"], delivered[sensor], color=BLUE, lw=1.4,
            marker="o", markersize=3, label=f"{sensor} (delivered)")
    anom = h[h["status"] == "ANOMALY"]
    if len(anom):
        ax.scatter(anom["t"], anom[sensor], color=AMBER, s=90, zorder=3,
                   edgecolors="white", linewidths=0.6, label="Anomaly flagged (AI)")
    blocked = h[h["status"] == "BLOCKED"]
    if len(blocked):
        ax.scatter(blocked["t"], blocked[sensor], marker="x", color=RED, s=90,
                   zorder=3, linewidths=2, label="Blocked at channel")
    ax.set_xlabel("reading number", color="#cccccc")
    ax.set_ylabel(sensor, color="#cccccc")
    ax.tick_params(colors="#cccccc")
    for sp in ax.spines.values(): sp.set_color("#444")
    leg = ax.legend(loc="upper left", fontsize=8, facecolor=DARK,
                    edgecolor="#444", labelcolor="#dddddd")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    return fig


st.markdown("## 💧 Smart Water Monitoring — Security Operations")
st.markdown("<span style='color:#888'>Real-time processing of sensor readings; "
            "sensor data and attacks are simulated for demonstration</span>",
            unsafe_allow_html=True)

st.sidebar.markdown("### Monitoring service")
path = st.sidebar.text_input("Dataset source", DATA_DEFAULT)

try:
    pipe, test, info, per_attack, replay_channel = load_everything(path)
except Exception as e:
    st.error(f"Could not start the service with source '{path}'. Place the "
             f"Pondsdata CSV in this folder or set the path in the sidebar.\n\n{e}")
    st.stop()

if "channel" not in st.session_state:
    st.session_state.channel = []
    st.session_state.detector = []
    st.session_state.history = []
    st.session_state.idx = 0
    st.session_state.counts = {"processed": 0, "delivered": 0, "blocked": 0, "anomaly": 0}
    st.session_state.last_attack = None

def push(reading, packet=None, source="sensor"):
    c = st.session_state.counts
    v = pipe.process(reading, packet=packet)
    c["processed"] += 1
    if not v["delivered"]:
        c["blocked"] += 1
        status = "BLOCKED"
        st.session_state.channel.insert(0, {"source": source, "status": "BLOCKED",
                                            "reason": v["reason"]})
    else:
        c["delivered"] += 1
        st.session_state.channel.insert(0, {"source": source, "status": "ALLOWED",
                                            "reason": "Encrypted, authenticated, fresh"})
        status = "ANOMALY" if v["verdict"] == "anomaly" else "NORMAL"
        if status == "ANOMALY": c["anomaly"] += 1
        st.session_state.detector.insert(0, {"source": source, "verdict": status,
                                             "reason": v["reason"], "score": v["score"]})
    # record the point for the live signal chart
    entry = {s: float(reading[s]) for s in SENSORS}
    entry["t"] = c["processed"]; entry["status"] = status; entry["source"] = source
    st.session_state.history.append(entry)
    st.session_state.history = st.session_state.history[-80:]
    st.session_state.channel = st.session_state.channel[:12]
    st.session_state.detector = st.session_state.detector[:12]
    return v

st.sidebar.markdown("---")
with st.sidebar.expander("🧪 Security testing (simulated)", expanded=True):
    st.caption("Injected attacks for demonstration — not live traffic.")
    normal = {s: float(test[test.y == 0].iloc[0][s]) for s in SENSORS}
    if st.button("Tamper a packet", use_container_width=True):
        pkt = bytearray(pipe.channel.send(normal)); pkt[10] ^= 0x01
        v = push(normal, packet=bytes(pkt), source="tamper attack")
        st.session_state.last_attack = ("tamper", v)
    if st.button("Replay a packet", use_container_width=True):
        good = pipe.channel.send(normal); pipe.channel.receive(good)
        v = push(normal, packet=good, source="replay attack")
        st.session_state.last_attack = ("replay", v)
    if st.button("Inject false data (FDI)", use_container_width=True):
        bad = dict(normal)
        for s in ["NITRATE(PPM)", "TURBIDITY", "DO"]: bad[s] = bad[s] * 5 + 500
        v = push(bad, source="FDI attack")
        st.session_state.last_attack = ("fdi", v)
    if st.button("Inject spoofed reading", use_container_width=True):
        bad = {s: val * 1.5 + 50 for s, val in normal.items()}
        v = push(bad, source="spoof attack")
        st.session_state.last_attack = ("spoof", v)
st.sidebar.markdown("---")
st.sidebar.caption("Secure Smart Water Monitoring framework · integrated runtime "
                   "(secure channel + AI detection).")

c = st.session_state.counts
k1, k2, k3, k4 = st.columns(4)
k1.metric("Readings processed", c["processed"])
k2.metric("Delivered", c["delivered"])
k3.metric("🛡️ Blocked at channel", c["blocked"])
k4.metric("⚠️ Anomalies flagged", c["anomaly"])

if c["blocked"] or c["anomaly"]:
    st.warning(f"Security events detected: {c['blocked']} packet(s) blocked at the "
               f"channel, {c['anomaly']} reading(s) flagged by the AI detector.")
else:
    st.success("System nominal — no threats detected in processed readings.")

b1, b2, _ = st.columns([1, 1, 3])
if b1.button("▶ Simulate 1 sensor reading"):
    row = test.iloc[st.session_state.idx % len(test)]
    push({s: float(row[s]) for s in SENSORS}, source=f"sensor ({row['attack_type']})")
    st.session_state.idx += 1
if b2.button("⏩ Simulate 25 sensor readings"):
    for _ in range(25):
        row = test.iloc[st.session_state.idx % len(test)]
        push({s: float(row[s]) for s in SENSORS}, source=f"sensor ({row['attack_type']})")
        st.session_state.idx += 1

if st.session_state.last_attack:
    atk, v = st.session_state.last_attack
    if not v["delivered"]:
        st.error(f"🛡️ **{atk.upper()} attack BLOCKED at the secure channel (Layer 1)** — "
                 f"{v['reason']} The reading never reached the AI model.")
    elif v["verdict"] == "anomaly":
        st.warning(f"⚠️ **{atk.upper()} reading FLAGGED by the AI detector (Layer 2)** — "
                   f"anomaly score {v['score']:.2f} ≥ threshold {pipe.threshold:.2f}. "
                   f"The packet was validly transmitted, so Layer 1 allowed it, but "
                   f"its values are implausible so the AI flagged it.")
    else:
        st.info(f"{atk.upper()} reading was delivered and scored normal "
                f"(score {v['score']:.2f}).")

# ---------------- live sensor signal ----------------
st.markdown("### Live sensor signal")
if st.session_state.history:
    plot_options = ["DO", "PH", "TURBIDITY", "TEMP"] = ["DO", "PH", "TURBIDITY", "TEMP"]
    sensor_choice = st.selectbox("Water parameter to plot", plot_options, index=0)
    fig = signal_chart(st.session_state.history, sensor_choice)
    st.pyplot(fig); plt.close(fig)
    st.caption("Blue line = delivered sensor stream. Amber dots = readings the AI "
               "flagged as anomalous. Red crosses = packets blocked at the secure "
               "channel before reaching the AI.")
else:
    st.info("Press ‘Simulate 25 sensor readings’ to see the live signal, then try "
            "the attack buttons in the sidebar.")

# ---------------- two layers ----------------
st.markdown("### Live monitoring")
st.caption("Tamper and replay attacks are stopped at Layer 1 (the secure channel) "
           "and never reach the AI. False-data and spoof attacks are validly "
           "transmitted, so they pass Layer 1 and are caught by Layer 2 (the AI). "
           "This is defence in depth — two layers for two kinds of threat.")
L, R = st.columns(2)
with L:
    st.markdown("#### 🛡️ Layer 1 · Channel security")
    st.caption("Encryption · authentication · replay protection")
    st.markdown(feed_html(st.session_state.channel, "channel"), unsafe_allow_html=True)
with R:
    st.markdown("#### ⚠️ Layer 2 · AI anomaly detection")
    st.caption("Random Forest · flags abnormal or falsified readings")
    st.markdown(feed_html(st.session_state.detector, "detector"), unsafe_allow_html=True)

# ---------------- evaluation ----------------
st.markdown("---")
st.markdown("### Model evaluation (held-out test set)")
st.caption("Measured results from evaluating the trained model on unseen test "
           "data — the same figures reported in the Implementation & Testing report.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Random Forest F1", "0.82")
m2.metric("Precision", "0.74")
m3.metric("Recall", "0.92")
m4.metric("PR-AUC", "0.83")

r1, r2 = st.columns(2)
with r1:
    st.markdown("#### Detection rate by attack type (AI model)")
    st.bar_chart(pd.DataFrame({"detection rate": per_attack}))
with r2:
    st.markdown("#### Replay mitigation: model vs secure channel")
    st.bar_chart(pd.DataFrame({"replay handled": {
        "AI model only": per_attack.get("replay", 0.2),
        "Secure channel": replay_channel}}))
    st.caption(f"The secure channel rejects {replay_channel*100:.0f}% of replayed "
               f"packets, versus ~{per_attack.get('replay',0.2)*100:.0f}% by the "
               f"model alone — the two layers combine for defence in depth.")
"""
visualize.py
------------
Produces the data-visualization mockups: a data overview, per-model confusion
matrices, precision-recall curves, a metric comparison, per-attack detection
rates, latency comparison, and a detection timeline. All figures are saved to
the results/ folder.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"

NAVY = "#0F4761"; TEAL = "#2E7D8A"; GREEN = "#2E7D5B"; RED = "#9C3B3B"
ORANGE = "#C77D2E"; GREY = "#5A6570"
PALETTE = {"Isolation Forest": TEAL, "Random Forest": NAVY, "Autoencoder": ORANGE}


def data_overview(df, sensors, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    # class balance
    counts = df["y"].value_counts().sort_index()
    axes[0, 0].bar(["Normal", "Anomaly"], counts.values, color=[GREEN, RED])
    axes[0, 0].set_title("Class balance (test + train combined)")
    for i, v in enumerate(counts.values):
        axes[0, 0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    # attack composition
    comp = df["attack_type"].value_counts()
    axes[0, 1].barh(comp.index[::-1], comp.values[::-1], color=NAVY)
    axes[0, 1].set_title("Record composition by type")
    # pH distribution normal vs anomaly
    for lab, col, name in [(0, GREEN, "Normal"), (1, RED, "Anomaly")]:
        axes[1, 0].hist(df[df.y == lab]["PH"], bins=40, alpha=0.6,
                        color=col, label=name, density=True)
    axes[1, 0].set_title("pH distribution"); axes[1, 0].legend()
    # DO vs Turbidity scatter sample
    s = df.sample(min(4000, len(df)), random_state=1)
    axes[1, 1].scatter(s["DO"], s["TURBIDITY"], c=np.where(s.y == 1, RED, GREEN),
                       s=6, alpha=0.4)
    axes[1, 1].set_xlabel("Dissolved oxygen"); axes[1, 1].set_ylabel("Turbidity")
    axes[1, 1].set_title("DO vs turbidity (red = anomaly)")
    fig.suptitle("Figure A. Dataset overview", fontweight="bold", color=NAVY)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig_data_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def confusion_matrices(results, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, r in zip(axes, results):
        cm = r["cm"]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(r["metrics"]["model"], color=NAVY, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Anomaly"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Anomaly"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        thr = cm.max() / 2
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > thr else "black", fontsize=10)
    fig.suptitle("Figure B. Confusion matrices", fontweight="bold", color=NAVY)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig_confusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def pr_curves(results, outdir):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for r in results:
        rec, prec = r["pr_curve"]
        ax.plot(rec, prec, label=f"{r['metrics']['model']} "
                f"(AP={r['metrics']['pr_auc']:.2f})",
                color=PALETTE[r["metrics"]["model"]], lw=2)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Figure C. Precision-Recall curves", fontweight="bold", color=NAVY)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def metric_bars(summary, outdir):
    metrics = ["precision", "recall", "f1", "pr_auc"]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(metrics)); w = 0.25
    for i, (_, row) in enumerate(summary.iterrows()):
        ax.bar(x + i * w, [row[m] for m in metrics], w,
               label=row["model"], color=PALETTE[row["model"]])
    ax.set_xticks(x + w); ax.set_xticklabels(["Precision", "Recall", "F1", "PR-AUC"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Figure D. Model performance comparison", fontweight="bold", color=NAVY)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def per_attack(results, outdir):
    attacks = ["quality", "fdi", "spoof", "replay", "drift"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(attacks)); w = 0.25
    for i, r in enumerate(results):
        vals = [r["per_attack"].get(a, 0) for a in attacks]
        ax.bar(x + i * w, vals, w, label=r["metrics"]["model"],
               color=PALETTE[r["metrics"]["model"]])
    ax.set_xticks(x + w)
    ax.set_xticklabels(["Quality", "FDI", "Spoof", "Replay", "Drift"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Detection rate")
    ax.set_title("Figure E. Detection rate by attack type", fontweight="bold", color=NAVY)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig_per_attack.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def latency_bar(summary, outdir):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(summary["model"], summary["latency_ms_per_1k"],
           color=[PALETTE[m] for m in summary["model"]])
    ax.set_ylabel("Latency (ms per 1,000 readings)")
    ax.set_title("Figure F. Inference latency (efficiency)", fontweight="bold", color=NAVY)
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(summary["latency_ms_per_1k"]):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig_latency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def detection_timeline(df_test, best_result, outdir, sensor="DO", n=600):
    """Show a slice of the test stream with true anomalies and detections."""
    sl = df_test.iloc[:n].reset_index(drop=True)
    pred = best_result["y_pred"][:n]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(sl.index, sl[sensor], color=GREY, lw=0.8, label=f"{sensor} reading")
    true_idx = sl.index[sl["y"] == 1]
    ax.scatter(true_idx, sl.loc[true_idx, sensor], color=RED, s=28,
               label="True anomaly", zorder=3)
    det_idx = sl.index[(pred == 1)]
    ax.scatter(det_idx, sl.loc[det_idx, sensor], facecolors="none",
               edgecolors=NAVY, s=80, label="Flagged by model", zorder=2)
    ax.set_title(f"Figure G. Detection timeline ({best_result['metrics']['model']})",
                 fontweight="bold", color=NAVY)
    ax.set_xlabel("Test sample index"); ax.set_ylabel(sensor)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig_timeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

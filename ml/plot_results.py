"""Plot comparison figures from the dashboard JSON: python ml/plot_results.py

Writes PNGs to Docs/figures/.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import CONFIG, DASHBOARD_DATA_DIR, REPO_DIR
from models import MODELS

OUT_DIR = REPO_DIR / "Docs" / "figures"
COLORS = {"densenet": "#2a78d6", "mobilenet": "#eb6834"}  # validated categorical slots 1-2
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a63", "#e6e5e0", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def load():
    data = {}
    for name, info in MODELS.items():
        data[name] = {
            "label": info["display_name"],
            "metrics": json.loads((DASHBOARD_DATA_DIR / info["metrics_file"]).read_text()),
            "history": json.loads((DASHBOARD_DATA_DIR / info["history_file"]).read_text()),
        }
    return data


def learning_curves(data):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    phase_end = CONFIG["phase1_epochs"] - 0.5
    for ax, key, title in ((axes[0], "loss", "Loss (lower is better)"), (axes[1], "acc", "Accuracy")):
        for name, d in data.items():
            h = d["history"]
            epochs = np.arange(1, len(h[f"train_{key}"]) + 1)
            ax.plot(epochs, h[f"train_{key}"], color=COLORS[name], lw=2, label=f"{d['label']} train")
            ax.plot(epochs, h[f"val_{key}"], color=COLORS[name], lw=2, ls="--", label=f"{d['label']} val")
        ax.axvline(phase_end + 1, color=MUTED, lw=1, ls=":")
        ax.text(phase_end + 1.3, ax.get_ylim()[1], "phase 2: fine-tune all", color=MUTED, va="top", fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        if key == "acc":
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[1].legend(frameon=False, loc="lower right", fontsize=9)
    fig.suptitle("Learning curves (selected seed of each model)", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def test_metrics(data):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    metrics = [("accuracy", "Accuracy"), ("f1_score", "F1 score"), ("precision", "Precision")]
    x = np.arange(len(metrics))
    width = 0.36
    for i, (name, d) in enumerate(data.items()):
        m = d["metrics"]
        offset = (i - 0.5) * width
        means = [m[k] for k, _ in metrics]
        stds = [m.get(f"{k}_std", 0) for k, _ in metrics]
        ax.bar(x + offset, means, width * 0.94, color=COLORS[name], label=d["label"], zorder=2)
        ax.errorbar(x + offset, means, yerr=stds, fmt="none", ecolor=INK, capsize=4, lw=1.2, zorder=3)
        for j, (k, _) in enumerate(metrics):
            seeds = [s[k] for s in m["seeds"]]
            ax.scatter([x[j] + offset] * len(seeds), seeds, s=22, color=SURFACE, edgecolor=INK, lw=1, zorder=4)
            ax.text(x[j] + offset, means[j] + stds[j] + 0.015, f"{means[j]:.1%}", ha="center", fontsize=10, color=INK)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylim(0.5, 0.85)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(f"Test set (n={data['densenet']['metrics']['test_samples']}): mean ± SD over 3 seeds, dots = each seed")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return fig


def confusion_matrices(data):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, (name, d) in zip(axes, data.items()):
        m = d["metrics"]
        cm = np.array(m["confusion_matrix"])
        classes = m["classes"]
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum(axis=1).max())
        ax.grid(False)
        for r in range(2):
            for c in range(2):
                share = cm[r, c] / cm[r].sum()
                ax.text(c, r, f"{cm[r, c]}\n{share:.0%}", ha="center", va="center", fontsize=13,
                        color="white" if share > 0.5 else INK)
        ax.set_xticks([0, 1], classes)
        ax.set_yticks([0, 1], classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{d['label']} (seed {m['selected_seed']})")
    fig.suptitle("Confusion matrix on the test set. Row % = recall per class", x=0.01, ha="left",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def efficiency(data):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    panels = [
        ("Latency per image (ms)", lambda m: float(m["inference"].rstrip("ms"))),
        ("Weights file (MB)", lambda m: float(m["size"].rstrip("MB"))),
        ("Parameters (millions)", lambda m: float(m["params"].rstrip("M"))),
    ]
    for ax, (title, get) in zip(axes, panels):
        names = list(data)
        values = [get(data[n]["metrics"]) for n in names]
        ax.bar([data[n]["label"] for n in names], values, color=[COLORS[n] for n in names], width=0.55, zorder=2)
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:g}", ha="center", va="bottom", fontsize=11, color=INK)
        ax.set_title(title)
        ax.margins(y=0.15)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Speed and size (lower is better)", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load()
    for filename, make in (("learning_curves.png", learning_curves), ("test_metrics.png", test_metrics),
                           ("confusion_matrices.png", confusion_matrices), ("efficiency.png", efficiency)):
        fig = make(data)
        fig.savefig(OUT_DIR / filename, dpi=150)
        plt.close(fig)
        print(OUT_DIR / filename)

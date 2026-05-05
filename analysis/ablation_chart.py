"""
ablation_chart.py — Ablation Bar Chart Generator
=================================================
Reads master_results.csv (written by ResultsLogger after each experiment run)
and generates a horizontal bar chart comparing mean IoU ± std across E0–E9.

CSV format expected (columns):
    experiment, mean_iou, std_iou, scores

Output:  <figures_dir>/ablation_chart.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths (match BaseConfig defaults)
# ---------------------------------------------------------------------------
RESULTS_DIR = "/content/drive/MyDrive/lane_detection_data/outputs/results"
FIGURES_DIR = "/content/drive/MyDrive/lane_detection_data/outputs/figures"

# Display order + labels
EXP_ORDER = [
    "e0_baseline",
    "e1_optimizer",
    "e2_activation",
    "e3_regularization",
    "e4_augmentation",
    "e5_batchnorm",
    "e6_architecture",
    "e7_transfer",
    "e8_loss",
    "e9_ablation",
]

LABELS = {
    "e0_baseline":       "E0  Baseline CNN",
    "e1_optimizer":      "E1  Adam optimizer",
    "e2_activation":     "E2  GELU activation",
    "e3_regularization": "E3  Stronger regularisation",
    "e4_augmentation":   "E4  Heavy augmentation",
    "e5_batchnorm":      "E5  No BatchNorm",
    "e6_architecture":   "E6  U-Net scratch",
    "e7_transfer":       "E7  Transfer learning ★",
    "e8_loss":           "E8  DiceBCE loss",
    "e9_ablation":       "E9  Ablation (no aug)",
}

# Colour: CNN experiments in blue family, U-Net in green family
COLORS = {
    "e0_baseline":       "#1e40af",
    "e1_optimizer":      "#2563eb",
    "e2_activation":     "#3b82f6",
    "e3_regularization": "#60a5fa",
    "e4_augmentation":   "#93c5fd",
    "e5_batchnorm":      "#bfdbfe",
    "e6_architecture":   "#065f46",
    "e7_transfer":       "#059669",
    "e8_loss":           "#10b981",
    "e9_ablation":       "#6ee7b7",
}


def create_ablation_chart(
    results_dir: str = RESULTS_DIR,
    figures_dir: str = FIGURES_DIR,
) -> None:
    """Generate the ablation comparison bar chart from master_results.csv."""
    print("\n[ablation_chart] Generating ablation chart...")

    csv_path = Path(results_dir) / "master_results.csv"
    if not csv_path.exists():
        print(f"  [SKIP] master_results.csv not found at {csv_path}")
        print("         Run all experiments first, then re-run analysis.")
        return

    df = pd.read_csv(csv_path)
    # Keep only the last entry per experiment (in case of reruns)
    df = df.drop_duplicates(subset="experiment", keep="last")

    # Reorder to match EXP_ORDER (only include experiments that have results)
    present = [e for e in EXP_ORDER if e in df["experiment"].values]
    df = df.set_index("experiment").reindex(present).reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))

    y_pos = np.arange(len(df))
    colors = [COLORS.get(e, "#94a3b8") for e in df["experiment"]]

    bars = ax.barh(
        y_pos,
        df["mean_iou"],
        xerr=df["std_iou"],
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        capsize=4,
        error_kw={"elinewidth": 1.5, "ecolor": "#374151"},
        height=0.65,
    )

    # Value labels on bars
    for bar, val, std in zip(bars, df["mean_iou"], df["std_iou"]):
        ax.text(
            bar.get_width() + std + 0.004,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", ha="left", fontsize=9, color="#111827",
        )

    # Baseline reference line
    if "e0_baseline" in df["experiment"].values:
        baseline = df.loc[df["experiment"] == "e0_baseline", "mean_iou"].values[0]
        ax.axvline(baseline, color="#ef4444", linestyle="--", linewidth=1.2,
                   label=f"E0 baseline ({baseline:.4f})")
        ax.legend(fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([LABELS.get(e, e) for e in df["experiment"]], fontsize=10)
    ax.set_xlabel("Mean Val IoU (3-seed average)", fontsize=11)
    ax.set_title("Ablation Study: E0–E9 Experiment Comparison", fontsize=13, fontweight="bold")
    ax.set_xlim(0, min(df["mean_iou"].max() * 1.25, 1.0))
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # Legend patches for colour groups
    cnn_patch   = mpatches.Patch(color="#3b82f6", label="CNN experiments (E0–E5)")
    unet_patch  = mpatches.Patch(color="#059669", label="U-Net experiments (E6–E9)")
    ax.legend(handles=[cnn_patch, unet_patch], loc="lower right", fontsize=9)

    plt.tight_layout()
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ablation_chart.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved: {out_path}")
    print("[ablation_chart] Done.")


if __name__ == "__main__":
    create_ablation_chart()

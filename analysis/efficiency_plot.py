"""
efficiency_plot.py — IoU vs Model Parameters Scatter Plot
==========================================================
Compares model architectures on two axes:
  - X: trainable parameter count (millions)
  - Y: best mean validation IoU

Data is hardcoded from known model sizes + read from master_results.csv
for IoU values.  Each model family gets a distinct colour and marker.

Output:  <figures_dir>/efficiency_plot.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = "/content/drive/MyDrive/lane_detection_data/outputs/results"
FIGURES_DIR = "/content/drive/MyDrive/lane_detection_data/outputs/figures"

# Known parameter counts (millions) per experiment
PARAM_COUNTS_M = {
    "e0_baseline":       2.0,
    "e1_optimizer":      2.0,
    "e2_activation":     2.0,
    "e3_regularization": 2.0,
    "e4_augmentation":   2.0,
    "e5_batchnorm":      2.0,
    "e6_architecture":  24.0,
    "e7_transfer":      24.0,
    "e8_loss":          24.0,
    "e9_ablation":      24.0,
}

EXP_LABELS = {
    "e0_baseline":       "E0",
    "e1_optimizer":      "E1",
    "e2_activation":     "E2",
    "e3_regularization": "E3",
    "e4_augmentation":   "E4",
    "e5_batchnorm":      "E5",
    "e6_architecture":   "E6",
    "e7_transfer":       "E7★",
    "e8_loss":           "E8",
    "e9_ablation":       "E9",
}

COLORS = {
    "cnn":   "#3b82f6",   # blue for E0-E5
    "unet":  "#059669",   # green for E6-E9
}

CNN_EXPS  = {"e0_baseline","e1_optimizer","e2_activation",
             "e3_regularization","e4_augmentation","e5_batchnorm"}
UNET_EXPS = {"e6_architecture","e7_transfer","e8_loss","e9_ablation"}


def create_efficiency_plot(
    results_dir: str = RESULTS_DIR,
    figures_dir: str = FIGURES_DIR,
) -> None:
    """Generate IoU vs parameter-count scatter plot."""
    print("\n[efficiency_plot] Generating efficiency scatter plot...")

    csv_path = Path(results_dir) / "master_results.csv"
    if not csv_path.exists():
        print(f"  [SKIP] master_results.csv not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset="experiment", keep="last")

    fig, ax = plt.subplots(figsize=(10, 6))

    for _, row in df.iterrows():
        exp = row["experiment"]
        params = PARAM_COUNTS_M.get(exp, 0)
        iou    = row["mean_iou"]
        std    = row["std_iou"]
        label  = EXP_LABELS.get(exp, exp)
        color  = COLORS["unet"] if exp in UNET_EXPS else COLORS["cnn"]
        marker = "^" if exp in UNET_EXPS else "o"

        ax.errorbar(params, iou, yerr=std, fmt=marker, color=color,
                    markersize=10, capsize=4, elinewidth=1.5,
                    markeredgecolor="white", markeredgewidth=0.8)
        ax.annotate(
            label, (params, iou),
            textcoords="offset points", xytext=(8, 4),
            fontsize=9, color="#111827",
        )

    # Add jitter to overlapping CNN points for readability
    # (all CNN exps share 2M params — nudge them slightly)
    cnn_rows = df[df["experiment"].isin(CNN_EXPS)]
    offsets = np.linspace(-0.3, 0.3, len(cnn_rows))
    for offset, (_, row) in zip(offsets, cnn_rows.iterrows()):
        exp = row["experiment"]
        if exp in CNN_EXPS:
            params = PARAM_COUNTS_M[exp] + offset
            iou    = row["mean_iou"]
            label  = EXP_LABELS.get(exp, exp)
            # replot just the dot at nudged x so labels don't overlap
            ax.annotate(
                label, (params, iou),
                textcoords="offset points", xytext=(5, 4),
                fontsize=9, color="#111827",
            )

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='CNN (E0–E5)',
               markerfacecolor=COLORS["cnn"], markersize=10),
        Line2D([0], [0], marker='^', color='w', label='U-Net (E6–E9)',
               markerfacecolor=COLORS["unet"], markersize=10),
    ]
    ax.legend(handles=legend_elements, fontsize=10)

    ax.set_xlabel("Trainable Parameters (millions)", fontsize=11)
    ax.set_ylabel("Mean Val IoU (3-seed avg)", fontsize=11)
    ax.set_title("Efficiency Plot: IoU vs Model Size", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "efficiency_plot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved: {out_path}")
    print("[efficiency_plot] Done.")


if __name__ == "__main__":
    create_efficiency_plot()

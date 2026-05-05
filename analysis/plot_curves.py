"""
plot_curves.py — Training & Validation Curve Generator
=======================================================
Reads per-seed CSV logs produced by CSVLogger and generates:
  - Loss curves (train + val) per experiment
  - IoU curves  (train + val) per experiment
  - One PNG per experiment saved to figures_dir

CSV format expected (columns):
    epoch, train_loss, train_iou, val_loss, val_iou, val_auc_iou, val_dice, lr

Output:  <figures_dir>/curves/<exp_name>_curves.png
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (match BaseConfig defaults)
# ---------------------------------------------------------------------------
LOGS_DIR    = "/content/drive/MyDrive/lane_detection_data/outputs/logs"
FIGURES_DIR = "/content/drive/MyDrive/lane_detection_data/outputs/figures"

# All defined experiments (name → display label)
EXPERIMENTS = {
    "e0_baseline":      "E0 – Baseline CNN",
    "e1_optimizer":     "E1 – Optimizer (Adam)",
    "e2_activation":    "E2 – Activation (GELU)",
    "e3_regularization":"E3 – Regularisation",
    "e4_augmentation":  "E4 – Heavy Augmentation",
    "e5_batchnorm":     "E5 – No BatchNorm",
    "e6_architecture":  "E6 – U-Net Scratch",
    "e7_transfer":      "E7 – Transfer Learning",
    "e8_loss":          "E8 – DiceBCE Loss",
    "e9_ablation":      "E9 – Ablation (no aug)",
}

SEEDS = [42, 123, 777]
PALETTE = ["#3b82f6", "#f59e0b", "#10b981"]  # blue, amber, green per seed


def _load_logs(exp_name: str, logs_dir: str) -> list[pd.DataFrame]:
    """Return a list of DataFrames (one per seed) for an experiment."""
    dfs = []
    for seed in SEEDS:
        path = Path(logs_dir) / f"{exp_name}_s{seed}.csv"
        if path.exists():
            dfs.append(pd.read_csv(path))
    return dfs


def _plot_experiment(exp_name: str, label: str, logs_dir: str, figures_dir: str) -> None:
    dfs = _load_logs(exp_name, logs_dir)
    if not dfs:
        print(f"  [SKIP] No logs found for {exp_name}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(label, fontsize=14, fontweight="bold", y=1.02)

    for ax, (y_train, y_val, ylabel) in zip(
        axes,
        [
            ("train_loss", "val_loss", "Loss"),
            ("train_iou",  "val_iou",  "IoU"),
        ],
    ):
        for i, (df, seed) in enumerate(zip(dfs, SEEDS)):
            if y_train in df.columns:
                ax.plot(df["epoch"], df[y_train], color=PALETTE[i],
                        linestyle="--", alpha=0.7, label=f"Train s{seed}")
            if y_val in df.columns:
                ax.plot(df["epoch"], df[y_val], color=PALETTE[i],
                        linestyle="-",  alpha=0.9, label=f"Val   s{seed}")

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{ylabel} Curves", fontsize=12)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_dir = Path(figures_dir) / "curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{exp_name}_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved: {out_path}")


def plot_all_curves(
    logs_dir: str = LOGS_DIR,
    figures_dir: str = FIGURES_DIR,
) -> None:
    """Generate training/validation curve PNGs for every completed experiment."""
    print("\n[plot_curves] Generating training curves...")
    for exp_name, label in EXPERIMENTS.items():
        _plot_experiment(exp_name, label, logs_dir, figures_dir)
    print("[plot_curves] Done.")


if __name__ == "__main__":
    plot_all_curves()

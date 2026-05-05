"""
results_table.py — Master Results Table Generator
==================================================
Reads master_results.csv and produces:
  1. A formatted console/text table
  2. A styled PNG table image suitable for the project report

CSV format expected (columns):
    experiment, mean_iou, std_iou, scores

Output:  <figures_dir>/results_table.png
         <results_dir>/results_summary.txt
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths (match BaseConfig defaults)
# ---------------------------------------------------------------------------
RESULTS_DIR = "/content/drive/MyDrive/lane_detection_data/outputs/results"
FIGURES_DIR = "/content/drive/MyDrive/lane_detection_data/outputs/figures"

EXP_LABELS = {
    "e0_baseline":       "E0 – Baseline CNN",
    "e1_optimizer":      "E1 – Adam Optimizer",
    "e2_activation":     "E2 – GELU Activation",
    "e3_regularization": "E3 – Regularisation",
    "e4_augmentation":   "E4 – Heavy Aug",
    "e5_batchnorm":      "E5 – No BatchNorm",
    "e6_architecture":   "E6 – U-Net Scratch",
    "e7_transfer":       "E7 – Transfer Learning",
    "e8_loss":           "E8 – DiceBCE Loss",
    "e9_ablation":       "E9 – Ablation",
}

EXP_VARIABLE = {
    "e0_baseline":       "—  (baseline)",
    "e1_optimizer":      "optimizer_name = adam",
    "e2_activation":     "activation = gelu",
    "e3_regularization": "dropout=0.3, wd=1e-3",
    "e4_augmentation":   "train_augmentation = heavy",
    "e5_batchnorm":      "use_batchnorm = False",
    "e6_architecture":   "model = unet_smp (scratch)",
    "e7_transfer":       "pretrained = True",
    "e8_loss":           "loss_name = dice_bce",
    "e9_ablation":       "train_augmentation = none",
}


def build_results_table(
    results_dir: str = RESULTS_DIR,
    figures_dir: str = FIGURES_DIR,
) -> None:
    """Build and save the master results table (PNG + text)."""
    print("\n[results_table] Building master results table...")

    csv_path = Path(results_dir) / "master_results.csv"
    if not csv_path.exists():
        print(f"  [SKIP] master_results.csv not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset="experiment", keep="last")

    # Enrich table
    df["Label"]    = df["experiment"].map(lambda e: EXP_LABELS.get(e, e))
    df["Variable"] = df["experiment"].map(lambda e: EXP_VARIABLE.get(e, "—"))
    df["Mean IoU"] = df["mean_iou"].map(lambda x: f"{x:.4f}")
    df["Std IoU"]  = df["std_iou"].map(lambda x: f"±{x:.4f}")
    df["vs E0"]    = ""

    # vs-baseline delta
    if "e0_baseline" in df["experiment"].values:
        baseline_iou = df.loc[df["experiment"] == "e0_baseline", "mean_iou"].values[0]
        df["vs E0"] = df["mean_iou"].map(
            lambda x: f"+{x - baseline_iou:.4f}" if x > baseline_iou else f"{x - baseline_iou:.4f}"
        )

    display_cols = ["Label", "Variable", "Mean IoU", "Std IoU", "vs E0"]
    table_df = df[display_cols].reset_index(drop=True)

    # ── Text summary ──────────────────────────────────────────────────────
    txt_path = Path(results_dir) / "results_summary.txt"
    with open(txt_path, "w") as f:
        f.write("Ultra Lane Detection — Master Results\n")
        f.write("=" * 70 + "\n")
        f.write(table_df.to_string(index=False))
        f.write("\n\n")
        best_row = df.loc[df["mean_iou"].idxmax()]
        f.write(f"Best experiment : {best_row['Label']}\n")
        f.write(f"Best mean IoU   : {best_row['mean_iou']:.4f} ± {best_row['std_iou']:.4f}\n")
    print(f"  [OK] Text summary: {txt_path}")
    print(table_df.to_string(index=False))

    # ── PNG table ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, max(4, len(table_df) * 0.55 + 1.5)))
    ax.axis("off")

    tbl = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)

    # Header styling
    for j in range(len(display_cols)):
        tbl[(0, j)].set_facecolor("#1e3a5f")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Row alternating colours + highlight best
    best_idx = df["mean_iou"].idxmax()
    for i in range(1, len(table_df) + 1):
        row_color = "#f0f7ff" if i % 2 == 0 else "white"
        if i - 1 == best_idx:
            row_color = "#d1fae5"  # green highlight for best
        for j in range(len(display_cols)):
            tbl[(i, j)].set_facecolor(row_color)

    fig.suptitle("Ultra Lane Detection — Experiment Results", fontsize=13,
                 fontweight="bold", y=0.98)
    plt.tight_layout()

    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results_table.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] PNG table: {out_path}")
    print("[results_table] Done.")


if __name__ == "__main__":
    build_results_table()

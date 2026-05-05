"""
confusion_matrix.py — Pixel-Level Confusion Matrix Generator
=============================================================
Loads the best checkpoint for each experiment, runs inference on the
validation set, and computes a 2×2 pixel-level confusion matrix
(background vs lane).

Outputs one PNG per experiment:
    <figures_dir>/confusion/<exp_name>_confusion.png

Also writes a combined summary figure:
    <figures_dir>/confusion_summary.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CHECKPOINTS_DIR = "/content/drive/MyDrive/lane_detection_data/outputs/checkpoints"
FIGURES_DIR     = "/content/drive/MyDrive/lane_detection_data/outputs/figures"
DATA_ROOT       = "/content/fast_data"

# Experiments to analyse and their model type
EXPERIMENTS = {
    "e0_baseline":      "baseline_cnn",
    "e1_optimizer":     "baseline_cnn",
    "e2_activation":    "baseline_cnn",
    "e3_regularization":"baseline_cnn",
    "e4_augmentation":  "baseline_cnn",
    "e5_batchnorm":     "baseline_cnn",
    "e6_architecture":  "unet_smp",
    "e7_transfer":      "unet_smp",
    "e8_loss":          "unet_smp",
    "e9_ablation":      "unet_smp",
}

SEEDS   = [42, 123, 777]
CLASSES = ["Background", "Lane"]


def _plot_cm(cm: np.ndarray, title: str, ax: plt.Axes) -> None:
    """Draw a 2×2 confusion matrix on the given axes."""
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(CLASSES, fontsize=9)
    ax.set_yticks([0, 1]); ax.set_yticklabels(CLASSES, fontsize=9, rotation=90, va="center")
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual",    fontsize=9)

    total = cm.sum()
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct   = 100 * count / total if total > 0 else 0
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{count:,}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=8, color=color)


@torch.no_grad()
def _compute_cm_for_experiment(
    exp_name: str,
    model_name: str,
    device: torch.device,
) -> np.ndarray | None:
    """Load best checkpoint + val loader and accumulate pixel confusion matrix."""
    try:
        from configs.cnn_config import CNNConfig
        from configs.unet_config import UNetConfig
        from data.fast_dataset import build_fast_dataloaders
        from models import get_model
        from training.checkpoint import load_checkpoint

        if model_name == "baseline_cnn":
            cfg = CNNConfig(experiment_name=exp_name, data_root=DATA_ROOT)
        else:
            from dataclasses import replace
            cfg = replace(
                UNetConfig(experiment_name=exp_name, data_root=DATA_ROOT),
                pretrained=False,  # weights loaded from checkpoint, not hub
            )

        # Use seed 42 best checkpoint for confusion matrix
        ckpt_path = Path(CHECKPOINTS_DIR) / f"{exp_name}_s42_best.pth"
        if not ckpt_path.exists():
            print(f"    [SKIP] Checkpoint not found: {ckpt_path}")
            return None

        model = get_model(cfg).to(device)
        load_checkpoint(str(ckpt_path), model, optimizer=None, scheduler=None)
        model.eval()

        _, val_loader = build_fast_dataloaders(cfg)

        cm = np.zeros((2, 2), dtype=np.int64)
        for images, masks in val_loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            preds  = (torch.sigmoid(logits) > 0.5).cpu().long().squeeze(1)
            masks  = masks.cpu().long().squeeze(1)
            for pred, mask in zip(preds, masks):
                flat_pred = pred.view(-1).numpy()
                flat_mask = mask.view(-1).numpy()
                for act, pre in zip(flat_mask, flat_pred):
                    cm[act, pre] += 1

        return cm

    except Exception as exc:
        print(f"    [ERROR] {exp_name}: {exc}")
        return None


def build_confusion_matrix(
    figures_dir: str = FIGURES_DIR,
) -> None:
    """Generate confusion matrix PNGs for all completed experiments."""
    print("\n[confusion_matrix] Generating confusion matrices...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    out_dir = Path(figures_dir) / "confusion"
    out_dir.mkdir(parents=True, exist_ok=True)

    cms: dict[str, np.ndarray] = {}

    for exp_name, model_name in EXPERIMENTS.items():
        print(f"  Processing {exp_name}...")
        cm = _compute_cm_for_experiment(exp_name, model_name, device)
        if cm is None:
            continue
        cms[exp_name] = cm

        # Per-experiment figure
        fig, ax = plt.subplots(figsize=(5, 4))
        _plot_cm(cm, exp_name, ax)
        plt.tight_layout()
        fig.savefig(out_dir / f"{exp_name}_confusion.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    [OK] {exp_name}")

    # Combined summary figure
    if cms:
        n = len(cms)
        cols = min(5, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
        axes = np.array(axes).flatten()
        for ax_i, (exp_name, cm) in enumerate(cms.items()):
            _plot_cm(cm, exp_name, axes[ax_i])
        for ax_i in range(len(cms), len(axes)):
            axes[ax_i].set_visible(False)
        fig.suptitle("Pixel-Level Confusion Matrices — All Experiments",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        out_path = Path(figures_dir) / "confusion_summary.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [OK] Summary: {out_path}")

    print("[confusion_matrix] Done.")


if __name__ == "__main__":
    build_confusion_matrix()

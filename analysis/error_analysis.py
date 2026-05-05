"""
error_analysis.py — Failure Gallery Generator
==============================================
Loads the best checkpoint for a given experiment (default: E7 transfer learning),
runs inference on the validation set, ranks predictions by IoU, and saves:
  - A "failure gallery": worst N predictions (lowest IoU)
  - A "success gallery": best N predictions  (highest IoU)

Each gallery is a grid of triplets:  [Input Image | Ground Truth | Prediction]

Output:
    <figures_dir>/error_analysis/failures_<exp_name>.png
    <figures_dir>/error_analysis/successes_<exp_name>.png
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CHECKPOINTS_DIR = "/content/drive/MyDrive/lane_detection_data/outputs/checkpoints"
FIGURES_DIR     = "/content/drive/MyDrive/lane_detection_data/outputs/figures"
DATA_ROOT       = "/content/fast_data"

# ImageNet denormalisation constants
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])


def _denorm(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalised image tensor (C,H,W) → RGB uint8 numpy array."""
    img = tensor.cpu().permute(1, 2, 0).numpy()
    img = img * STD + MEAN
    return np.clip(img, 0, 1)


def _iou_per_image(logits: torch.Tensor, masks: torch.Tensor,
                   threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """Return per-image IoU scores (shape: [B])."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    masks = masks.float()
    inter = (preds * masks).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) - inter
    return (inter + eps) / (union + eps)


def _make_gallery(
    samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    title: str,
    out_path: Path,
    n: int = 8,
) -> None:
    """Save a grid of [image | GT mask | predicted mask] triplets."""
    samples = samples[:n]
    cols = 3
    rows = len(samples)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.0))
    if rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input Image", "Ground Truth", "Prediction"]
    for j, ct in enumerate(col_titles):
        axes[0, j].set_title(ct, fontsize=11, fontweight="bold")

    for i, (img, gt, pred, iou) in enumerate(samples):
        axes[i, 0].imshow(img)
        axes[i, 0].set_ylabel(f"IoU={iou:.3f}", fontsize=9, rotation=0,
                               labelpad=50, va="center")
        axes[i, 1].imshow(gt.squeeze(),  cmap="gray", vmin=0, vmax=1)
        axes[i, 2].imshow(pred.squeeze(), cmap="gray", vmin=0, vmax=1)
        for ax in axes[i]:
            ax.axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path}")


@torch.no_grad()
def generate_failure_gallery(
    exp_name:    str = "e7_transfer",
    model_name:  str = "unet_smp",
    seed:        int = 42,
    n_samples:   int = 8,
    figures_dir: str = FIGURES_DIR,
) -> None:
    """
    Generate failure and success galleries for the given experiment.
    Defaults to E7 (best model) but any completed experiment works.
    """
    print(f"\n[error_analysis] Generating galleries for {exp_name}...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

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
                pretrained=False,
            )

        ckpt_path = Path(CHECKPOINTS_DIR) / f"{exp_name}_s{seed}_best.pth"
        if not ckpt_path.exists():
            print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
            return

        model = get_model(cfg).to(device)
        load_checkpoint(str(ckpt_path), model, optimizer=None, scheduler=None)
        model.eval()

        _, val_loader = build_fast_dataloaders(cfg)

        # Collect all samples with per-image IoU
        all_samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []

        for images, masks in val_loader:
            images_dev = images.to(device, non_blocking=True)
            logits = model(images_dev)
            ious   = _iou_per_image(logits, masks.to(device)).cpu()
            preds  = (torch.sigmoid(logits) > 0.5).float().cpu()

            for img_t, gt_t, pred_t, iou_val in zip(images, masks, preds, ious):
                img_np  = _denorm(img_t)
                gt_np   = gt_t.numpy()
                pred_np = pred_t.numpy()
                all_samples.append((img_np, gt_np, pred_np, iou_val.item()))

        if not all_samples:
            print("  [SKIP] No samples found in val loader.")
            return

        # Sort by IoU
        all_samples.sort(key=lambda x: x[3])
        failures  = all_samples[:n_samples]          # lowest IoU
        successes = all_samples[-n_samples:][::-1]   # highest IoU

        out_dir = Path(figures_dir) / "error_analysis"

        _make_gallery(
            failures,
            title=f"Failure Gallery — {exp_name} (worst {n_samples} by IoU)",
            out_path=out_dir / f"failures_{exp_name}.png",
            n=n_samples,
        )
        _make_gallery(
            successes,
            title=f"Success Gallery — {exp_name} (best {n_samples} by IoU)",
            out_path=out_dir / f"successes_{exp_name}.png",
            n=n_samples,
        )

    except Exception as exc:
        print(f"  [ERROR] {exc}")
        raise

    print("[error_analysis] Done.")


if __name__ == "__main__":
    # Generate galleries for best model (E7) by default
    generate_failure_gallery(exp_name="e7_transfer", model_name="unet_smp")
    # Also generate for baseline for comparison
    generate_failure_gallery(exp_name="e0_baseline", model_name="baseline_cnn")

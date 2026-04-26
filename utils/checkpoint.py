from pathlib import Path
import torch
import torch.nn as nn


def save_checkpoint(state: dict, filepath: str) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(filepath: str, model: nn.Module, optimizer=None) -> dict:
    checkpoint = torch.load(filepath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return {
        "epoch": checkpoint.get("epoch", 0),
        "best_val_iou": checkpoint.get("best_val_iou", 0.0),
        "experiment_name": checkpoint.get("experiment_name", ""),
    }

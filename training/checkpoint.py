from pathlib import Path

import torch


def save_checkpoint(path: str, model, optimizer, scheduler, epoch: int, best_iou: float) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_iou": best_iou,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
        },
        path,
    )


def load_checkpoint(path: str, model, optimizer=None, scheduler=None):
    if not Path(path).exists():
        return 0, -1.0
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"])
    if optimizer and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler and payload.get("scheduler"):
        scheduler.load_state_dict(payload["scheduler"])
    return payload.get("epoch", 0), payload.get("best_iou", 0.0)

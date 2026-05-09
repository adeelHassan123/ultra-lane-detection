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
    state_dict = payload["model"]

    # Handle torch.compile() prefix mismatch
    # Case 1: Checkpoint has "_orig_mod." but model doesn't expect it → strip it
    # Case 2: Model expects "_orig_mod." (compiled) but checkpoint doesn't have it → add it
    has_orig_mod_in_ckpt = any(k.startswith("_orig_mod.") for k in state_dict.keys())
    model_is_compiled = hasattr(model, "_orig_mod")

    if has_orig_mod_in_ckpt and not model_is_compiled:
        # Strip prefix: compiled checkpoint → non-compiled model
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    elif not has_orig_mod_in_ckpt and model_is_compiled:
        # Add prefix: non-compiled checkpoint → compiled model
        state_dict = {"_orig_mod." + k: v for k, v in state_dict.items()}
    # else: keys already match, no transformation needed

    model.load_state_dict(state_dict, strict=True)
    if optimizer and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler and payload.get("scheduler"):
        scheduler.load_state_dict(payload["scheduler"])
    return payload.get("epoch", 0), payload.get("best_iou", 0.0)

import torch
from tqdm import tqdm

from metrics.segmentation import iou_score


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, cfg):
    model.train()
    losses = []
    ious = []
    pbar = tqdm(loader, desc="train", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        ious.append(iou_score(logits.detach(), masks.detach(), threshold=cfg.threshold))
        pbar.set_postfix(loss=f"{losses[-1]:.4f}", iou=f"{ious[-1]:.4f}")
    return sum(losses) / len(losses), sum(ious) / len(ious)

import torch
from tqdm import tqdm

from metrics.segmentation import iou_score


@torch.no_grad()
def validate(model, loader, criterion, device, cfg):
    model.eval()
    losses = []
    ious = []
    for images, masks in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, masks)
        losses.append(loss.item())
        ious.append(iou_score(logits, masks, threshold=cfg.threshold))
    return sum(losses) / len(losses), sum(ious) / len(ious)

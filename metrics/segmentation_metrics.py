import torch


def iou_score(preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def dice_score(preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return (2 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)


def pixel_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    correct = (preds == targets).sum().float()
    return correct / targets.numel()


def compute_metrics(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> dict:
    with torch.no_grad():
        probs = torch.sigmoid(logits.detach().cpu())
        targets = targets.detach().cpu()
        preds = (probs >= threshold).float()

        return {
            "iou": iou_score(preds, targets).item(),
            "dice": dice_score(preds, targets).item(),
            "pixel_acc": pixel_accuracy(preds, targets).item(),
        }

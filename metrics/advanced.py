import torch
from sklearn.metrics import precision_recall_fscore_support


def auc_iou(logits: torch.Tensor, targets: torch.Tensor, thresholds=None) -> float:
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    probs = torch.sigmoid(logits)
    scores = []
    for threshold in thresholds:
        preds = (probs > threshold).float()
        inter = (preds * targets).sum()
        union = preds.sum() + targets.sum() - inter
        scores.append(float((inter + 1e-6) / (union + 1e-6)))
    return sum(scores) / len(scores)


def precision_recall_f1(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).detach().cpu().numpy().astype(np.uint8).reshape(-1)
    y_true = targets.detach().cpu().numpy().astype(np.uint8).reshape(-1)
    p, r, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    return float(p), float(r), float(f1)

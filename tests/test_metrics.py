import torch
import pytest
from metrics.segmentation import iou_score, dice_score, pixel_accuracy

def test_metrics_perfect():
    pred_logits = torch.ones(2, 1, 64, 64) * 10.0 # High logits -> prob ~1.0
    target = torch.ones(2, 1, 64, 64)
    
    iou = iou_score(pred_logits, target)
    dice = dice_score(pred_logits, target)
    acc = pixel_accuracy(pred_logits, target)
    
    assert iou == pytest.approx(1.0, abs=1e-4)
    assert dice == pytest.approx(1.0, abs=1e-4)
    assert acc == pytest.approx(1.0, abs=1e-4)

def test_metrics_wrong():
    pred_logits = torch.ones(2, 1, 64, 64) * -10.0 # Low logits -> prob ~0.0
    target = torch.ones(2, 1, 64, 64)
    
    iou = iou_score(pred_logits, target)
    dice = dice_score(pred_logits, target)
    acc = pixel_accuracy(pred_logits, target)
    
    assert iou == pytest.approx(0.0, abs=1e-4)
    assert dice == pytest.approx(0.0, abs=1e-4)
    assert acc == pytest.approx(0.0, abs=1e-4)
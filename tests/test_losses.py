import torch
import pytest
from losses import get_loss
from dataclasses import dataclass

@dataclass
class DummyConfig:
    loss_name: str
    positive_weight: float = 1.0
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    tversky_smooth: float = 1e-6
    tversky_weight: float = 0.9

def test_tversky_loss():
    cfg = DummyConfig(loss_name='tversky')
    tversky = get_loss(cfg)
    logits = torch.randn(2, 1, 64, 64)
    masks = torch.randint(0, 2, (2, 1, 64, 64)).float()
    loss = tversky(logits, masks)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0

def test_combined_loss():
    cfg = DummyConfig(loss_name='combined', positive_weight=10.0)
    combined = get_loss(cfg)
    logits = torch.randn(2, 1, 64, 64)
    masks = torch.randint(0, 2, (2, 1, 64, 64)).float()
    loss = combined(logits, masks)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0

def test_dice_bce_loss():
    cfg = DummyConfig(loss_name='dice_bce')
    dice_bce = get_loss(cfg)
    logits = torch.randn(2, 1, 64, 64)
    masks = torch.randint(0, 2, (2, 1, 64, 64)).float()
    loss = dice_bce(logits, masks)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0
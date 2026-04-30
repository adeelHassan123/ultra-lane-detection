from pathlib import Path

from configs.cnn_config import CNNConfig
from models import get_model
from training.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_save_load(tmp_path: Path):
    cfg = CNNConfig()
    model = get_model(cfg)
    path = tmp_path / "ckpt.pth"
    save_checkpoint(str(path), model, optimizer=None, scheduler=None, epoch=3, best_iou=0.55)
    epoch, best = load_checkpoint(str(path), model)
    assert epoch == 3
    assert best == 0.55

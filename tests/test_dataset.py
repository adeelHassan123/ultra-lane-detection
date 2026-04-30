from pathlib import Path

import cv2
import numpy as np

from data.dataset import TuSimpleLaneDataset


def test_dataset_shape_and_range(tmp_path: Path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    cv2.imwrite(str(images / "sample.png"), img)
    cv2.imwrite(str(masks / "sample.png"), mask)
    ds = TuSimpleLaneDataset(str(images), str(masks))
    x, y = ds[0]
    assert x.shape[0] == 3
    assert y.shape[0] == 1
    assert y.max().item() <= 1.0

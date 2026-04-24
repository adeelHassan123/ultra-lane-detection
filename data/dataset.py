from pathlib import Path
from typing import Callable, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class TuSimpleLaneDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, transforms: Callable = None) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transforms = transforms
        self.samples: List[Tuple[Path, Path]] = self._index_samples()

    def _index_samples(self) -> List[Tuple[Path, Path]]:
        image_paths = sorted(self.images_dir.glob("*.jpg")) + sorted(self.images_dir.glob("*.png"))
        samples = []
        for img_path in image_paths:
            mask_path = self.masks_dir / f"{img_path.stem}.png"
            if mask_path.exists():
                samples.append((img_path, mask_path))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index]
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).float()

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return image, mask

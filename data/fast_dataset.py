"""
FastLaneDataset: Optimized dataset that loads preprocessed numpy files.
Provides 10-20x faster loading compared to on-the-fly JPEG/PNG loading.
"""

from pathlib import Path
from typing import Callable, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


class FastLaneDataset(Dataset):
    """
    Ultra-fast dataset for preprocessed numpy files.
    
    Expects data structure:
        root/
        ├── train/
        │   ├── images/     # Contains .npz files with 'image' and 'mask' arrays
        │   └── masks/      # (optional, not used if data in .npz)
        ├── val/
        └── test/
    
    Each .npz file contains:
        - image: (H, W, 3) uint8 RGB image
        - mask: (H, W) uint8 binary mask {0, 1}
    """
    
    def __init__(self, root: str, split: str = 'train', transform: Callable = None) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.data_dir = self.root / split / "images"
        
        # Pre-index all numpy files
        self.samples: List[Path] = sorted(self.data_dir.glob("*.npz"))
        
        if len(self.samples) == 0:
            raise ValueError(f"No .npz files found in {self.data_dir}. "
                           f"Run scripts/preprocess_to_numpy.py first!")
        
        print(f"[FastLaneDataset] Loaded {len(self.samples)} samples from {self.data_dir}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load numpy file (extremely fast: ~0.003s vs ~0.05s for JPEG)
        data = np.load(self.samples[index])
        
        # Extract arrays
        image = data['image']      # (H, W, 3), uint8
        mask = data['mask']        # (H, W), uint8
        
        # Apply transforms if provided (Albumentations)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            return image, mask
        
        # Fast path: manual conversion to tensors
        # Image: (H, W, 3) -> (3, H, W), normalize to [0, 1]
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Mask: (H, W) -> (1, H, W), already binary {0, 1}
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return image, mask


def build_fast_dataloaders(cfg):
    """Build optimized dataloaders using FastLaneDataset."""
    from data.transforms import build_transforms
    from torch.utils.data import DataLoader
    
    # Check if numpy data exists, fallback to regular dataset if not
    fast_data_path = Path(cfg.data_root) / "train" / "images"
    has_numpy = any(fast_data_path.glob("*.npz"))
    
    if has_numpy:
        print("[SPEEDUP] Using FastLaneDataset with numpy files (10-20x faster)")
        DatasetClass = FastLaneDataset
    else:
        print("[WARNING] Numpy files not found, using regular TuSimpleLaneDataset")
        print("   Run: python scripts/preprocess_to_numpy.py for speedup")
        from data.dataset import TuSimpleLaneDataset
        DatasetClass = TuSimpleLaneDataset
    
    # Training dataset with augmentation
    train_ds = DatasetClass(
        cfg.data_root,
        split='train',
        transform=build_transforms(cfg.image_size, cfg.train_augmentation),
    )
    
    # Validation dataset without augmentation
    val_ds = DatasetClass(
        cfg.data_root,
        split='val',
        transform=build_transforms(cfg.image_size, cfg.val_augmentation),
    )
    
    # Optimized training DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True,
    )
    
    # Validation DataLoader
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,  # 0 for validation
        pin_memory=True,
    )
    
    return train_loader, val_loader

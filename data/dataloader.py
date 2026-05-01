from torch.utils.data import DataLoader

from data.dataset import TuSimpleLaneDataset
from data.transforms import build_transforms


def build_dataloaders(cfg):
    train_ds = TuSimpleLaneDataset(
        cfg.train_images_dir,
        cfg.train_masks_dir,
        transforms=build_transforms(cfg.image_size, cfg.train_augmentation),
    )
    val_ds = TuSimpleLaneDataset(
        cfg.val_images_dir,
        cfg.val_masks_dir,
        transforms=build_transforms(cfg.image_size, cfg.val_augmentation),
    )
    # Optimized training DataLoader with maximum performance settings
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=4,      # Preload 4 batches ahead
        drop_last=True,          # Consistent batch sizes for speed
    )
    # Validation uses num_workers=0 for deterministic behavior
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,           # 0 for validation (deterministic, faster startup)
        pin_memory=True,
    )
    return train_loader, val_loader

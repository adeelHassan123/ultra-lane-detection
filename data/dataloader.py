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
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader

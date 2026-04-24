# `data/` Guide

This folder handles dataset preparation and loading.

## Files

- `preprocess.py`: one-time preprocessing utilities:
  - copy source images
  - dilate thin lane masks
- `validate.py`: strict dataset sanity checks (pairing, shape, binary mask checks).
- `dataset.py`: PyTorch dataset class for image/mask pairs.
- `transforms.py`: Albumentations preprocessing and augmentation pipelines.
- `dataloader.py`: train/val DataLoader factory from config.

## Typical flow

1. `scripts/preprocess_data.py` calls `copy_images` and `dilate_masks`.
2. It then calls `validate_dataset`.
3. During training, dataloaders read processed data through `TuSimpleLaneDataset`.


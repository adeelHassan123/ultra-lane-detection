# `training/` Guide

This folder is the training engine.

## Files

- `trainer.py`: main orchestrator class.
- `train_epoch.py`: one training pass.
- `validate_epoch.py`: one validation pass.
- `checkpoint.py`: save/load full run state.
- `early_stopping.py`: stop training when val metric plateaus.

## Training lifecycle

1. Build model/loss/dataloaders.
2. Build optimizer/scheduler.
3. Optionally resume checkpoint.
4. Train/validate loop per epoch.
5. Log metrics and save best checkpoint.
6. Stop early if no improvement.

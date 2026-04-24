# `configs/` Guide

This folder defines configuration objects for experiments.

## Files

- `base.py`: global defaults for data paths, model/training hyperparameters, and outputs.
- `cnn_config.py`: baseline CNN-oriented defaults.
- `unet_config.py`: U-Net oriented defaults (heavier model, longer training).
- `augmentation_config.py`: augmentation policy metadata.

## How to use

`scripts/run_experiment.py` creates a config and passes it into `run_experiment(cfg)`.

Never hardcode critical training values in training code. Prefer config fields.
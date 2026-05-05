"""
E4: Data Augmentation Comparison
==================================
Variable changed:  train_augmentation  → "heavy"  (vs "light" in E0)
Hypothesis: Aggressive augmentation (motion blur, gaussian noise, higher
            brightness jitter) forces the model to learn texture-invariant
            lane features, improving robustness.
Config change (heavy vs light augmentation):
  Light:  horizontal_flip=0.5, brightness_contrast=0.2
  Heavy:  + motion_blur=0.25, gauss_noise=0.2, brightness_contrast=0.4
Expected outcome: slower convergence but better generalisation (+2–5% IoU).
Baseline (E0) mean IoU: 0.4968
"""

from dataclasses import replace

from configs.cnn_config import CNNConfig


def get_config():
    base = CNNConfig(experiment_name="e4_augmentation")
    return replace(
        base,
        train_augmentation="heavy",     # Changed: "light" → "heavy"
    )

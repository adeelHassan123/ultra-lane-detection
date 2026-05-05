"""
E6: Architecture Comparison — U-Net from Scratch (no pretrained weights)
=========================================================================
Variable changed:  model_name → "unet_smp", pretrained → False
Hypothesis: U-Net's skip connections recover spatial detail lost during
            downsampling, which should significantly help thin lane structures
            vs the baseline CNN encoder-decoder without skip connections.
Expected outcome: +10–15% IoU over E0 CNN, demonstrating skip-connection value.
Architecture: U-Net with ResNet-34 encoder (randomly initialised)
              ~24M parameters | 45 epochs | heavy augmentation
Baseline (E0) mean IoU: 0.4968
"""

from dataclasses import replace

from configs.unet_config import UNetConfig


def get_config():
    base = UNetConfig(experiment_name="e6_architecture")
    return replace(
        base,
        pretrained=False,           # Changed: no ImageNet weights (scratch init)
        train_augmentation="heavy", # Heavy aug to compensate for random init
    )
